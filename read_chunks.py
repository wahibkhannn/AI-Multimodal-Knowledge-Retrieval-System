"""
read_chunks.py — Embedding + Chunking + Qdrant Indexing
Uses SentenceTransformer (BAAI/bge-m3) for true batched local embeddings.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"


import logging
logging.getLogger("transformers.dynamic_module_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import json
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import streamlit as st

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer

import torch
# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
load_dotenv()

QDRANT_CLOUD_URL  = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
TRANSCRIPTS_FOLDER = "transcripts"

TARGET_WORDS  = 150
OVERLAP_WORDS = 30
EMBED_BATCH   = 256       # chunks per encode() call — tune to your RAM/VRAM
EMBED_MODEL   = "BAAI/bge-m3"
VECTOR_DIM    = 1024        # bge-m3 output dimension

# =========================
# MODEL — loaded once at import time
# =========================

# @st.cache_resource
# def load_model():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     log.info(f"Loading embedding model: {EMBED_MODEL} ...")

#     return SentenceTransformer(
#         "BAAI/bge-m3",
#         device=device)
# Remove @st.cache_resource and the global _embed_model initialization.
# Replace it with a singleton getter:

# DEVICE DETECTION
# =========================
def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        # Optimization 2: Add Mac (mps) support
        log.info("✓ Loading embedding model...")

        device = _get_device()
        if device == "cuda":
            log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        dtype = torch.float16 if device != "cpu" else torch.float32
        log.info(f"Loading embedding model: {EMBED_MODEL} on {device} ({dtype}) ...")
        t0 = time.perf_counter() 
        # Optimization 3: Load in fp16 (Half Precision) to save 50% RAM and load faster
        _embed_model = SentenceTransformer(
            EMBED_MODEL, 
            device=device,
            model_kwargs={
                "torch_dtype": dtype,
                # "use_safetensors": True
            }
        )

        # ── Warm-up pass: eliminates first-call latency spike ────
        _ = _embed_model.encode(["warmup"], normalize_embeddings=True, batch_size=1)

        elapsed = time.perf_counter() - t0
        vram_mb = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else 0
        log.info(
            f"✓ Model ready in {elapsed:.1f}s | device={device}"
            + (f" | VRAM={vram_mb:.0f}MB" if vram_mb else "")
        )
        # log.info("✓ Embedding model ready")
       
    return _embed_model

# =========================
# QDRANT CLIENT
# =========================
if not QDRANT_CLOUD_URL or not QDRANT_API_KEY:
    raise EnvironmentError("QDRANT_URL and QDRANT_API_KEY must be set in .env")

qdrant_client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY, timeout=300)
log.info("✓ Connected to Qdrant Cloud")


# =========================
# EMBEDDING
# =========================
def create_embedding(text: str) -> Optional[List[float]]:
    """
    Single-text embedding — used by query_engine at query time.
    Returns a plain Python list for Qdrant compatibility.
    """
    try:
        model = get_embed_model()

        vec = model.encode(
            text,
            # show_progress_bar=True, 
            normalize_embeddings=True)
        return vec.tolist()
    except Exception as e:
        log.error(f"Embedding error: {e}")
        return None


def create_embeddings_batch(
    texts: List[str],
    batch_size: int = EMBED_BATCH,
) -> List[Optional[List[float]]]:
    """
    Batch embedding — used during indexing for maximum throughput.
    Falls back to None for individual failures without crashing the batch.

    Args:
        texts:      List of strings to embed.
        batch_size: Chunks per forward pass (tune to your hardware).

    Returns:
        List of embedding vectors (same length as `texts`).
        Individual failures are None.
    """
    if not texts:
        return []

    model = get_embed_model()
    results: List[Optional[List[float]]] = [None] * len(texts)

    try:

        t0 = time.perf_counter()

        # encode() handles internal mini-batching; normalize for cosine search
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
            # device=model.device
        )


        elapsed = time.perf_counter() - t0
        
        for i, vec in enumerate(vecs):
            results[i] = vec.tolist()

        vram_mb = (
            torch.cuda.memory_allocated() / 1024**2
            if str(model.device) == "cuda" else 0
        )
        log.info(
            f"✓ Embedded {len(texts)} chunks | {elapsed:.2f}s "
            f"({len(texts)/elapsed:.1f} chunks/s)"
            + (f" | VRAM={vram_mb:.0f}MB" if vram_mb else "")
        )

    except Exception as e:
        log.error(f"Batch embedding failed: {e}. Falling back to single embeddings.")
        for i, text in enumerate(texts):
            results[i] = create_embedding(text)
    return results


# =========================
# CHUNKING — video transcripts
# =========================
def chunk_transcript(transcript_data: Dict, video_title: str) -> List[Dict]:
    """
    Sliding-window chunker with word-level overlap for video transcripts.

    Args:
        transcript_data: JSON from transformation.py (has 'chunks' key).
        video_title:     Used as source_name in payload.

    Returns:
        List of chunk dicts ready for embedding + upsert.
    """
    raw_chunks = transcript_data.get("chunks", [])
    if not raw_chunks:
        log.warning(f"No segments found in transcript for: {video_title}")
        return []

    combined: List[Dict] = []
    current: Dict = {"text": "", "start": 0.0, "end": 0.0}

    for seg in raw_chunks:
        seg_text  = seg.get("text", "").strip()
        if not seg_text:
            continue

        seg_words = len(seg_text.split())
        cur_words = len(current["text"].split())

        if cur_words + seg_words > TARGET_WORDS and current["text"]:
            combined.append(_make_video_chunk(current, video_title))

            # overlap: carry last N words into next chunk for context continuity
            words   = current["text"].split()
            overlap = " ".join(words[-OVERLAP_WORDS:]) if len(words) > OVERLAP_WORDS else current["text"]
            current = {"text": overlap + " " + seg_text, "start": seg["start"], "end": seg["end"]}
        else:
            if not current["text"]:
                current["start"] = seg["start"]
            current["text"] += " " + seg_text
            current["end"]    = seg["end"]

    if current["text"].strip():
        combined.append(_make_video_chunk(current, video_title))

    # attach stable metadata
    for i, chunk in enumerate(combined):
        chunk["chunk_id"]    = str(uuid.uuid4())
        chunk["chunk_index"] = i
        chunk["word_count"]  = len(chunk["text"].split())

    log.info(f"✓ Chunked '{video_title}' → {len(combined)} chunks")
    return combined


def _make_video_chunk(current: Dict, video_title: str) -> Dict:
    return {
        "text":        current["text"].strip(),
        "start":       current["start"],
        "end":         current["end"],
        "source_name": video_title,
        "source_type": "video",
    }


# =========================
# CHUNKING — PDF (text-based)
# =========================
def chunk_pdf_text(pages: List[Dict], pdf_title: str) -> List[Dict]:
    """
    Chunk pre-extracted PDF pages using the same sliding-window strategy.

    Each item in `pages` is expected to have: {'page': int, 'text': str}
    """
    combined: List[Dict] = []
    current: Dict = {"text": "", "page_start": 1, "page_end": 1}

    for page in pages:
        page_text = page.get("text", "").strip()
        page_num  = page.get("page", 0)
        if not page_text:
            continue

        seg_words = len(page_text.split())
        cur_words = len(current["text"].split())

        if cur_words + seg_words > TARGET_WORDS and current["text"]:
            combined.append(_make_pdf_chunk(current, pdf_title))
            words   = current["text"].split()
            overlap = " ".join(words[-OVERLAP_WORDS:]) if len(words) > OVERLAP_WORDS else current["text"]
            current = {"text": overlap + " " + page_text, "page_start": page_num, "page_end": page_num}
        else:
            if not current["text"]:
                current["page_start"] = page_num
            current["text"]    += " " + page_text
            current["page_end"] = page_num

    if current["text"].strip():
        combined.append(_make_pdf_chunk(current, pdf_title))

    for i, chunk in enumerate(combined):
        chunk["chunk_id"]    = str(uuid.uuid4())
        chunk["chunk_index"] = i
        chunk["word_count"]  = len(chunk["text"].split())

    log.info(f"✓ Chunked PDF '{pdf_title}' → {len(combined)} chunks")
    return combined


def _make_pdf_chunk(current: Dict, pdf_title: str) -> Dict:
    return {
        "text":        current["text"].strip(),
        "page_number":  current["page_start"],
        "page_end":    current["page_end"],
        "source_name": pdf_title,
        "source_type": "pdf",
    }


# =========================
# QDRANT — COLLECTION
# =========================
def ensure_collection(collection_name: str, recreate: bool = False) -> None:
    """
    Create collection if it doesn't exist; optionally recreate.
    """
    existing = {c.name for c in qdrant_client.get_collections().collections}

    if collection_name in existing:
        if recreate:
            qdrant_client.delete_collection(collection_name)
            log.info(f"Deleted existing collection: {collection_name}")
        else:
            log.info(f"Collection already exists: {collection_name}")
            return

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    log.info(f"✓ Created collection: {collection_name}")


# =========================
# QDRANT — UPSERT
# =========================


def upsert_chunks(
    chunks: List[Dict],
    collection_name: str,
    batch_size: int = 50,
) -> Tuple[int, int]:
    """
    Embed all chunks in one batched call, then upsert to Qdrant in pages.

    Args:
        chunks:          List of chunk dicts (must have 'text' key).
        collection_name: Target Qdrant collection.
        batch_size:      Points per upsert call (network batch, not embed batch).

    Returns:
        (success_count, failure_count)
    """
    if not chunks:
        log.warning("upsert_chunks called with empty list.")
        return 0, 0

    texts = [c["text"] for c in chunks]

    # ── Batch embed ────────────────────────────────────────────
    embeddings = create_embeddings_batch(texts)

    # ── Build PointStructs (skip failed embeds) ────────────────
    points: List[PointStruct] = []

    n_failed = 0
    for chunk, vec in zip(chunks, embeddings):
        if vec is None:
            log.warning(f"Skipping chunk (embed failed): {chunk.get('chunk_id')}")
            n_failed += 1
            continue
        points.append(
            PointStruct(
                id=chunk["chunk_id"],
                vector=vec,
                payload=chunk,
            )
        )

    if not points:
        log.error("All embeddings failed — nothing to upsert.")
        return 0, n_failed
    
    # ── Upsert in network batches ──────────────────────────────
    log.info(f"Upserting {len(points)} points to '{collection_name}'...")
    
    def _upsert_batch(batch: List[PointStruct]) -> Tuple[int, int]:
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            return len(batch), 0
        except Exception as e:
            log.error(f"Batch Failed: {e}")
            return 0, len(batch)
    success=0
    # failed=0

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(_upsert_batch, points[i : i + batch_size]): i
            for i in range(0, len(points), batch_size)
        }
        for future in as_completed(futures):
            s, f = future.result()
            success  += s
            n_failed += f
            log.info(f"  ↳ Upserted {success}/{len(points)}")
 
    log.info(f"✓ Upsert complete — {success} ok, {n_failed} failed")
    return success, n_failed
        


# =========================
# HIGH-LEVEL: index transcript file
# =========================
def index_transcript_file(
    transcript_path: str,
    collection_name: str,
    recreate: bool = False,
) -> bool:
    """
    Full pipeline: load JSON → chunk → embed → upsert.
    Returns True on success.
    """
    path = Path(transcript_path)
    if not path.exists():
        log.error(f"Transcript not found: {transcript_path}")
        return False

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.error(f"Failed to load transcript: {e}")
        return False

    title  = data.get("audio_file", path.stem)
    chunks = chunk_transcript(data, video_title=title)

    if not chunks:
        log.warning("No chunks produced — aborting.")
        return False

    ensure_collection(collection_name, recreate=recreate)
    success, failed = upsert_chunks(chunks, collection_name)
    return failed == 0 or success > 0


# =========================
# MAIN — smoke test
# =========================
if __name__ == "__main__":
    log.info("\n── Smoke test ──────────────────────────────────────")

    # 1. Single embed
    vec = create_embedding("What is gradient descent?")
    assert vec and len(vec) == VECTOR_DIM, "Single embed failed"
    log.info(f"✓ Single embed dim={len(vec)}")

    # 2. Batch embed
    test_texts = [f"This is test sentence number {i}." for i in range(10)]
    vecs = create_embeddings_batch(test_texts, batch_size=4)
    assert len(vecs) == 10 and all(v is not None for v in vecs), "Batch embed failed"
    log.info(f"✓ Batch embed — {len(vecs)} vectors, dim={len(vecs[0])}")

    # 3. Transcript file (optional)
    sample = Path(TRANSCRIPTS_FOLDER) / "sample.json"
    if sample.exists():
        ok = index_transcript_file(str(sample), collection_name="smoke_test", recreate=True)
        log.info(f"✓ Index transcript: {'OK' if ok else 'FAILED'}")
    else:
        log.info("(No sample.json — skipping index test)")

    log.info("── All checks passed ───────────────────────────────")