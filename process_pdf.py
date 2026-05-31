"""
PDF Session Manager
Mirrors VideoSession from run.py — uploads PDF, chunks text, indexes into Qdrant, supports Q&A, cleans up.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import uuid
import shutil
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader
from qdrant_client.models import Distance, VectorParams, PointStruct

from read_chunks import create_embedding, qdrant_client
from query_engine import ask_question

from read_chunks import ensure_collection, upsert_chunks

# =========================
# TEXT CHUNKING
# =========================
def split_text(text: str, chunk_size: int = 250, overlap: int = 50) -> List[Dict]:
    """Split plain text into overlapping word chunks."""
    words = text.split()
    chunks = []

    i = 0
    chunk_index = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": chunk_index,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start": 0,   # PDFs have no timestamps — kept for schema compat
            "end": 0,
        })

        i += chunk_size - overlap   # slide window with overlap
        chunk_index += 1

    return chunks


# =========================
# PDF SESSION
# =========================
class PDFSession:
    """
    Manages a single PDF Q&A session:
    - Extract & chunk PDF text
    - Create a temporary Qdrant collection
    - Embed and index chunks
    - Answer questions via query_engine
    - Cleanup on exit
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_name = Path(pdf_path).stem
        self.session_id = str(uuid.uuid4())[:8]
        self.collection_name = f"pdf_session_{self.session_id}"

        # Working dir for any temp files
        self.work_dir = Path("temp_session") / f"pdf_{self.session_id}"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        print(f"📄 PDF Session started: {self.session_id}")
        print(f"   File: {self.pdf_name}\n")

    # --------------------------------------------------
    def process_pdf(self):
        """Full pipeline: extract → chunk → embed → index."""

        # ── Step 1: Extract text ──────────────────────
        print("Step 1/3: Extracting PDF text...")
        reader = PdfReader(self.pdf_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        if not full_text.strip():
            raise ValueError("Could not extract any text from the PDF (scanned image PDF?)")

        print(f"   Extracted ~{len(full_text.split())} words from {len(reader.pages)} pages")

        # ── Step 2: Chunk ─────────────────────────────
        print("Step 2/3: Chunking text...")
        chunks = split_text(full_text)
        print(f"   Created {len(chunks)} chunks")

        # ── Step 3: Embed & index ─────────────────────
        print("Step 3/3: Embedding & indexing...")

        # Create temporary Qdrant collection
        try:
            qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )
            qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source_name",
            field_schema="keyword"
            )

            qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_index",
                field_schema="integer"
            )
        except Exception:
            pass  # already exists

        points = []
        # for i, chunk in enumerate(chunks):
        #     embedding = create_embedding(chunk["text"])
        #     if not embedding:
        #         continue

        #     points.append(PointStruct(
        #         id=chunk["chunk_id"],
        #         vector=embedding,
        #         payload={
        #             "chunk_id": chunk["chunk_id"],
        #             "source_name": self.pdf_name ,
        #             "source_type": "pdf",  
        #             "text": chunk["text"],
        #             "start": chunk["start"],
        #             "end": chunk["end"],
        #             "chunk_index": chunk["chunk_index"],
        #             "word_count": chunk["word_count"],
        #             "page_number": (chunk["chunk_index"]//2)+1
        #         }
        #     ))

        #     if (i + 1) % 5 == 0 or i == len(chunks) - 1:
        #         print(f"   Progress: {i + 1}/{len(chunks)}", end="\r")

        # if points:
        #     qdrant_client.upsert(
        #         collection_name=self.collection_name,
        #         points=points
        #     )

        ensure_collection(self.collection_name)

        qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="source_name",
            field_schema="keyword"
        )

        qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_index",
            field_schema="integer"
        )

        
        upsert_chunks(chunks, self.collection_name)
        success, failed = upsert_chunks(
            chunks,
             self.collection_name
        )

        print(f"\n   Indexed {success} chunks")
   
        # print(f"\n   Indexed {len(points)} chunks")
        print("\n--------------------------")
        print("✅ READY TO ANSWER PDF QUESTIONS!")
        print("--------------------------\n")

    # --------------------------------------------------
    def ask(self, question: str) -> Dict:
        return ask_question(question, collection_name=self.collection_name)

    # --------------------------------------------------
    def cleanup(self):
        """Delete temporary Qdrant collection and local files."""
        print(f"\n🧹 Cleaning up PDF session {self.session_id}...")

        try:
            qdrant_client.delete_collection(self.collection_name)
            print("   ✓ Deleted Qdrant collection")
        except Exception as e:
            print(f"   Collection cleanup: {e}")

        try:
            shutil.rmtree(self.work_dir)
            print("   ✓ Deleted temp files")
        except Exception as e:
            print(f"   File cleanup: {e}")

        print("   Done!\n")