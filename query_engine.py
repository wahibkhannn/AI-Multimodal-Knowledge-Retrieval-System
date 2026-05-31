"""
Enhanced Query Engine - v2.0
- Dual LLM support (Ollama + Gemini)
- Better error handling
- Context expansion
- Relevance scoring
- Caching support
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv
import google.generativeai as genai

# from read_chunks import create_embedding, qdrant_client

load_dotenv()

import logging
log = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # or mistral, qwen, etc.
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() == "true"  # Toggle in .env

# =========================
# GEMINI INIT — google.genai (new SDK)
# =========================
gemini_client = None
 
if USE_GEMINI:
    try:
        from google import genai
        from google.genai import types as genai_types
 
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        GEMINI_MODEL  = "gemini-2.5-flash"
        log.info("✓ Query Engine ready [Gemini]")
    except Exception as e:
        log.warning(f"Gemini init failed ({e}) — falling back to Ollama")
        USE_GEMINI = False
 
if not USE_GEMINI:
    log.info("✓ Query Engine ready [Ollama]")
 


def _get_search_deps():
    from read_chunks import create_embedding, qdrant_client
    return create_embedding, qdrant_client
# =========================
# SEARCH WITH FILTERING
# =========================

def search_videos(
    query: str,
    collection_name: str,
    top_k: int = 5,
    min_score: float = 0.35,  # Filter low-relevance results
    video_filter: str = None  # Search specific video
) -> List[Dict]:
    """
    Search for relevant chunks in Qdrant
    
    Args:
        query: User's question
        collection_name: Optional custom collection
        top_k: Number of results
        min_score: Minimum relevance score (0-1)
        video_filter: Optional video title to search within
    
    Returns:
        List of relevant chunks with metadata
    """

    # collection = collection_name
    
    if not collection_name:
        raise ValueError("Collection name missing. Session not initialized.")
    # Get query embedding
    create_embedding, qdrant_client = _get_search_deps()

    query_embedding = create_embedding(query)
    if query_embedding is None:
        log.warning("Failed to embed query")
        return []
    
    try:
        # Build filter if specific video requested
        search_filter = None
        if video_filter:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            

            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="source_name",
                        match=MatchValue(value=video_filter)
                    )
                ]
            )
        
        # Query Qdrant (v1.16+ API)
        result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
            score_threshold=min_score  # Only return results above threshold
        )
        
        # Extract payloads
        chunks = [point.payload for point in result.points]
        log.info(f"✓ Retrieved {len(chunks)} chunks (min_score={min_score})")
        # print(f"✓ Found {len(chunks)} relevant chunks (min score: {min_score})")
        return chunks
    
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


# =========================
# CONTEXT EXPANSION
# =========================
def expand_context(chunks: List[Dict],collection_name, window: int = 1) -> List[Dict]:
    """
    Expand search results with neighboring chunks for better context
    
    Args:
        chunks: Initial search results
        window: Number of chunks before/after to include
    
    Returns:
        Expanded list of chunks
    """
    if not chunks:
        return []
    
    # expanded_ids = set()
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    _, qdrant_client = _get_search_deps()

    video_chunks: Dict[str,set] = {}
    
    # Group by video and collect indices
    for chunk in chunks:


        src = chunk.get('source_name') or chunk.get('video_title', 'unknown')
        idx = chunk.get('chunk_index', 0)
        video_chunks.setdefault(src, set())

      
        # Add this chunk + neighbors
        for offset in range(-window, window + 1):
            video_chunks[src].add(idx + offset)
    
    # Fetch expanded chunks from Qdrant
    expanded = []
    for src, indices in video_chunks.items():
        for idx in sorted(indices):
            if idx < 0:
                continue
            
            try:                
                result = qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="source_name", match=MatchValue(value=src)),
                            FieldCondition(key="chunk_index", match=MatchValue(value=idx))
                        ]
                    ),
                    limit=1,
                    with_payload=True
                )
                
                if result[0]:
                    expanded.append(result[0][0].payload)
            
            except Exception as e:
                print(f"⚠️  Failed to fetch chunk {src}:{idx} - {e}")
                continue
    
    # Sort by video and chunk order
    # expanded.sort(key=lambda x: (x.get('video_title', ''), x.get('chunk_index', 0)))
    expanded.sort(
        key=lambda x: (
            (x.get('source_name') or x.get('video_title', '')),
            x.get('chunk_index', 0)
        )
    )

    
    log.info(f"✓ Context expanded → {len(expanded)} chunks (window={window})")
    return expanded


# =========================
# LLM GENERATION (DUAL)
# =========================
def generate_with_ollama(prompt: str) -> Optional[str]:
    """Generate answer using local Ollama"""
    import json
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0.35,
                    "num_predict": 512
                }
            },
            stream=True,
            timeout=120
        )
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                token= chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break

    except requests.exceptions.ConnectionError:
        yield "❌ Ollama not running. Start with: ollama serve"
        
    except Exception as e:
        yield f"\n❌ Error: {e}"
    

def generate_with_gemini_stream(prompt: str) -> Optional[str]:
    """Generate answer using Gemini API"""
    if not gemini_client:
        yield "❌ Gemini Client not initialized!"
        return
    try:
       for chunk in gemini_client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.33,
                max_output_tokens=750,
            ),
        ):
            if chunk.text:
                for ch in chunk.text:
                    yield ch
        # return response.text.strip()
    
    except Exception as e:
        log.error(f"Gemini streaming error: {e}")
        yield f"❌ Gemini error"

def generate_answer_stream(query: str, chunks: List[Dict]):
    if not chunks:
        yield "❌ No relevant information found in the video transcripts."
        yield {"sources": [], "confidence": 0.0}
        return
    prompt = build_prompt(query, chunks)  # reuse your prompt logic

    if USE_GEMINI:
        for token in generate_with_gemini_stream(prompt):
            yield token

        # yield generate_with_gemini(prompt)
    else:
        for token in generate_with_ollama(prompt):
            yield token
    yield {
        "sources": chunks[:4],
        "confidence": 0.85,
        "llm_used": "Gemini" if USE_GEMINI else "Ollama"
    }

def build_prompt(query: str, chunks: List[Dict]) -> str:
    """Build the RAG prompt from query and retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "").strip()
        title = chunk.get("source_name") or chunk.get("video_title", "Source")

        if chunk.get("source_type") == "pdf":
            page = chunk.get("page_number", chunk.get("chunk_index", 0))
            ref = f"[Page {page}]"
        else:
            start = chunk.get("start", 0)
            mins = int(start // 60)
            secs = int(start % 60)
            ref = f"[{mins}:{secs:02d}]"

        if text:
            context_parts.append(f"[Source {i} - {title} {ref}]:\n{text}")

    context = "\n\n".join(context_parts)

    return f"""You are an intelligent educational video assistant.

Your task:
- Answer the user's question using ONLY the video transcript context below
- Use proper headings (## Section Title, ###)
- Use bullet points where needed
- Never put text on the same line as the heading
- Synthesize information from multiple sources if strictly needed
- Be clear, concise, and educational
- Highlight important terms in **bold** and italic as needed.
- Combine information without repeating the same section
- Keep timestamps like [mins:secs] for reference citations like [1:32]
- For pdf input use page references like [Page N] for citing pages.
- If the transcript doesn't contain enough info, say so honestly
- Do not give output beyond 750 tokens.

Video Transcript Context:
{context}

Student's Question:
{query}

Provide a well-structured answer with timestamp citations:"""
# =========================
# GENERATE ANSWER (RAG)
# =========================
def generate_answer(query: str, chunks: List[Dict]) -> Dict:
    """
    Generate answer using retrieved chunks
    
    Args:
        query: User's question
        chunks: Retrieved context chunks
    
    Returns:
        {answer, sources, confidence}
    """
    if not chunks:
        return {
            "answer": "❌ No relevant information found in the video transcripts.",
            "sources": [],
            "confidence": 0.0
        }
    
    # Build context with timestamps
#     context_parts = []
#     for i, chunk in enumerate(chunks, 1):
#         start = chunk.get("start", 0)
#         mins = int(start // 60)
#         secs = int(start % 60)
#         text = chunk.get("text", "").strip()
#         # title = chunk.get("video_title", "Video")
#         title = chunk.get("source_name") or chunk.get("video_title", "Source")

#         if chunk.get("source_type") == "pdf":
#             # page = chunk.get("chunk_index", 0)
#             page = chunk.get("page_number", chunk.get("chunk_index", 0))
#             ref = f"[Page {page}]"
#         else:
#             start = chunk.get("start", 0)
#             mins = int(start // 60)
#             secs = int(start % 60)
#             ref = f"[{mins}:{secs:02d}]"

#         if text:
#             # context_parts.append(
#             #     f"[Source {i} - {title} at {mins}:{secs:02d}]:\n{text}"
#             # )
#             context_parts.append(
#                 f"[Source {i} - {title} {ref}]:\n{text}"
#             )
    
#     context = "\n\n".join(context_parts)
    
#     # Enhanced prompt
    # prompt = f"""You are an intelligent educational video assistant.

# Your task:
# - Answer the user's question using ONLY the video transcript context below
# - Use proper headings (## Section Title, ###)
# - Use bullet points where needed
# - Never put text on the same line as the heading
# - Synthesize information from multiple sources if strictly needed 
# - Be clear, concise, and educational
# - Highlight important terms in **bold** and italic as needed.
# - Combine information without repeating the same section
# - Keep timestamps like [mins:secs] for reference citations like [1:32]"
# - For pdf input use reference {ref} for referencing page number of the pdf.
# - If the transcript doesn't contain enough info, say so honestly
# - Do not give output beyond 45 tokens.
# Video Transcript Context:
# {context}

# Student's Question:
# {query}

# Provide a well-structured answer with timestamp citations:"""
    prompt = build_prompt(query, chunks)
    # Generate with preferred LLM
    if USE_GEMINI:
        answer_text = generate_with_gemini_stream(prompt)
        if answer_text is None:
            log.warning("⚠️ Gemini failed — trying Ollama")
            answer_text = generate_with_ollama(prompt)
    else:
        answer_text = generate_with_ollama(prompt)
    
    # Fallback if both fail
    if answer_text is None:
        answer_text = "❌ Error: Could not generate answer. Please check LLM service."
    
    # Calculate confidence based on chunk relevance
    # (If chunks have score, use that; otherwise estimate)
    confidence = 0.8 if chunks else 0.0
    
    return {
        "answer": answer_text.strip(),
        "sources": chunks[:4],  # Top 4 sources
        "confidence": confidence,
        "llm_used": "Gemini" if USE_GEMINI else "Ollama"
    }


# =========================
# FULL PIPELINE
# =========================
def ask_question(
    query: str,
    collection_name: str = None,
    expand: bool = True,
    video_filter: str = None
) -> Dict:
    """
    Complete RAG pipeline: Search → Expand → Generate
    
    Args:
        query: User's question
        collection_name: Optional custom collection
        expand: Whether to expand context with neighbors
        video_filter: Search only specific video
    
    Returns:
        {query, answer, sources, confidence, llm_used}
    """
    print(f"\n{'='*60}")
    print(f"💬 Question: {query}")
    print(f"{'='*60}\n")
    
    # Step 1: Search
    chunks = search_videos(
        query,
        collection_name,
        top_k=5,
        video_filter=video_filter
    )
    
    # Step 2: Optionally expand context
    if expand and chunks:
        chunks = expand_context(chunks, collection_name, window=1)
    
    # Step 3: Generate answer
    result = generate_answer(query, chunks)
    result["query"] = query
    
    return result


# =========================
# PRETTY PRINT
# =========================
def print_answer(result: Dict):
    """Pretty print the answer with sources"""
    print(f"\n{'─'*60}")
    print(f"📝 ANSWER ({result.get('llm_used', 'Unknown')} - Confidence: {result.get('confidence', 0):.0%})")
    print(f"{'─'*60}\n")
    
    print(result.get("answer", "No answer generated"))
    
    if result.get("sources"):
        print(f"\n{'─'*60}")
        print(f"📚 SOURCES")
        print(f"{'─'*60}\n")
        
        for i, source in enumerate(result["sources"], 1):
            start = source.get("start", 0)
            mins = int(start // 60)
            secs = int(start % 60)
            title = source.get("video_title", "Unknown")
            text = source.get("text", "")
            
            print(f"{i}. {title} [{mins}:{secs:02d}]")
            print(f"   {text[:150]}...")
            print()


# =========================
# CLI TESTING
# =========================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\n📖 Usage:")
        print("  python query_engine.py <question>")
        print("\n📝 Example:")
        print('  python query_engine.py "How to take input in Java?"')
        print()
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    result = ask_question(question, expand=True)
    print_answer(result)
    
    # Save to JSON for debugging
    import json
    with open("last_query.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full result saved to: last_query.json")