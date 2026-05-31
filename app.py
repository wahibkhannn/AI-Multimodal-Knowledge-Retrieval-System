"""
Streamlit UI for Video & PDF Q&A System
Supports: Video upload (MP4/MKV/MOV) + PDF upload
"""
import bootstrap  # 🔥 THIS LINE FIRST
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import warnings
warnings.filterwarnings("ignore", message="Accessing `__path__`.*")
warnings.filterwarnings("ignore", category=FutureWarning)


import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

import streamlit as st
import atexit
import threading

from dotenv import load_dotenv
import time
start = time.time()





load_dotenv()

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# ── Lazy imports (after env vars set) ────────────────────────
from query_engine import ask_question
from run import VideoSession
from process_pdf import PDFSession



# =========================
# MODEL PRE-WARM
# Runs once in a background thread so the first upload doesn't
# block the UI for 15-20 seconds.
# =========================
@st.cache_resource(show_spinner=False)
def _prewarm_model():
    """Load + warm-up the embedding model exactly once per process."""
    def _load():
        try:
            from read_chunks import get_embed_model
            model = get_embed_model()          # loads weights + runs warmup pass
            model.encode(["hello world"], show_progress_bar=False, batch_size=1)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Model pre-warm failed: {e}")
 
    t = threading.Thread(target=_load, daemon=True, name="model-prewarm")
    t.start()
    return t   # caller can .join() if needed
 
_prewarm_model()

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"], div[data-testid="stMarkdownContainer"] {
        color: #111 !important;
    }
    @media (prefers-color-scheme: dark) {
        html, body, [class*="css"], div[data-testid="stMarkdownContainer"] {
            color: #f1f1f1 !important;
        }
        .answer-box { background-color: #1e1e1e !important; border-left-color: #8a7dff !important; }
    }
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle { text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem; }
    .stTextInput input { font-size: 1.1rem; }
    .upload-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🤖 AI Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Ask questions about your videos or PDFs and get instant answers</p>',
    unsafe_allow_html=True
)

# ── Upload Section ────────────────────────────────────────────
st.markdown("## 📁 Upload Your Content")
tab_video, tab_pdf = st.tabs(["🎞️ Video", "📄 PDF"])

# ── VIDEO TAB ─────────────────────────────────────────────────
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a lecture or educational video (MP4/MKV/MOV):",
        type=["mp4", "mkv", "mov"],
        # accept_multiple_files=False,
        key="video_uploader"
    )

    if uploaded_video is not None:
        temp_dir = "uploaded_videos"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.success(f"✅ Uploaded: {uploaded_video.name}")

        if st.button("▶️ Process Video", key="process_video_btn"):
            with st.spinner("⏳ Processing video… this may take a minute."):
                try:
                    # Clean up previous session if any
                    if "active_session" in st.session_state:
                        try:
                            st.session_state["active_session"].cleanup()
                        except Exception:
                            pass

                    session = VideoSession(video_path)
                    session.process_video()
                    st.session_state["active_session"] = session
                    st.session_state["source_type"] = "video"
                    st.session_state["source_name"] = uploaded_video.name
                    st.success("🎉 Video processed! Ask questions below.")
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")

# ── PDF TAB ───────────────────────────────────────────────────
with tab_pdf:
    uploaded_pdf = st.file_uploader(
        "Upload a PDF document:",
        type=["pdf"],
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    if uploaded_pdf is not None:
        temp_dir = "uploaded_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.success(f"✅ Uploaded: {uploaded_pdf.name}")

        if st.button("📖 Process PDF", key="process_pdf_btn"):
            with st.spinner("⏳ Processing PDF…"):
                try:
                    # Clean up previous session if any
                    if "active_session" in st.session_state:
                        try:
                            st.session_state["active_session"].cleanup()
                        except Exception:
                            pass

                    session = PDFSession(pdf_path)
                    session.process_pdf()
                    st.session_state["active_session"] = session
                    st.session_state["source_type"] = "pdf"
                    st.session_state["source_name"] = uploaded_pdf.name
                    st.success("🎉 PDF processed! Ask questions below.")
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")

# ── Active Source Indicator ───────────────────────────────────
if "active_session" in st.session_state:
    src_type = st.session_state.get("source_type", "file")
    src_name = st.session_state.get("source_name", "Unknown")
    icon = "🎞️" if src_type == "video" else "📄"
    # 🎬 Video Player with Jump Support (GLOBAL)
    if st.session_state.get("source_type") == "video":
        video_path = st.session_state["active_session"].video_path

        jump_time = st.session_state.pop("jump_to", None)
        if jump_time is not None:
            st.video(video_path, start_time=int(jump_time))
        else:
            st.video(video_path)

    st.info(f"{icon} **Active source:** {src_name}")

st.markdown("---")

# ── Chat Section ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([5, 1])

with col1:
    user_query = st.text_input(
        "Your question:",
        placeholder="e.g., What is the main topic discussed?",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("🔍 Ask", use_container_width=True, type="primary")

# ── Process Query ─────────────────────────────────────────────
# if ask_button and user_query.strip():
#     with st.spinner("🤔 Thinking..."):
#         if "active_session" in st.session_state:
#             session = st.session_state["active_session"]
#             result = ask_question(user_query, collection_name=session.collection_name)
#         else:
#             st.warning("⚠️ Please upload and process a video or PDF first.")
#             result = {"answer": "No content loaded. Please upload a video or PDF above."}
        
#         placeholder_idx = len(st.session_state.chat_history)
#         st.session_state.chat_history.append({
#             "question": user_query,
#             "answer": result.get("answer", "No answer available."),
#             "sources": result.get("sources", []),
#             "confidence": result.get("confidence", 0.0)
#         })
if ask_button and user_query.strip():
    if "active_session" not in st.session_state:
        st.warning("⚠️ Please upload and process a video or PDF first.")
    else:
        session = st.session_state["active_session"]

        from query_engine import search_videos, expand_context, generate_answer_stream

        # 🔍 Step 1: Search (fast)
        chunks = search_videos(user_query, session.collection_name, top_k=5)

        # 🔍 Step 2: Expand (optional)
        if chunks:
            chunks = expand_context(chunks, session.collection_name, window=1)

        # 🧠 Step 3: Reserve slot in chat history
        placeholder_idx = len(st.session_state.chat_history)

        st.session_state.chat_history.append({
            "question": user_query,
            "answer": "",
            "sources": [],
            "confidence": 0.0
        })

        # 🧑‍💻 Show question immediately
        st.markdown(f"""
        <div style="
            background-color:#e8f0fe;
            color:#111;
            padding:1rem;
            border-radius:10px;
            margin-bottom:0.5rem;
            max-width:80%;
        ">
            <strong>🧑‍💻 You:</strong><br>{user_query}
        </div>
        """, unsafe_allow_html=True)

        collected_tokens = []
        final_meta = {"sources": [], "confidence": 0.0, "llm_used": ""}

        # ⚡ STREAM WRAPPER
        def token_stream():
            for item in generate_answer_stream(user_query, chunks):
                if isinstance(item, dict):
                    final_meta.update(item)
                else:
                    collected_tokens.append(item)
                    yield item

        # 🚀 STREAM HERE
        full_answer = st.write_stream(token_stream())

        # 💾 Save final result
        st.session_state.chat_history[placeholder_idx].update({
            "answer": full_answer,
            "sources": final_meta["sources"],
            "confidence": final_meta["confidence"]
        })

        st.progress(final_meta["confidence"])
        st.caption(f"Confidence: {final_meta['confidence']*100:.1f}%")

        st.rerun()

# ── Render Answer Block ───────────────────────────────────────
def render_answer_block(answer_text: str):
    if not answer_text:
        st.markdown('<div style="color:#aaa">No answer available.</div>', unsafe_allow_html=True)
        return

    parts = answer_text.split("```")

    for i, part in enumerate(parts):
        if i % 2 == 0:
            plain = part.strip()
            if plain:
                # Styled container START
                st.markdown("""
                <div style="
                    background: linear-gradient(145deg, #1e1e2f, #25253a);
                    padding: 1.2rem;
                    border-radius: 14px;
                    border: 1px solid rgba(255,255,255,0.08);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                    max-width: 700px;
                    margin-bottom: 10px;
                ">
                """, unsafe_allow_html=True)

                # ✅ Proper Markdown rendering
                st.markdown(plain)

                # Styled container END
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            code = part.strip()
            first_line = code.splitlines()[0] if code.splitlines() else ""
            lang = None

            if first_line in ("python", "java", "js", "javascript", "c", "cpp", "bash", "html", "css"):
                code_body = "\n".join(code.splitlines()[1:]).strip()
                lang = first_line
            else:
                code_body = code

            st.markdown(
                '<div style="margin-top:6px; margin-bottom:6px; font-weight:600; color:#aaa;">Code:</div>',
                unsafe_allow_html=True
            )
            st.code(code_body, language=lang or "text")


# ── Chat History ──────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("---")

    for chat_idx, chat in enumerate(st.session_state.chat_history):
        # Question bubble
        st.markdown(f"""
        <div style="
            background-color:#e8f0fe;
            color:#111;
            padding:1rem;
            border-radius:10px;
            margin-bottom:0.5rem;
            max-width:80%;
            white-space:pre-wrap;
            word-wrap:break-word;
        ">
            <strong>🧑‍💻 You:</strong><br>{chat['question']}
        </div>
        """, unsafe_allow_html=True)

        render_answer_block(chat["answer"])
        confidence = chat.get("confidence", 0.0)
        st.progress(confidence)
        st.caption(f"Confidence: {confidence*100:.1f}%")

        sources = chat.get("sources", [])
        if sources:
            st.markdown(
                '<div style="margin-top: 0.5rem; color:#667eea; font-weight:600;">🔗 References</div>',
                unsafe_allow_html=True
            )
            for src in sources:
                title = src.get("source_name") or src.get("video_title") or "Untitled"
                if src.get("source_type")== "video":
                    ts = src.get("start", 0)

                    if ts:
                        mins = int(ts // 60)
                        secs = int(ts % 60)
                        ts_label = f"{mins}:{secs:02d}"
                    else:
                        ts_label = "—"
                else:
                    ts= None
                    page = src.get("page_number", None)
                    # page = src.get("chunk_index", None)
                    ts_label = f"Page {page}" if page is not None else "📄"

                snippet = src.get("text", "")[:120]

                # Layout: title + button side by side
                col1, col2 = st.columns([6, 1])

                with col1:
                    chunk_id = src.get("chunk_id")
                    st.markdown(
                        # f'<div id="{title}_{ts}"></div>',
                        f'<div id="ref_{chat_idx}_{chunk_id}"></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(145deg, #1e1e2f, #25253a);
                        padding:0.9rem;
                        border-radius:10px;
                        border: 1px solid rgba(255,255,255,0.08);
                        margin:0.4rem 0;
                        color:#111;
                    ">
                       <div style="font-weight:600; color:#eaeaff;">
                        {title}
                    </div>
                    <div style="font-size:0.85rem; color:#aaa;">
                        {snippet}…
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with col2:  
                    chunk_id = src.get("chunk_id")
                    key = f"btn_{chat_idx}_{chunk_id}"
                    if st.button(f"▶ {ts_label}", key=key):
                        st.session_state["jump_to"] = ts
                        st.markdown(
                            f"""
                            <script>
                            window.location.hash = "ref_{chat_idx}_{chunk_id}";
                            </script>
                            """,
                            unsafe_allow_html=True
                        )

                        st.rerun()

    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("🧹 End Session"):
        if "active_session" in st.session_state:
            st.session_state["active_session"].cleanup()
            del st.session_state["active_session"]
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #888;">
        <h3>👋 Welcome!</h3>
        <p>Upload a <strong>video</strong> or <strong>PDF</strong> above, then ask any question about its content.</p>
    </div>
    """, unsafe_allow_html=True)

def cleanup(self):
    try:
        from read_chunks import qdrant_client
        
        qdrant_client.delete_collection(self.collection_name)
        print(f"🧹 Deleted collection: {self.collection_name}")
        
    except Exception as e:
        print(f"⚠️ Cleanup error: {e}")
# ── Cleanup on Exit ───────────────────────────────────────────
def cleanup_on_exit():
    if "active_session" in st.session_state:
        try:
            st.session_state["active_session"].cleanup()
            print("🧹 Session cleaned up on exit")
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")


atexit.register(cleanup_on_exit)
print(f"Startup time: {time.time() - start:.2f}s")