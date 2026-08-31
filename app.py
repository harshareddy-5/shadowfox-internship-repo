"""
DocuMind - AI Document Question Answering Assistant (Streamlit Web Interface).

Provides a full RAG user interface for uploading documents, adjusting hyperparameters,
triggering FAISS vector index creation, asking questions, and inspecting grounded answers
alongside source metadata and vector distances.
"""

import streamlit as st

# Set page config as the very first Streamlit call
st.set_page_config(
    page_title="DocuMind – AI Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished modern UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 4px solid #2563EB;
        padding: 10px 14px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Import internal modules
from src.document_loader import load_documents_from_uploaded_files
from src.chunker import split_documents, get_chunk_statistics
from src.embeddings import load_embedding_model
from src.vector_store import build_vector_store
from src.rag_pipeline import run_rag_pipeline
from src.llm import get_groq_api_key

# Initialize Session State Variables
if "processed" not in st.session_state:
    st.session_state.processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "avg_chunk_size" not in st.session_state:
    st.session_state.avg_chunk_size = 0
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ DocuMind Controls")

    st.subheader("1. Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Select one or more PDF or TXT documents to analyze."
    )

    st.subheader("2. Hyperparameters")
    chunk_size = st.slider(
        "Chunk Size (chars)",
        min_value=200,
        max_value=2000,
        value=800,
        step=50,
        help="Target size for text chunks. Recommended: ~800"
    )

    chunk_overlap = st.slider(
        "Chunk Overlap (chars)",
        min_value=0,
        max_value=500,
        value=150,
        step=25,
        help="Overlap between adjacent chunks to maintain context boundaries."
    )

    top_k = st.slider(
        "Retrieved Chunks (Top K)",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Number of vector similarity search matches to retrieve."
    )

    st.subheader("3. API Configuration")
    env_api_key = get_groq_api_key()
    user_api_key = st.text_input(
        "Groq API Key",
        value=env_api_key if env_api_key else "",
        type="password",
        help="Required for grounded LLM generation. Get a free key at console.groq.com"
    )

    model_name = st.selectbox(
        "Groq Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0,
        help="Select the Groq chat model for answer generation."
    )

    st.markdown("---")

    # Process Documents Button
    process_button = st.button("🚀 Process Documents", use_container_width=True, type="primary")

    # Clear Session State Button
    if st.button("🗑️ Clear Session State", use_container_width=True):
        st.session_state.processed = False
        st.session_state.vector_store = None
        st.session_state.doc_count = 0
        st.session_state.total_pages = 0
        st.session_state.chunk_count = 0
        st.session_state.avg_chunk_size = 0
        st.session_state.processed_files = []
        st.session_state.qa_history = []
        st.rerun()

# Document Processing Trigger
if process_button:
    if not uploaded_files:
        st.sidebar.error("⚠️ Please upload at least one PDF or TXT document.")
    else:
        with st.status("Processing Documents through RAG Pipeline...", expanded=True) as status:
            try:
                st.write("📄 Loading documents & extracting text...")
                raw_docs, load_errors = load_documents_from_uploaded_files(uploaded_files)

                if load_errors:
                    for err in load_errors:
                        st.sidebar.error(f"Error loading file: {err}")

                if not raw_docs:
                    st.error("No valid text could be extracted from uploaded files.")
                    status.update(label="Processing Failed", state="error")
                else:
                    st.write(f"✓ Documents loaded: {len(uploaded_files)} files")

                    # Compute page count
                    pages_count = sum(1 for d in raw_docs if d.metadata.get("page") is not None)
                    page_str_display = pages_count if pages_count > 0 else 'TXT'
                    st.write(f"✓ Text extracted across {page_str_display} pages/sections")

                    # Chunking
                    st.write(f"✂️ Splitting text (Size: {chunk_size}, Overlap: {chunk_overlap})...")
                    chunks = split_documents(
                        raw_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    stats = get_chunk_statistics(chunks)
                    st.write(
                        f"✓ Chunks created: {stats['total_chunks']} chunks "
                        f"(Avg {stats['avg_chunk_size']} chars)"
                    )

                    # Embedding generation
                    st.write("🧠 Loading embedding model (all-MiniLM-L6-v2)...")
                    embedding_model = load_embedding_model()
                    st.write("✓ Embeddings generated for all chunks")

                    # FAISS Index Creation
                    st.write("⚡ Building local FAISS vector index...")
                    vector_store = build_vector_store(chunks, embedding_model)
                    st.write("✓ FAISS index successfully created")

                    # Update Session State
                    st.session_state.vector_store = vector_store
                    st.session_state.processed = True
                    st.session_state.doc_count = len(uploaded_files)
                    st.session_state.total_pages = pages_count
                    st.session_state.chunk_count = stats['total_chunks']
                    st.session_state.avg_chunk_size = stats['avg_chunk_size']
                    st.session_state.processed_files = [f.name for f in uploaded_files]

                    status.update(
                        label="✅ RAG Index Ready!",
                        state="complete",
                        expanded=False
                    )
                    st.toast("Documents processed and indexed successfully!", icon="✅")
            except Exception as e:  # pylint: disable=broad-exception-caught
                status.update(label="❌ Processing Error", state="error")
                st.error(f"Error processing documents: {str(e)}")

# --- MAIN AREA ---
st.markdown('<div class="main-title">📚 DocuMind – AI Document Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Ask questions about your documents using Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True
)

# Document Status Dashboard
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Documents", st.session_state.doc_count)
with col2:
    page_display = st.session_state.total_pages if st.session_state.total_pages > 0 else "N/A"
    st.metric("Pages Extracted", page_display)
with col3:
    st.metric("Chunks Created", st.session_state.chunk_count)
with col4:
    st.metric("Embedding Model", "all-MiniLM-L6-v2")
with col5:
    status_str = "🟢 Ready" if st.session_state.processed else "🔴 Not Indexed"
    st.metric("Vector Store", status_str)

st.markdown("---")

# API Key Validation Notice
active_api_key = user_api_key or env_api_key
if not active_api_key:
    st.warning(
        "⚠️ **Groq API Key Required**: Please enter your Groq API key in the sidebar "
        "or configure `GROQ_API_KEY` in your `.env` file to generate answers."
    )

# Question Area
st.subheader("🔍 Ask a Question")
question_input = st.text_input(
    "Enter your question about the uploaded documents:",
    placeholder="e.g., What are the main key findings described in section 3?",
    key="question_box"
)

col_ask, col_space = st.columns([1, 4])
with col_ask:
    ask_button = st.button("💬 Ask Question", type="primary", use_container_width=True)

if ask_button:
    if not question_input or not question_input.strip():
        st.error("⚠️ Please enter a question.")
    elif not st.session_state.processed or st.session_state.vector_store is None:
        st.error(
            "⚠️ Question asked before documents are processed. "
            "Please upload documents and click 'Process Documents' in the sidebar."
        )
    elif not active_api_key:
        st.error("⚠️ Groq API Key is missing. Please provide a valid Groq API Key in the sidebar.")
    else:
        with st.spinner("Retrieving document chunks & generating grounded answer..."):
            result = run_rag_pipeline(
                vector_store=st.session_state.vector_store,
                question=question_input,
                top_k=top_k,
                api_key=active_api_key,
                model_name=model_name
            )

            if not result["success"]:
                st.error(f"❌ Error: {result['error']}")
            else:
                # Save to history
                st.session_state.qa_history.insert(0, {
                    "question": question_input,
                    "answer": result["answer"],
                    "sources": result["sources"]
                })

# Display Recent Answer & Context
if st.session_state.qa_history:
    latest = st.session_state.qa_history[0]

    st.markdown("### 💡 Answer")
    st.info(latest["answer"])

    st.markdown("### 📚 Retrieved Sources & Context")
    if not latest["sources"]:
        st.warning("No relevant document chunks matched the search query threshold.")
    else:
        for i, src in enumerate(latest["sources"], 1):
            page_str = f"Page {src['page']}" if src["page"] is not None else "Page N/A"
            expander_title = (
                f"Source {i} – {src['source']} – {page_str} (Distance: {src['distance']})"
            )

            with st.expander(expander_title, expanded=(i == 1)):
                st.markdown(
                    f"**Filename:** `{src['source']}` | "
                    f"**Page:** `{page_str}` | "
                    f"**Vector L2 Distance:** `{src['distance']}`"
                )
                st.markdown("**Relevant Text Chunk:**")
                st.markdown(
                    f'<div class="source-box">{src["content"]}</div>',
                    unsafe_allow_html=True
                )

# Previous QA History
if len(st.session_state.qa_history) > 1:
    with st.expander("🕒 View Query History", expanded=False):
        for past in st.session_state.qa_history[1:]:
            st.markdown(f"**Q:** {past['question']}")
            st.markdown(f"**A:** {past['answer']}")
            st.markdown("---")
