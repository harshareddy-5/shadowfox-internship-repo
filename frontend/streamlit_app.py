import os
import json
import time
import requests
import streamlit as st

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")
HEALTH_URL = API_BASE_URL.replace("/api/v1", "") + "/health"

st.set_page_config(
    page_title="DocuMind AI — Production RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for professional enterprise look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-grounded {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-ungrounded {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .citation-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 8px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def fetch_health():
    """Fetch health status from FastAPI backend."""
    try:
        res = requests.get(HEALTH_URL, timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        return None
    return None


def fetch_documents():
    """Fetch list of indexed documents."""
    try:
        res = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if res.status_code == 200:
            return res.json().get("documents", [])
    except Exception as e:
        st.sidebar.error(f"Backend offline or unreachable: {str(e)}")
    return []


def upload_document_api(file_obj, chunk_size, chunk_overlap):
    """Upload document to FastAPI backend."""
    try:
        files = {"file": (file_obj.name, file_obj.getvalue(), file_obj.type)}
        data = {}
        if chunk_size:
            data["chunk_size"] = chunk_size
        if chunk_overlap:
            data["chunk_overlap"] = chunk_overlap

        res = requests.post(f"{API_BASE_URL}/documents/upload", files=files, data=data, timeout=60)
        return res.status_code, res.json()
    except Exception as e:
        return 500, {"message": f"Upload request failed: {str(e)}"}


def delete_document_api(document_id):
    """Delete document by ID."""
    try:
        res = requests.delete(f"{API_BASE_URL}/documents/{document_id}", timeout=10)
        return res.status_code == 200
    except Exception:
        return False


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain--v1.png", width=64)
    st.title("DocuMind AI")
    st.caption("Production RAG Knowledge Assistant")
    st.markdown("---")

    # Document Upload Section
    st.subheader("📤 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Knowledge Source",
        type=["pdf", "txt", "md"],
        help="Upload PDF, TXT, or Markdown documents to build knowledge base."
    )

    with st.expander("⚙️ Advanced Ingestion Settings", expanded=False):
        custom_chunk_size = st.number_input("Chunk Size (tokens/chars)", min_value=200, max_value=2000, value=800, step=100)
        custom_chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=120, step=20)

    if uploaded_file is not None:
        if st.button("Ingest & Index Document", type="primary", use_container_width=True):
            status_container = st.empty()
            status_container.info("⏳ Uploading file...")
            time.sleep(0.3)
            status_container.info("⚙️ Extracting text & parsing sections...")
            time.sleep(0.3)
            status_container.info("🧠 Generating embeddings (BAAI/bge-small-en-v1.5)...")

            code, resp = upload_document_api(uploaded_file, custom_chunk_size, custom_chunk_overlap)

            if code == 201:
                status_container.success(f"✅ Ingested '{resp['filename']}' ({resp['total_chunks']} chunks)!")
                time.sleep(1)
                st.rerun()
            else:
                status_container.error(f"❌ Ingestion Failed: {resp.get('message', 'Unknown error')}")

    st.markdown("---")

    # Indexed Documents List & Filter
    st.subheader("📚 Knowledge Base")
    documents = fetch_documents()

    selected_doc_ids = []
    if documents:
        st.write(f"**Total Indexed Documents:** {len(documents)}")

        # Scoped Retrieval Selector
        doc_options = {f"{d['filename']} ({d['total_chunks']} chunks)": d['document_id'] for d in documents}
        selected_doc_names = st.multiselect(
            "Scoped Document Search Filter",
            options=list(doc_options.keys()),
            help="Select specific documents to limit search context. If empty, searches all documents."
        )
        selected_doc_ids = [doc_options[name] for name in selected_doc_names]

        with st.expander("Manage Documents", expanded=False):
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                col1.caption(f"📄 **{doc['filename']}** ({doc['total_chunks']} chunks)")
                if col2.button("🗑️", key=f"del_{doc['document_id']}"):
                    if delete_document_api(doc['document_id']):
                        st.success("Deleted!")
                        st.rerun()
    else:
        st.info("No documents indexed yet. Upload a file above!")

    st.markdown("---")

    # Retrieval Settings
    st.subheader("🛠️ RAG Pipeline Controls")
    top_k_val = st.slider("Retrieval Top K", min_value=3, max_value=20, value=10)
    enable_rerank_val = st.checkbox("Enable Neural Reranker (bge-reranker-base)", value=True)

    # Health Check Telemetry
    health_data = fetch_health()
    if health_data:
        st.markdown("---")
        st.caption(f"🟢 **Backend Status:** {health_data['status'].upper()}")
        st.caption(f"📊 **Vectors Indexed:** {health_data['vector_store_total_chunks']}")
        st.caption(f"⚡ **LLM Provider:** {health_data['llm_provider']}")

# --- MAIN PANEL ---
st.markdown('<div class="main-title">DocuMind AI Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Grounded Document Question Answering powered by LangGraph, FAISS & Google Gemini</div>',
    unsafe_allow_html=True
)

query_input = st.text_area(
    "Ask a question based on your uploaded documents:",
    placeholder="e.g., What is supervised learning? Or summarize the key findings in Section 3.",
    height=100
)

col_ask, col_clear = st.columns([1, 5])
ask_clicked = col_ask.button("🔍 Ask Question", type="primary", use_container_width=True)

if ask_clicked and query_input.strip():
    if not documents:
        st.warning("⚠️ Please upload at least one document before asking questions.")
    else:
        status_box = st.status("🚀 Processing RAG Pipeline...", expanded=True)

        status_box.write("1️⃣ Validating query & checking context...")
        status_box.write("2️⃣ Rewriting query for optimal retrieval...")
        status_box.write(f"3️⃣ Searching FAISS vector index (Selected scope: {len(selected_doc_ids) if selected_doc_ids else 'ALL'})...")
        status_box.write("4️⃣ Neural Reranking (BAAI/bge-reranker-base)...")
        status_box.write("5️⃣ Context Relevance Quality Filter...")
        status_box.write("6️⃣ Generating Grounded Answer with Google Gemini...")
        status_box.write("7️⃣ Groundedness & Hallucination Audit...")

        # Request to API
        payload = {
            "query": query_input,
            "selected_document_ids": selected_doc_ids if selected_doc_ids else None,
            "top_k": top_k_val,
            "enable_reranking": enable_rerank_val
        }

        try:
            res = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=90)
            status_box.update(label="✅ RAG Pipeline Complete!", state="complete", expanded=False)

            if res.status_code == 200:
                data = res.json()

                st.markdown("---")

                # Metrics summary row
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)

                is_grounded = data.get("grounded", False)
                quality = data.get("retrieval_quality", "UNKNOWN")
                exec_time = data.get("execution_time_seconds", 0.0)

                with mcol1:
                    badge_html = f'<span class="badge-grounded">GROUNDED</span>' if is_grounded else '<span class="badge-ungrounded">UNGROUNDED / REFUSAL</span>'
                    st.markdown(f"**Groundedness:** {badge_html}", unsafe_allow_html=True)

                with mcol2:
                    st.markdown(f"**Context Quality:** `{quality}`")

                with mcol3:
                    st.markdown(f"**Retrieved Chunks:** `{len(data.get('retrieved_chunks', []))}`")

                with mcol4:
                    st.markdown(f"**Latency:** `{exec_time}s`")

                st.markdown("### 📝 Answer")
                st.markdown(data.get("answer", "No answer produced."))

                # Query Rewriting Observability
                rewritten = data.get("rewritten_query")
                if rewritten and rewritten.strip() != query_input.strip():
                    st.info(f"💡 **Query Refinement:** Rewritten for search as: *\"{rewritten}\"*")

                # Citations
                citations = data.get("citations", [])
                if citations:
                    st.markdown("### 📍 Source Citations")
                    for cit in citations:
                        page_info = f", Page {cit['page']}" if cit.get('page') else ""
                        st.markdown(
                            f'<div class="citation-box">📄 <b>{cit["filename"]}</b>{page_info} | Chunk ID: <code>{cit["chunk_id"]}</code> | Relevance: <code>{cit["relevance_score"]:.4f}</code></div>',
                            unsafe_allow_html=True
                        )

                # Retrieved Context Observability
                chunks = data.get("retrieved_chunks", [])
                if chunks:
                    with st.expander("🔍 Inspect Retrieved Chunks (Observability)", expanded=False):
                        for idx, chunk in enumerate(chunks, start=1):
                            st.markdown(f"**Chunk #{idx}** — Document: `{chunk['filename']}` | Page: `{chunk.get('page', 'N/A')}` | Score: `{chunk.get('rerank_score') or chunk['similarity_score']:.4f}`")
                            st.code(chunk["text"], language="text")

                # Workflow Pipeline Stages Passed
                stages = data.get("stages_passed", [])
                if stages:
                    st.caption("⚙️ **Pipeline Stages Passed:** " + " ➔ ".join([f"`{s}`" for s in stages]))

            else:
                st.error(f"❌ Error {res.status_code}: {res.json().get('detail', 'Query execution failed')}")

        except Exception as e:
            status_box.update(label="❌ Pipeline Error", state="error")
            st.error(f"Failed to connect to RAG backend service: {str(e)}")

st.markdown("---")
st.caption("DocuMind AI © 2026 — Advanced AI Engineer Internship Task | Built with Python, FastAPI, LangGraph & FAISS")
