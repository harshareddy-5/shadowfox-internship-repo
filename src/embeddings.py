"""
Embeddings Module.

Loads and caches the sentence-transformers/all-MiniLM-L6-v2 model using Streamlit's
cache_resource decorator for optimized memory reuse.
"""

import streamlit as st

# Attempt import from langchain_huggingface first, falling back to langchain_community
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Loads and caches the Hugging Face SentenceTransformer embedding model.
    Using @st.cache_resource ensures the model weights are loaded into memory only once.

    Model: sentence-transformers/all-MiniLM-L6-v2
    - Output vector size: 384 dimensions
    - Optimized for speed, efficiency, and high semantic retrieval performance.

    Args:
        model_name (str): Name of the HuggingFace model repository.

    Returns:
        HuggingFaceEmbeddings: Instantiated embedding model.
    """
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
