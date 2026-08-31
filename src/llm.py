"""
LLM Integration Module.

Manages Groq API interactions, system prompts, API key retrieval, and grounded answer generation
using Llama chat models.
"""

import os
from typing import Optional
from dotenv import load_dotenv
import groq

# Load environment variables from .env file
load_dotenv()

# Default Groq Llama model
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama3-8b-8192"

SYSTEM_PROMPT = """You are DocuMind, a strict document question-answering AI assistant.

Your task is to answer the user's question using ONLY the provided document context.

Strict Guidelines:
1. Base your answer STRICTLY on the retrieved document context provided.
2. Do NOT rely on outside knowledge, assumptions, or external facts.
3. If the answer cannot be determined from the context, clearly state:
   "I couldn't find sufficient information in the uploaded documents to answer this question."
4. Do not invent facts, citations, page numbers, or sources.
5. Keep your answer clear, direct, and well-structured.
"""


def get_groq_api_key() -> Optional[str]:
    """
    Retrieves the Groq API Key from environment variables or Streamlit secrets.

    Returns:
        Optional[str]: API Key string or None if not configured.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            import streamlit as st  # pylint: disable=import-outside-toplevel
            if "GROQ_API_KEY" in st.secrets:
                api_key = st.secrets["GROQ_API_KEY"]
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    if api_key and (api_key.strip() == "" or "your_groq_api_key" in api_key):
        return None

    return api_key


def generate_grounded_answer(
    context: str,
    question: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.0
) -> str:
    """
    Sends the retrieved document context and user question to the Groq LLM API
    to generate a context-grounded response.

    Args:
        context (str): Retrieved document text chunks with source metadata.
        question (str): User natural language question.
        api_key (Optional[str]): Groq API key. If None, retrieves from env/secrets.
        model_name (str): Model identifier (default llama-3.3-70b-versatile).
        temperature (float): Controls response variance. Default 0.0 for deterministic answers.

    Returns:
        str: Grounded answer text.

    Raises:
        ValueError: If API key is missing or invalid.
        RuntimeError: If Groq API request fails.
    """
    key = api_key or get_groq_api_key()

    if not key:
        raise ValueError(
            "Groq API Key is missing. Please set GROQ_API_KEY in your .env file "
            "or Streamlit sidebar."
        )

    user_prompt = (
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"---\n\n"
        f"USER QUESTION:\n{question}\n\n"
        "Answer the user question strictly based on the DOCUMENT CONTEXT above. "
        "If the context does not contain the answer, state clearly that the "
        "information is not available in the uploaded documents."
    )

    try:
        client = groq.Groq(api_key=key)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except (groq.NotFoundError, Exception) as model_err:  # pylint: disable=broad-exception-caught
            if "model_not_found" in str(model_err) or isinstance(model_err, groq.NotFoundError):
                response = client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1024
                )
                return response.choices[0].message.content.strip()
            raise model_err

    except groq.AuthenticationError as auth_err:
        raise ValueError("Invalid Groq API key provided. Please check your credentials.") from auth_err
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise RuntimeError(f"Groq API error: {str(e)}") from e
