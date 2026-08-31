"""
Utility script to generate sample PDF for testing DocuMind RAG Assistant.
Run this script using python to create sample_documents/ai_research_report.pdf
"""

import fitz # PyMuPDF
import os

def create_sample_pdf(output_path="ai_research_report.pdf"):
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    text_page1 = (
        "DocuMind AI Research Report - Volume 1: Retrieval Augmented Generation\n\n"
        "1. Executive Summary\n"
        "Retrieval-Augmented Generation (RAG) combines dense vector retrieval with large language models (LLMs) "
        "to deliver factual, grounded document question answering. By indexing text chunks into a vector store like FAISS, "
        "the architecture retrieves relevant context dynamically, mitigating hallucination.\n\n"
        "2. Vector Embeddings and Similarity Search\n"
        "Dense embedding models like sentence-transformers/all-MiniLM-L6-v2 project document chunks into a 384-dimensional "
        "vector space. FAISS (Facebook AI Similarity Search) provides high-performance L2 distance and inner product index "
        "searches, allowing sub-millisecond retrieval across millions of documents."
    )
    page1.insert_text((50, 50), text_page1, fontsize=11)
    
    # Page 2
    page2 = doc.new_page()
    text_page2 = (
        "DocuMind AI Research Report - Volume 1: Retrieval Augmented Generation (Continued)\n\n"
        "3. Chunking Strategies and Overlap Tuning\n"
        "Text chunking is critical for effective RAG. Selecting a chunk size of 800 characters with an overlap of 150 characters "
        "balances semantic completeness and embedding resolution. Overlapping text prevents context fragmentation across page and paragraph boundaries.\n\n"
        "4. Conclusion & Future Roadmap\n"
        "Empirical benchmarks demonstrate a 94.2% reduction in LLM hallucinations when using strict prompt grounding "
        "combined with vector distance thresholding at 1.35."
    )
    page2.insert_text((50, 50), text_page2, fontsize=11)
    
    doc.save(output_path)
    doc.close()
    print(f"Sample PDF successfully created at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(dir_path, "ai_research_report.pdf")
    create_sample_pdf(out_file)
