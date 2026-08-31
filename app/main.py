from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_ingestion import router as ingestion_router
from app.api.routes_query import router as query_router
from app.core.config import settings
from app.core.exceptions import DocuMindException, documind_exception_handler
from app.core.logging import logger
from app.retrieval.faiss_store import get_vector_store
from app.schemas.response import HealthResponse


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-Grade Multi-Step RAG Knowledge Assistant powered by FastAPI, LangGraph, FAISS, and Google Gemini.",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handler
app.add_exception_handler(DocuMindException, documind_exception_handler)

# Include API routers
app.include_router(ingestion_router, prefix=settings.API_PREFIX)
app.include_router(query_router, prefix=settings.API_PREFIX)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Health check and system telemetry"
)
async def health_check():
    """Returns application health, indexed vector store stats, and active provider info."""
    vector_store = get_vector_store()
    docs = vector_store.get_indexed_documents()

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        vector_store_indexed_documents=len(docs),
        vector_store_total_chunks=vector_store.index.ntotal,
        llm_provider=settings.LLM_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL_NAME
    )


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}...")
    settings.ensure_directories()
    # Pre-warm vector store from disk
    vector_store = get_vector_store()
    logger.info(f"Vector store initialized with {vector_store.index.ntotal} total chunks.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
