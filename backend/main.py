from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import os
from dotenv import load_dotenv
import logging

from services.rag_service import RAGService
from models.schemas import QueryRequest, QueryResponse, DocumentResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global rag_service

    # Startup
    try:
        logger.info("Initializing RAG service...")
        rag_service = RAGService()
        await rag_service.initialize()
        logger.info("RAG service initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise
    finally:
        # Cleanup (if needed)
        logger.info("Application shutdown")


app = FastAPI(
    title="AI Safety RAG Chatbot API",
    description="A RAG-powered chatbot for AI Safety and Ethics questions",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "AI Safety RAG Chatbot API is running"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "ai-safety-rag-api",
        "rag_service_initialized": rag_service is not None,
    }


@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    """Main chat endpoint for RAG queries"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        response = await rag_service.query(
            question=request.question, max_chunks=request.max_chunks or 5
        )
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentResponse])
async def get_documents():
    """Get information about ingested documents"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        return await rag_service.get_document_info()
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
