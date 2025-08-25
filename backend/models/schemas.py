from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    max_chunks: Optional[int] = 5

class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    total_chunks_found: int
    processing_time: float

class DocumentResponse(BaseModel):
    filename: str
    chunk_count: int
    file_size: int
    ingested_at: str
