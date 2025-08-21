"""
Pydantic models for request/response validation and data structures.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User's question or message")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="Generated response to the user's question")
    sources: List[str] = Field(..., description="List of source document filenames used")
    chunks_used: int = Field(..., ge=0, description="Number of document chunks used in response")
    processing_time_ms: int = Field(..., ge=0, description="Total processing time in milliseconds")


class DocumentMetadata(BaseModel):
    """Metadata for tracking processed documents."""
    file_hash: str = Field(..., description="SHA256 hash of the file")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    last_modified: datetime = Field(..., description="Last modification timestamp")
    processed_date: datetime = Field(..., description="When the document was processed")
    chunk_count: int = Field(..., ge=0, description="Number of chunks created from this document")
    pinecone_ids: List[str] = Field(..., description="List of Pinecone IDs for all chunks")
    status: str = Field(..., description="Processing status: uploaded|processing|failed")


class ChunkMetadata(BaseModel):
    """Metadata for individual document chunks."""
    source_document: str = Field(..., description="Original document filename")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk within the document")
    text_preview: str = Field(..., max_length=100, description="First 100 characters of chunk text")
    char_start: int = Field(..., ge=0, description="Starting character position in original document")
    char_end: int = Field(..., ge=0, description="Ending character position in original document")
    pinecone_id: str = Field(..., description="Unique Pinecone ID for this chunk")
    embedding_created: datetime = Field(..., description="When the embedding was created")


class ProcessingSession(BaseModel):
    """Metadata for a document processing session."""
    session_id: str = Field(..., description="Unique session identifier")
    start_time: datetime = Field(..., description="Session start timestamp")
    end_time: Optional[datetime] = Field(None, description="Session end timestamp")
    documents_processed: List[str] = Field(..., description="List of documents processed in this session")
    new_documents: int = Field(..., ge=0, description="Number of new documents processed")
    skipped_duplicates: int = Field(..., ge=0, description="Number of duplicate documents skipped")
    total_chunks_created: int = Field(..., ge=0, description="Total chunks created in this session")
    total_tokens_used: int = Field(..., ge=0, description="Total tokens used for embeddings")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")


class IngestionLog(BaseModel):
    """Complete ingestion log structure."""
    documents: Dict[str, DocumentMetadata] = Field(default_factory=dict, description="Document metadata by filename")
    processing_stats: Dict[str, Any] = Field(default_factory=dict, description="Overall processing statistics")


class ChunkMapping(BaseModel):
    """Complete chunk mapping structure."""
    chunks: Dict[str, ChunkMetadata] = Field(default_factory=dict, description="Chunk metadata by chunk ID")


class ProcessingHistory(BaseModel):
    """Complete processing history structure."""
    sessions: List[ProcessingSession] = Field(default_factory=list, description="List of all processing sessions")


class PineconeChunkMetadata(BaseModel):
    """Metadata structure sent to Pinecone for each chunk."""
    source_document: str = Field(..., description="Original document filename")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk within the document")
    document_id: str = Field(..., description="Document identifier")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Full chunk text content")
    char_start: int = Field(..., ge=0, description="Starting character position in original document")
    char_end: int = Field(..., ge=0, description="Ending character position in original document")
    processed_date: datetime = Field(..., description="When the document was processed")
    file_hash: str = Field(..., description="SHA256 hash of source file")
    processing_session: str = Field(..., description="Processing session ID")
