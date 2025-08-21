"""
Comprehensive document tracking system for RAG chatbot.
Provides duplicate detection, processing history, and chunk lineage tracking.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import os

from ..shared.utils import (
    setup_logging, log_with_context, get_file_hash, get_file_metadata,
    safe_json_load, safe_json_save
)
from ..shared.models import (
    IngestionLog, ChunkMapping, ProcessingHistory, 
    DocumentMetadata, ChunkMetadata, ProcessingSession
)


class DocumentTracker:
    """
    Tracks document processing state, prevents duplicates, and maintains lineage.
    """
    
    def __init__(self, data_logs_dir: str = "data/logs"):
        """
        Initialize the document tracker.
        
        Args:
            data_logs_dir: Directory to store tracking logs
        """
        self.logger = setup_logging(__name__)
        self.data_logs_dir = Path(data_logs_dir)
        
        # Ensure logs directory exists
        self.data_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Define log file paths
        self.ingestion_log_path = self.data_logs_dir / "ingestion_log.json"
        self.chunk_mapping_path = self.data_logs_dir / "chunk_mapping.json"
        self.processing_history_path = self.data_logs_dir / "processing_history.json"
        
        # Load existing data
        self._load_tracking_data()
        
        log_with_context(
            self.logger, 20,
            "Document tracker initialized",
            logs_directory=str(self.data_logs_dir),
            existing_documents=len(self.ingestion_log.documents),
            existing_chunks=len(self.chunk_mapping.chunks),
            processing_sessions=len(self.processing_history.sessions)
        )
    
    def _load_tracking_data(self):
        """Load all tracking data from JSON files."""
        try:
            # Load ingestion log
            ingestion_data = safe_json_load(self.ingestion_log_path, {
                "documents": {},
                "processing_stats": {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "last_update": None
                }
            })
            self.ingestion_log = IngestionLog(**ingestion_data)
            
            # Load chunk mapping
            chunk_data = safe_json_load(self.chunk_mapping_path, {"chunks": {}})
            self.chunk_mapping = ChunkMapping(**chunk_data)
            
            # Load processing history
            history_data = safe_json_load(self.processing_history_path, {"sessions": []})
            self.processing_history = ProcessingHistory(**history_data)
            
            log_with_context(
                self.logger, 10,
                "Tracking data loaded successfully"
            )
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to load tracking data: {e}"
            )
            # Initialize with empty data if loading fails
            self.ingestion_log = IngestionLog()
            self.chunk_mapping = ChunkMapping()
            self.processing_history = ProcessingHistory()
    
    def _save_tracking_data(self):
        """Save all tracking data to JSON files."""
        try:
            # Save ingestion log
            safe_json_save(
                self.ingestion_log.model_dump(), 
                self.ingestion_log_path
            )
            
            # Save chunk mapping
            safe_json_save(
                self.chunk_mapping.model_dump(), 
                self.chunk_mapping_path
            )
            
            # Save processing history
            safe_json_save(
                self.processing_history.model_dump(), 
                self.processing_history_path
            )
            
            log_with_context(
                self.logger, 10,
                "Tracking data saved successfully"
            )
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to save tracking data: {e}"
            )
            raise
    
    def check_document_exists(self, file_path: str) -> Tuple[bool, Optional[DocumentMetadata]]:
        """
        Check if document has already been processed.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (exists, document_metadata)
        """
        try:
            file_name = os.path.basename(file_path)
            
            if file_name in self.ingestion_log.documents:
                doc_meta = self.ingestion_log.documents[file_name]
                
                log_with_context(
                    self.logger, 10,
                    f"Document found in tracking: {file_name}",
                    status=doc_meta.status,
                    chunk_count=doc_meta.chunk_count
                )
                
                return True, doc_meta
            
            return False, None
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error checking document existence: {e}",
                file_path=file_path
            )
            return False, None
    
    def has_document_changed(self, file_path: str) -> bool:
        """
        Check if document has changed since last processing.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if document has changed or is new
        """
        try:
            exists, doc_meta = self.check_document_exists(file_path)
            
            if not exists:
                return True  # New document
            
            # Get current file metadata
            current_metadata = get_file_metadata(file_path)
            
            # Compare hash and modification time
            hash_changed = current_metadata['file_hash'] != doc_meta.file_hash
            size_changed = current_metadata['file_size'] != doc_meta.file_size
            time_changed = current_metadata['last_modified'] != doc_meta.last_modified
            
            changed = hash_changed or size_changed or time_changed
            
            if changed:
                log_with_context(
                    self.logger, 20,
                    f"Document has changed: {os.path.basename(file_path)}",
                    hash_changed=hash_changed,
                    size_changed=size_changed,
                    time_changed=time_changed
                )
            
            return changed
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error checking document changes: {e}",
                file_path=file_path
            )
            return True  # Assume changed if can't determine
    
    def start_processing_session(self) -> str:
        """
        Start a new processing session.
        
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        session = ProcessingSession(
            session_id=session_id,
            start_time=datetime.now(),
            documents_processed=[],
            new_documents=0,
            skipped_duplicates=0,
            total_chunks_created=0,
            total_tokens_used=0,
            errors=[]
        )
        
        self.processing_history.sessions.append(session)
        
        log_with_context(
            self.logger, 20,
            f"Started processing session: {session_id}"
        )
        
        return session_id
    
    def end_processing_session(self, session_id: str):
        """
        End a processing session.
        
        Args:
            session_id: Session ID to end
        """
        try:
            # Find the session
            session = None
            for s in self.processing_history.sessions:
                if s.session_id == session_id:
                    session = s
                    break
            
            if session:
                session.end_time = datetime.now()
                
                log_with_context(
                    self.logger, 20,
                    f"Ended processing session: {session_id}",
                    documents_processed=session.new_documents,
                    chunks_created=session.total_chunks_created,
                    duration=(session.end_time - session.start_time).total_seconds()
                )
            
            # Save the updated data
            self._save_tracking_data()
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error ending processing session: {e}",
                session_id=session_id
            )
    
    def log_document_processing(
        self, 
        file_path: str, 
        chunk_count: int, 
        pinecone_ids: List[str],
        session_id: str,
        status: str = "uploaded"
    ):
        """
        Log successful document processing.
        
        Args:
            file_path: Path to the processed document
            chunk_count: Number of chunks created
            pinecone_ids: List of Pinecone IDs for the chunks
            session_id: Processing session ID
            status: Processing status
        """
        try:
            file_name = os.path.basename(file_path)
            file_metadata = get_file_metadata(file_path)
            
            # Create document metadata
            doc_meta = DocumentMetadata(
                file_hash=file_metadata['file_hash'],
                file_size=file_metadata['file_size'],
                last_modified=file_metadata['last_modified'],
                processed_date=datetime.now(),
                chunk_count=chunk_count,
                pinecone_ids=pinecone_ids,
                status=status
            )
            
            # Add to ingestion log
            self.ingestion_log.documents[file_name] = doc_meta
            
            # Update processing stats
            self.ingestion_log.processing_stats = {
                "total_documents": len(self.ingestion_log.documents),
                "total_chunks": sum(doc.chunk_count for doc in self.ingestion_log.documents.values()),
                "last_update": datetime.now().isoformat()
            }
            
            # Update session
            self._update_session_stats(session_id, file_name, chunk_count, len(pinecone_ids))
            
            log_with_context(
                self.logger, 20,
                f"Logged document processing: {file_name}",
                chunk_count=chunk_count,
                status=status,
                session_id=session_id
            )
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error logging document processing: {e}",
                file_path=file_path
            )
            raise
    
    def update_chunk_mapping(self, chunks: List[ChunkMetadata]):
        """
        Update chunk mapping with new chunk metadata.
        
        Args:
            chunks: List of chunk metadata objects
        """
        try:
            for chunk in chunks:
                self.chunk_mapping.chunks[chunk.pinecone_id] = chunk
            
            log_with_context(
                self.logger, 20,
                f"Updated chunk mapping with {len(chunks)} chunks"
            )
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error updating chunk mapping: {e}"
            )
            raise
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            stats = {
                "total_documents": len(self.ingestion_log.documents),
                "total_chunks": len(self.chunk_mapping.chunks),
                "processing_sessions": len(self.processing_history.sessions),
                "last_update": self.ingestion_log.processing_stats.get("last_update"),
                "documents_by_status": {},
                "recent_activity": []
            }
            
            # Count documents by status
            for doc in self.ingestion_log.documents.values():
                status = doc.status
                stats["documents_by_status"][status] = stats["documents_by_status"].get(status, 0) + 1
            
            # Get recent sessions (last 5)
            recent_sessions = sorted(
                self.processing_history.sessions,
                key=lambda x: x.start_time,
                reverse=True
            )[:5]
            
            for session in recent_sessions:
                duration = 0
                if session.end_time:
                    duration = (session.end_time - session.start_time).total_seconds()
                
                stats["recent_activity"].append({
                    "session_id": session.session_id,
                    "start_time": session.start_time.isoformat(),
                    "documents_processed": session.new_documents,
                    "chunks_created": session.total_chunks_created,
                    "duration_seconds": duration
                })
            
            return stats
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error getting processing stats: {e}"
            )
            return {}
    
    def cleanup_deleted_files(self, source_directory: str = "data/raw"):
        """
        Remove tracking for files that no longer exist in the source directory.
        
        Args:
            source_directory: Directory containing source documents
        """
        try:
            source_path = Path(source_directory)
            
            if not source_path.exists():
                log_with_context(
                    self.logger, 30,
                    f"Source directory does not exist: {source_directory}"
                )
                return
            
            # Get list of current files
            current_files = set()
            for file_path in source_path.glob("*.txt"):
                current_files.add(file_path.name)
            
            # Find documents to remove
            documents_to_remove = []
            for file_name in self.ingestion_log.documents.keys():
                if file_name not in current_files:
                    documents_to_remove.append(file_name)
            
            # Remove deleted files from tracking
            chunks_to_remove = []
            for file_name in documents_to_remove:
                doc_meta = self.ingestion_log.documents[file_name]
                
                # Collect chunk IDs to remove
                chunks_to_remove.extend(doc_meta.pinecone_ids)
                
                # Remove document from ingestion log
                del self.ingestion_log.documents[file_name]
                
                log_with_context(
                    self.logger, 20,
                    f"Removed deleted file from tracking: {file_name}",
                    chunks_removed=len(doc_meta.pinecone_ids)
                )
            
            # Remove chunks from mapping
            for chunk_id in chunks_to_remove:
                if chunk_id in self.chunk_mapping.chunks:
                    del self.chunk_mapping.chunks[chunk_id]
            
            # Update processing stats
            self.ingestion_log.processing_stats = {
                "total_documents": len(self.ingestion_log.documents),
                "total_chunks": len(self.chunk_mapping.chunks),
                "last_update": datetime.now().isoformat()
            }
            
            if documents_to_remove:
                log_with_context(
                    self.logger, 20,
                    f"Cleanup completed: removed {len(documents_to_remove)} deleted files",
                    chunks_removed=len(chunks_to_remove)
                )
                
                # Save updated data
                self._save_tracking_data()
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error during cleanup: {e}"
            )
    
    def _update_session_stats(self, session_id: str, file_name: str, chunk_count: int, tokens_used: int):
        """Update session statistics."""
        for session in self.processing_history.sessions:
            if session.session_id == session_id:
                session.documents_processed.append(file_name)
                session.new_documents += 1
                session.total_chunks_created += chunk_count
                session.total_tokens_used += tokens_used
                break
    
    def add_session_error(self, session_id: str, error_message: str):
        """
        Add an error to the current session.
        
        Args:
            session_id: Session ID
            error_message: Error message to log
        """
        for session in self.processing_history.sessions:
            if session.session_id == session_id:
                session.errors.append(f"{datetime.now().isoformat()}: {error_message}")
                break
        
        log_with_context(
            self.logger, 40,
            f"Added error to session {session_id}: {error_message}"
        )
