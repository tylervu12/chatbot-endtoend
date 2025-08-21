#!/usr/bin/env python3
"""
Document processing script for RAG chatbot.
Processes company documents with chunking, embeddings, and comprehensive tracking.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.shared.utils import setup_logging, log_with_context, get_file_hash
from src.shared.models import ChunkMetadata, PineconeChunkMetadata
from src.shared.vector_store import VectorStoreClient
from src.local_processing.document_tracker import DocumentTracker

# Load environment variables
load_dotenv()


class DocumentProcessor:
    """
    Processes documents with chunking, embeddings, and tracking.
    """
    
    def __init__(
        self,
        source_dir: str = "data/raw",
        processed_dir: str = "data/processed", 
        logs_dir: str = "data/logs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document processor.
        
        Args:
            source_dir: Directory containing source documents
            processed_dir: Directory to save processed chunks
            logs_dir: Directory for tracking logs
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.logger = setup_logging(__name__)
        
        # Setup directories
        self.source_dir = Path(source_dir)
        self.processed_dir = Path(processed_dir)
        self.logs_dir = Path(logs_dir)
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tracker = DocumentTracker(data_logs_dir=str(self.logs_dir))
        self.vector_client = VectorStoreClient()
        
        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        log_with_context(
            self.logger, 20,
            "Document processor initialized",
            source_dir=str(self.source_dir),
            processed_dir=str(self.processed_dir),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def scan_source_directory(self) -> List[Path]:
        """
        Scan source directory for .txt files.
        
        Returns:
            List of text file paths
        """
        try:
            if not self.source_dir.exists():
                raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")
            
            text_files = list(self.source_dir.glob("*.txt"))
            
            log_with_context(
                self.logger, 20,
                f"Scanned source directory: found {len(text_files)} .txt files",
                source_dir=str(self.source_dir)
            )
            
            return text_files
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error scanning source directory: {e}",
                source_dir=str(self.source_dir)
            )
            raise
    
    def read_document(self, file_path: Path) -> str:
        """
        Read document content from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            log_with_context(
                self.logger, 10,
                f"Read document: {file_path.name}",
                content_length=len(content),
                file_size=file_path.stat().st_size
            )
            
            return content
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error reading document: {e}",
                file_path=str(file_path)
            )
            raise
    
    def create_chunks(self, content: str, document_id: str, file_name: str) -> Tuple[List[str], List[ChunkMetadata]]:
        """
        Split document content into chunks with metadata, prepending filename to each chunk.
        
        Args:
            content: Document content
            document_id: Unique document identifier
            file_name: Original file name
            
        Returns:
            Tuple of (chunk_texts_with_context, chunk_metadata_objects)
        """
        try:
            # Split text into chunks
            raw_chunks = self.text_splitter.split_text(content)
            
            # Prepend filename context to each chunk
            filename_header = f"Source: {file_name}\n\n"
            chunks_with_context = []
            chunk_metadata_list = []
            current_position = 0
            
            for i, raw_chunk_text in enumerate(raw_chunks):
                # Add filename context to chunk
                chunk_with_context = filename_header + raw_chunk_text
                chunks_with_context.append(chunk_with_context)
                
                # Find the position of the original chunk in the source content
                chunk_start = content.find(raw_chunk_text, current_position)
                chunk_end = chunk_start + len(raw_chunk_text)
                
                # Create chunk metadata (using the original text for preview, not the contextualized version)
                chunk_meta = ChunkMetadata(
                    source_document=file_name,
                    chunk_index=i,
                    text_preview=raw_chunk_text[:100],  # First 100 chars of original text as preview
                    char_start=chunk_start,
                    char_end=chunk_end,
                    pinecone_id=f"{document_id}_chunk_{i}",
                    embedding_created=datetime.now()
                )
                
                chunk_metadata_list.append(chunk_meta)
                current_position = chunk_start + 1  # Move past this chunk for next search
            
            log_with_context(
                self.logger, 20,
                f"Created {len(chunks_with_context)} chunks for {file_name} with filename context",
                document_id=document_id,
                avg_chunk_size=sum(len(c) for c in chunks_with_context) // len(chunks_with_context) if chunks_with_context else 0,
                context_header_length=len(filename_header)
            )
            
            return chunks_with_context, chunk_metadata_list
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error creating chunks: {e}",
                document_id=document_id,
                content_length=len(content)
            )
            raise
    
    def save_processed_chunks(
        self, 
        chunks: List[str], 
        chunk_metadata: List[ChunkMetadata], 
        document_id: str,
        session_id: str
    ) -> str:
        """
        Save processed chunks to disk with metadata.
        
        Args:
            chunks: List of chunk texts
            chunk_metadata: List of chunk metadata
            document_id: Document identifier
            session_id: Processing session ID
            
        Returns:
            Path to saved file
        """
        try:
            # Create output data structure
            processed_data = {
                "document_id": document_id,
                "processing_session": session_id,
                "processed_date": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "chunks": []
            }
            
            # Add chunks with metadata
            for i, (chunk_text, metadata) in enumerate(zip(chunks, chunk_metadata)):
                chunk_data = {
                    "chunk_index": i,
                    "chunk_id": metadata.pinecone_id,
                    "text": chunk_text,
                    "char_start": metadata.char_start,
                    "char_end": metadata.char_end,
                    "metadata": metadata.model_dump()
                }
                processed_data["chunks"].append(chunk_data)
            
            # Save to file
            output_file = self.processed_dir / f"{document_id}_processed.json"
            
            from src.shared.utils import safe_json_save
            success = safe_json_save(processed_data, str(output_file))
            
            if success:
                log_with_context(
                    self.logger, 20,
                    f"Saved processed chunks: {output_file.name}",
                    chunk_count=len(chunks),
                    file_size=output_file.stat().st_size if output_file.exists() else 0
                )
                return str(output_file)
            else:
                raise Exception("Failed to save processed chunks")
                
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error saving processed chunks: {e}",
                document_id=document_id
            )
            raise
    
    def process_document(self, file_path: Path, session_id: str) -> Dict[str, Any]:
        """
        Process a single document: read, chunk, and track.
        
        Args:
            file_path: Path to the document file
            session_id: Processing session ID
            
        Returns:
            Dictionary with processing results
        """
        try:
            file_name = file_path.name
            document_id = file_path.stem  # Filename without extension
            
            log_with_context(
                self.logger, 20,
                f"Processing document: {file_name}",
                document_id=document_id,
                session_id=session_id
            )
            
            # Read document content
            content = self.read_document(file_path)
            
            # Create chunks with filename context
            chunks, chunk_metadata = self.create_chunks(content, document_id, file_name)
            
            # Save processed chunks
            output_file = self.save_processed_chunks(chunks, chunk_metadata, document_id, session_id)
            
            # Prepare embeddings data for Pinecone
            pinecone_chunks = []
            for i, (chunk_text, metadata) in enumerate(zip(chunks, chunk_metadata)):
                pinecone_meta = PineconeChunkMetadata(
                    source_document=file_name,
                    chunk_index=i,
                    document_id=document_id,
                    chunk_id=metadata.pinecone_id,
                    text=chunk_text,
                    char_start=metadata.char_start,
                    char_end=metadata.char_end,
                    processed_date=datetime.now(),
                    file_hash=get_file_hash(str(file_path)),
                    processing_session=session_id
                )
                pinecone_chunks.append(pinecone_meta)
            
            # Update tracking
            pinecone_ids = [meta.pinecone_id for meta in chunk_metadata]
            self.tracker.update_chunk_mapping(chunk_metadata)
            self.tracker.log_document_processing(
                file_path=str(file_path),
                chunk_count=len(chunks),
                pinecone_ids=pinecone_ids,
                session_id=session_id,
                status="processed"
            )
            
            result = {
                "file_name": file_name,
                "document_id": document_id,
                "chunk_count": len(chunks),
                "chunks": pinecone_chunks,
                "pinecone_ids": pinecone_ids,
                "output_file": output_file,
                "success": True
            }
            
            log_with_context(
                self.logger, 20,
                f"Successfully processed document: {file_name}",
                chunk_count=len(chunks),
                document_id=document_id
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to process document {file_path.name}: {e}"
            log_with_context(
                self.logger, 40,
                error_msg,
                file_path=str(file_path),
                session_id=session_id
            )
            
            # Log error in session
            self.tracker.add_session_error(session_id, error_msg)
            
            return {
                "file_name": file_path.name,
                "success": False,
                "error": str(e)
            }
    
    def process_all_documents(self) -> Dict[str, Any]:
        """
        Process all documents in the source directory.
        
        Returns:
            Dictionary with processing summary
        """
        try:
            # Start processing session
            session_id = self.tracker.start_processing_session()
            
            log_with_context(
                self.logger, 20,
                f"Starting document processing session: {session_id}"
            )
            
            # Scan for documents
            source_files = self.scan_source_directory()
            
            if not source_files:
                log_with_context(
                    self.logger, 30,
                    "No .txt files found in source directory",
                    source_dir=str(self.source_dir)
                )
                self.tracker.end_processing_session(session_id)
                return {
                    "session_id": session_id,
                    "total_files": 0,
                    "processed": 0,
                    "skipped": 0,
                    "errors": 0,
                    "results": []
                }
            
            # Process each document
            results = []
            processed_count = 0
            skipped_count = 0
            error_count = 0
            
            for file_path in source_files:
                try:
                    # Check if document needs processing (unless force flag is set)
                    if not getattr(self, 'force_reprocess', False) and not self.tracker.has_document_changed(str(file_path)):
                        log_with_context(
                            self.logger, 20,
                            f"Skipping unchanged document: {file_path.name}"
                        )
                        skipped_count += 1
                        results.append({
                            "file_name": file_path.name,
                            "success": True,
                            "skipped": True
                        })
                        continue
                    
                    # Process the document
                    result = self.process_document(file_path, session_id)
                    results.append(result)
                    
                    if result["success"]:
                        processed_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_msg = f"Unexpected error processing {file_path.name}: {e}"
                    log_with_context(
                        self.logger, 40,
                        error_msg,
                        file_path=str(file_path)
                    )
                    
                    self.tracker.add_session_error(session_id, error_msg)
                    error_count += 1
                    
                    results.append({
                        "file_name": file_path.name,
                        "success": False,
                        "error": str(e)
                    })
            
            # End processing session
            self.tracker.end_processing_session(session_id)
            
            # Cleanup deleted files
            self.tracker.cleanup_deleted_files(str(self.source_dir))
            
            summary = {
                "session_id": session_id,
                "total_files": len(source_files),
                "processed": processed_count,
                "skipped": skipped_count,
                "errors": error_count,
                "results": results
            }
            
            log_with_context(
                self.logger, 20,
                "Document processing session completed",
                session_id=session_id,
                total_files=len(source_files),
                processed=processed_count,
                skipped=skipped_count,
                errors=error_count
            )
            
            return summary
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error in process_all_documents: {e}"
            )
            raise


def main():
    """Main function to run document processing."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process documents for RAG chatbot')
    parser.add_argument('--force', action='store_true', 
                       help='Force reprocessing of all documents, ignoring tracking')
    args = parser.parse_args()
    
    try:
        processor = DocumentProcessor()
        
        # Set force reprocess flag if specified
        if args.force:
            processor.force_reprocess = True
            print("🔄 Force reprocessing enabled - will process all documents")
        
        print("🚀 Starting document processing...")
        print(f"Source directory: {processor.source_dir}")
        print(f"Processed directory: {processor.processed_dir}")
        print()
        
        # Process all documents
        summary = processor.process_all_documents()
        
        # Print summary
        print("📊 Processing Summary:")
        print(f"Session ID: {summary['session_id']}")
        print(f"Total files found: {summary['total_files']}")
        print(f"Processed: {summary['processed']}")
        print(f"Skipped (unchanged): {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print()
        
        # Print individual results
        if summary['results']:
            print("📄 Individual Results:")
            for result in summary['results']:
                status = "✅" if result['success'] else "❌"
                if result.get('skipped'):
                    status = "⏭️"
                
                print(f"{status} {result['file_name']}")
                if not result['success'] and 'error' in result:
                    print(f"    Error: {result['error']}")
                elif result['success'] and not result.get('skipped'):
                    print(f"    Chunks: {result.get('chunk_count', 'N/A')}")
        
        print()
        print("✅ Document processing completed!")
        print(f"📁 Processed files saved to: {processor.processed_dir}")
        print(f"📋 Tracking logs saved to: {processor.logs_dir}")
        
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
