#!/usr/bin/env python3
"""
Pinecone upload script for RAG chatbot.
Uploads processed document chunks to Pinecone with comprehensive tracking and error handling.
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.shared.utils import setup_logging, log_with_context, safe_json_load
from src.shared.models import PineconeChunkMetadata
from src.shared.vector_store import VectorStoreClient, create_chatbot_index
from src.local_processing.document_tracker import DocumentTracker

# Load environment variables
load_dotenv()


class PineconeUploader:
    """
    Handles uploading processed document chunks to Pinecone.
    """
    
    def __init__(
        self,
        processed_dir: str = "data/processed",
        logs_dir: str = "data/logs",
        index_name: str = "chatbot-rag-index",
        batch_size: int = 100,
        max_retries: int = 3
    ):
        """
        Initialize the Pinecone uploader.
        
        Args:
            processed_dir: Directory containing processed chunk files
            logs_dir: Directory for tracking logs
            index_name: Name of the Pinecone index
            batch_size: Number of vectors to upload per batch
            max_retries: Maximum number of retry attempts for failed uploads
        """
        self.logger = setup_logging(__name__)
        
        # Setup directories
        self.processed_dir = Path(processed_dir)
        self.logs_dir = Path(logs_dir)
        self.index_name = index_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Initialize components
        self.tracker = DocumentTracker(data_logs_dir=str(self.logs_dir))
        self.vector_client = VectorStoreClient()
        
        log_with_context(
            self.logger, 20,
            "Pinecone uploader initialized",
            processed_dir=str(self.processed_dir),
            index_name=index_name,
            batch_size=batch_size
        )
    
    def ensure_index_exists(self) -> bool:
        """
        Ensure the Pinecone index exists, create if necessary.
        
        Returns:
            True if index exists or was created successfully
        """
        try:
            # Check if index exists
            if self.vector_client.pc.has_index(self.index_name):
                log_with_context(
                    self.logger, 20,
                    f"Pinecone index already exists: {self.index_name}"
                )
                return True
            
            # Create index
            log_with_context(
                self.logger, 20,
                f"Creating Pinecone index: {self.index_name}"
            )
            
            success = create_chatbot_index(self.index_name)
            
            if success:
                # Wait for index to be ready
                log_with_context(
                    self.logger, 20,
                    "Waiting for index to be ready..."
                )
                
                # Wait up to 60 seconds for index to be ready
                for i in range(12):  # 12 * 5 = 60 seconds
                    try:
                        stats = self.vector_client.get_index_stats(self.index_name)
                        log_with_context(
                            self.logger, 20,
                            f"Index is ready: {self.index_name}",
                            dimension=stats.get('dimension', 'unknown')
                        )
                        return True
                    except Exception:
                        time.sleep(5)
                        continue
                
                # If we get here, index might not be ready but let's try anyway
                log_with_context(
                    self.logger, 30,
                    "Index creation timeout, proceeding anyway"
                )
                return True
            
            return False
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error ensuring index exists: {e}",
                index_name=self.index_name
            )
            return False
    
    def load_processed_chunks(self) -> List[Dict[str, Any]]:
        """
        Load all processed chunk files from the processed directory.
        
        Returns:
            List of processed chunk data
        """
        try:
            if not self.processed_dir.exists():
                log_with_context(
                    self.logger, 30,
                    f"Processed directory does not exist: {self.processed_dir}"
                )
                return []
            
            # Find all processed JSON files
            processed_files = list(self.processed_dir.glob("*_processed.json"))
            
            if not processed_files:
                log_with_context(
                    self.logger, 30,
                    "No processed files found",
                    processed_dir=str(self.processed_dir)
                )
                return []
            
            all_chunks = []
            
            for file_path in processed_files:
                try:
                    data = safe_json_load(str(file_path), {})
                    
                    if data and 'chunks' in data:
                        # Add file metadata to each chunk
                        for chunk in data['chunks']:
                            chunk['source_file'] = file_path.name
                            chunk['processing_session'] = data.get('processing_session')
                            chunk['processed_date'] = data.get('processed_date')
                        
                        all_chunks.extend(data['chunks'])
                        
                        log_with_context(
                            self.logger, 10,
                            f"Loaded {len(data['chunks'])} chunks from {file_path.name}"
                        )
                    
                except Exception as e:
                    log_with_context(
                        self.logger, 40,
                        f"Error loading processed file: {e}",
                        file_path=str(file_path)
                    )
                    continue
            
            log_with_context(
                self.logger, 20,
                f"Loaded {len(all_chunks)} total chunks from {len(processed_files)} files"
            )
            
            return all_chunks
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error loading processed chunks: {e}"
            )
            return []
    
    def verify_chunk_data(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify chunk data integrity before upload.
        
        Args:
            chunks: List of chunk data
            
        Returns:
            List of verified chunks
        """
        verified_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Check required fields
                required_fields = ['chunk_id', 'text', 'metadata']
                missing_fields = [f for f in required_fields if f not in chunk]
                
                if missing_fields:
                    log_with_context(
                        self.logger, 40,
                        f"Chunk missing required fields: {missing_fields}",
                        chunk_index=i,
                        chunk_id=chunk.get('chunk_id', 'unknown')
                    )
                    continue
                
                # Check text content
                if not chunk['text'] or not isinstance(chunk['text'], str):
                    log_with_context(
                        self.logger, 40,
                        "Chunk has invalid text content",
                        chunk_index=i,
                        chunk_id=chunk.get('chunk_id', 'unknown')
                    )
                    continue
                
                # Check metadata
                if not chunk['metadata'] or not isinstance(chunk['metadata'], dict):
                    log_with_context(
                        self.logger, 40,
                        "Chunk has invalid metadata",
                        chunk_index=i,
                        chunk_id=chunk.get('chunk_id', 'unknown')
                    )
                    continue
                
                verified_chunks.append(chunk)
                
            except Exception as e:
                log_with_context(
                    self.logger, 40,
                    f"Error verifying chunk: {e}",
                    chunk_index=i
                )
                continue
        
        log_with_context(
            self.logger, 20,
            f"Verified {len(verified_chunks)} of {len(chunks)} chunks"
        )
        
        return verified_chunks
    
    def prepare_vectors_for_upload(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare chunk data for Pinecone upload by creating embeddings.
        
        Args:
            chunks: List of verified chunk data
            
        Returns:
            List of vectors ready for Pinecone upload
        """
        try:
            vectors = []
            
            # Extract texts for batch embedding generation
            texts = [chunk['text'] for chunk in chunks]
            chunk_ids = [chunk['chunk_id'] for chunk in chunks]
            
            log_with_context(
                self.logger, 20,
                f"Generating embeddings for {len(texts)} chunks"
            )
            
            # Generate embeddings in batches to avoid rate limits
            batch_size = 100  # OpenAI batch limit
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    batch_embeddings = self.vector_client.generate_embeddings(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    
                    log_with_context(
                        self.logger, 10,
                        f"Generated embeddings for batch {i//batch_size + 1}",
                        batch_size=len(batch_texts)
                    )
                    
                    # Small delay between batches
                    if i + batch_size < len(texts):
                        time.sleep(0.5)
                        
                except Exception as e:
                    log_with_context(
                        self.logger, 40,
                        f"Error generating embeddings for batch: {e}",
                        batch_start=i,
                        batch_size=len(batch_texts)
                    )
                    raise
            
            # Prepare vectors for Pinecone
            for chunk, embedding in zip(chunks, all_embeddings):
                vector_data = {
                    "id": chunk['chunk_id'],
                    "values": embedding,
                    "metadata": {
                        "text": chunk['text'],
                        "source_document": chunk['metadata'].get('source_document'),
                        "chunk_index": chunk['metadata'].get('chunk_index'),
                        "char_start": chunk['metadata'].get('char_start'),
                        "char_end": chunk['metadata'].get('char_end'),
                        "processed_date": chunk.get('processed_date'),
                        "processing_session": chunk.get('processing_session')
                    }
                }
                vectors.append(vector_data)
            
            log_with_context(
                self.logger, 20,
                f"Prepared {len(vectors)} vectors for upload"
            )
            
            return vectors
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error preparing vectors for upload: {e}"
            )
            raise
    
    def upload_vectors_with_retry(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upload vectors to Pinecone with retry logic.
        
        Args:
            vectors: List of vectors to upload
            
        Returns:
            Upload results dictionary
        """
        upload_results = {
            "total_vectors": len(vectors),
            "uploaded_count": 0,
            "failed_count": 0,
            "retry_attempts": 0,
            "errors": []
        }
        
        try:
            for attempt in range(self.max_retries):
                try:
                    upload_results["retry_attempts"] = attempt + 1
                    
                    log_with_context(
                        self.logger, 20,
                        f"Upload attempt {attempt + 1}/{self.max_retries}",
                        vector_count=len(vectors)
                    )
                    
                    # Upload vectors
                    result = self.vector_client.upsert_vectors(
                        index_name=self.index_name,
                        vectors=vectors,
                        batch_size=self.batch_size
                    )
                    
                    if result.get("success"):
                        upload_results["uploaded_count"] = result.get("upserted_count", 0)
                        
                        log_with_context(
                            self.logger, 20,
                            f"Successfully uploaded vectors on attempt {attempt + 1}",
                            uploaded_count=upload_results["uploaded_count"]
                        )
                        
                        return upload_results
                    else:
                        raise Exception("Upload returned success=False")
                        
                except Exception as e:
                    error_msg = f"Upload attempt {attempt + 1} failed: {e}"
                    upload_results["errors"].append(error_msg)
                    
                    log_with_context(
                        self.logger, 40,
                        error_msg,
                        attempt=attempt + 1,
                        max_retries=self.max_retries
                    )
                    
                    if attempt < self.max_retries - 1:
                        # Wait before retry (exponential backoff)
                        wait_time = 2 ** attempt
                        log_with_context(
                            self.logger, 20,
                            f"Waiting {wait_time} seconds before retry"
                        )
                        time.sleep(wait_time)
                    else:
                        upload_results["failed_count"] = len(vectors)
                        break
            
            return upload_results
            
        except Exception as e:
            upload_results["errors"].append(f"Unexpected error: {e}")
            upload_results["failed_count"] = len(vectors)
            
            log_with_context(
                self.logger, 40,
                f"Unexpected error during upload: {e}"
            )
            
            return upload_results
    
    def verify_upload(self, uploaded_vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify successful upload by querying Pinecone.
        
        Args:
            uploaded_vectors: List of vectors that were uploaded
            
        Returns:
            Verification results
        """
        verification_results = {
            "total_checked": 0,
            "found_count": 0,
            "missing_count": 0,
            "missing_ids": []
        }
        
        try:
            # Check a sample of uploaded vectors (max 10 for verification)
            sample_size = min(10, len(uploaded_vectors))
            sample_vectors = uploaded_vectors[:sample_size]
            
            index = self.vector_client.get_index(self.index_name)
            
            for vector in sample_vectors:
                try:
                    # Query Pinecone for this vector ID
                    result = index.fetch(ids=[vector["id"]])
                    
                    if vector["id"] in result.get("vectors", {}):
                        verification_results["found_count"] += 1
                    else:
                        verification_results["missing_count"] += 1
                        verification_results["missing_ids"].append(vector["id"])
                    
                    verification_results["total_checked"] += 1
                    
                except Exception as e:
                    log_with_context(
                        self.logger, 40,
                        f"Error verifying vector: {e}",
                        vector_id=vector["id"]
                    )
                    verification_results["missing_count"] += 1
                    verification_results["missing_ids"].append(vector["id"])
            
            log_with_context(
                self.logger, 20,
                "Upload verification completed",
                total_checked=verification_results["total_checked"],
                found=verification_results["found_count"],
                missing=verification_results["missing_count"]
            )
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error during upload verification: {e}"
            )
        
        return verification_results
    
    def update_tracking_status(self, chunks: List[Dict[str, Any]], upload_successful: bool):
        """
        Update document tracking with upload status.
        
        Args:
            chunks: List of processed chunks
            upload_successful: Whether upload was successful
        """
        try:
            # Group chunks by source document
            docs_by_name = {}
            for chunk in chunks:
                doc_name = chunk['metadata'].get('source_document')
                if doc_name:
                    if doc_name not in docs_by_name:
                        docs_by_name[doc_name] = []
                    docs_by_name[doc_name].append(chunk)
            
            # Update status for each document
            for doc_name, doc_chunks in docs_by_name.items():
                if doc_name in self.tracker.ingestion_log.documents:
                    doc_meta = self.tracker.ingestion_log.documents[doc_name]
                    doc_meta.status = "uploaded" if upload_successful else "upload_failed"
                    
                    log_with_context(
                        self.logger, 20,
                        f"Updated tracking status for {doc_name}",
                        status=doc_meta.status,
                        chunk_count=len(doc_chunks)
                    )
            
            # Save updated tracking data
            self.tracker._save_tracking_data()
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error updating tracking status: {e}"
            )
    
    def upload_all_chunks(self) -> Dict[str, Any]:
        """
        Upload all processed chunks to Pinecone.
        
        Returns:
            Comprehensive upload report
        """
        try:
            start_time = datetime.now()
            
            log_with_context(
                self.logger, 20,
                "Starting Pinecone upload process"
            )
            
            # Ensure index exists
            if not self.ensure_index_exists():
                raise Exception("Failed to ensure Pinecone index exists")
            
            # Load processed chunks
            chunks = self.load_processed_chunks()
            
            if not chunks:
                return {
                    "success": False,
                    "message": "No processed chunks found to upload",
                    "total_chunks": 0
                }
            
            # Verify chunk data
            verified_chunks = self.verify_chunk_data(chunks)
            
            if not verified_chunks:
                return {
                    "success": False,
                    "message": "No valid chunks found after verification",
                    "total_chunks": len(chunks),
                    "verified_chunks": 0
                }
            
            # Prepare vectors for upload
            vectors = self.prepare_vectors_for_upload(verified_chunks)
            
            # Upload vectors
            upload_results = self.upload_vectors_with_retry(vectors)
            
            # Verify upload
            verification_results = self.verify_upload(vectors) if upload_results["uploaded_count"] > 0 else {}
            
            # Update tracking status
            upload_successful = upload_results["uploaded_count"] > 0
            self.update_tracking_status(verified_chunks, upload_successful)
            
            # Generate final report
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            report = {
                "success": upload_successful,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "index_name": self.index_name,
                "total_chunks_found": len(chunks),
                "verified_chunks": len(verified_chunks),
                "upload_results": upload_results,
                "verification_results": verification_results,
                "message": "Upload completed successfully" if upload_successful else "Upload failed"
            }
            
            log_with_context(
                self.logger, 20,
                "Pinecone upload process completed",
                success=upload_successful,
                uploaded_count=upload_results["uploaded_count"],
                duration_seconds=duration
            )
            
            return report
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error in upload_all_chunks: {e}"
            )
            
            return {
                "success": False,
                "message": f"Upload failed with error: {e}",
                "error": str(e)
            }


def main():
    """Main function to run Pinecone upload."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Upload processed chunks to Pinecone')
    parser.add_argument('--index-name', default='chatbot-rag-index',
                       help='Name of the Pinecone index (default: chatbot-rag-index)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for uploads (default: 100)')
    args = parser.parse_args()
    
    try:
        uploader = PineconeUploader(
            index_name=args.index_name,
            batch_size=args.batch_size
        )
        
        print("🚀 Starting Pinecone upload...")
        print(f"Index name: {args.index_name}")
        print(f"Batch size: {args.batch_size}")
        print()
        
        # Upload all chunks
        report = uploader.upload_all_chunks()
        
        # Print report
        print("📊 Upload Report:")
        print(f"Success: {'✅' if report['success'] else '❌'}")
        print(f"Total chunks found: {report.get('total_chunks_found', 0)}")
        print(f"Verified chunks: {report.get('verified_chunks', 0)}")
        
        if 'upload_results' in report:
            upload_results = report['upload_results']
            print(f"Uploaded: {upload_results.get('uploaded_count', 0)}")
            print(f"Failed: {upload_results.get('failed_count', 0)}")
            print(f"Retry attempts: {upload_results.get('retry_attempts', 0)}")
        
        if 'verification_results' in report and report['verification_results']:
            verification = report['verification_results']
            print(f"Verification: {verification.get('found_count', 0)}/{verification.get('total_checked', 0)} found")
        
        print(f"Duration: {report.get('duration_seconds', 0):.2f} seconds")
        print(f"Message: {report.get('message', 'No message')}")
        
        if not report['success']:
            if 'upload_results' in report and report['upload_results'].get('errors'):
                print("\nErrors:")
                for error in report['upload_results']['errors']:
                    print(f"  - {error}")
            
            sys.exit(1)
        else:
            print("\n✅ Upload completed successfully!")
            print(f"📍 Index: {args.index_name}")
            print("🎯 Ready for RAG queries!")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
