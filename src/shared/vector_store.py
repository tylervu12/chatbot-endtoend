"""
Vector store utilities for Pinecone client and OpenAI embeddings.
Uses the new Pinecone Python SDK with OpenAI embeddings (bring your own vectors approach).
"""
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

from utils import setup_logging, log_with_context, get_secret_or_env
from models import PineconeChunkMetadata

# Load environment variables
load_dotenv()


class VectorStoreClient:
    """
    Pinecone vector store client with OpenAI embeddings.
    """
    
    def __init__(
        self, 
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536
    ):
        """
        Initialize the vector store client.
        
        Args:
            pinecone_api_key: Pinecone API key (if None, uses env var)
            openai_api_key: OpenAI API key (if None, uses env var) 
            embedding_model: OpenAI embedding model to use
            embedding_dimension: Dimension of the embedding vectors
        """
        self.logger = setup_logging(__name__)
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        
        # Initialize Pinecone client
        try:
            api_key = pinecone_api_key or get_secret_or_env(os.getenv("PINECONE_API_KEY_SECRET_ARN"), "PINECONE_API_KEY")
            self.pc = Pinecone(api_key=api_key)
            
            log_with_context(
                self.logger, 20, 
                "Pinecone client initialized successfully",
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension
            )
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to initialize Pinecone client: {e}"
            )
            raise
        
        # Initialize OpenAI client
        try:
            openai_key = openai_api_key or get_secret_or_env(os.getenv("OPENAI_API_KEY_SECRET_ARN"), "OPENAI_API_KEY")
            self.openai_client = openai.OpenAI(api_key=openai_key)
            
            log_with_context(
                self.logger, 20,
                "OpenAI client initialized successfully"
            )
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to initialize OpenAI client: {e}"
            )
            raise
    
    def create_index(
        self, 
        index_name: str, 
        cloud: str = "aws", 
        region: str = "us-east-1",
        metric: str = "cosine",
        deletion_protection: bool = False
    ) -> bool:
        """
        Create a new Pinecone index for storing vectors.
        
        Args:
            index_name: Name of the index to create
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
            metric: Distance metric (cosine, euclidean, dotproduct)
            deletion_protection: Whether to enable deletion protection
            
        Returns:
            True if created successfully, False if already exists
        """
        try:
            # Check if index already exists
            if self.pc.has_index(index_name):
                log_with_context(
                    self.logger, 20,
                    f"Index already exists: {index_name}"
                )
                return False
            
            # Create serverless index spec
            spec = ServerlessSpec(cloud=cloud, region=region)
            
            # Create the index
            self.pc.create_index(
                name=index_name,
                dimension=self.embedding_dimension,
                metric=metric,
                spec=spec,
                deletion_protection="enabled" if deletion_protection else "disabled"
            )
            
            log_with_context(
                self.logger, 20,
                f"Index created successfully: {index_name}",
                cloud=cloud,
                region=region,
                metric=metric,
                dimension=self.embedding_dimension
            )
            
            return True
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to create index {index_name}: {e}",
                cloud=cloud,
                region=region
            )
            raise
    
    def get_index(self, index_name: str):
        """
        Get a reference to an existing Pinecone index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Pinecone Index object
        """
        try:
            index = self.pc.Index(index_name)
            
            log_with_context(
                self.logger, 10,
                f"Connected to index: {index_name}"
            )
            
            return index
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to connect to index {index_name}: {e}"
            )
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # OpenAI embeddings API call
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            log_with_context(
                self.logger, 10,
                f"Generated embeddings for {len(texts)} texts",
                model=self.embedding_model,
                total_tokens=response.usage.total_tokens
            )
            
            return embeddings
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to generate embeddings: {e}",
                text_count=len(texts)
            )
            raise
    
    def upsert_vectors(
        self, 
        index_name: str, 
        vectors: List[Dict[str, Any]], 
        namespace: str = "",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone index in batches.
        
        Args:
            index_name: Name of the index
            vectors: List of vector dictionaries with id, values, and metadata
            namespace: Pinecone namespace (optional)
            batch_size: Number of vectors to upsert per batch
            
        Returns:
            Dictionary with upsert statistics
        """
        try:
            index = self.get_index(index_name)
            
            upserted_count = 0
            total_vectors = len(vectors)
            
            # Process in batches
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                
                # Upsert batch
                response = index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                
                upserted_count += response.upserted_count
                
                log_with_context(
                    self.logger, 10,
                    f"Upserted batch {i//batch_size + 1}",
                    batch_size=len(batch),
                    total_upserted=upserted_count,
                    total_vectors=total_vectors
                )
                
                # Small delay between batches to avoid rate limits
                time.sleep(0.1)
            
            log_with_context(
                self.logger, 20,
                f"Successfully upserted all vectors to {index_name}",
                total_upserted=upserted_count,
                namespace=namespace
            )
            
            return {
                "upserted_count": upserted_count,
                "total_vectors": total_vectors,
                "namespace": namespace,
                "success": True
            }
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to upsert vectors to {index_name}: {e}",
                vector_count=len(vectors),
                namespace=namespace
            )
            raise
    
    def query_vectors(
        self, 
        index_name: str, 
        query_text: str, 
        top_k: int = 5,
        namespace: str = "",
        score_threshold: float = 0.0,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the vector index for similar vectors.
        
        Args:
            index_name: Name of the index
            query_text: Text to search for
            top_k: Number of results to return
            namespace: Pinecone namespace (optional)
            score_threshold: Minimum similarity score threshold
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching vectors with metadata
        """
        try:
            index = self.get_index(index_name)
            
            # Generate embedding for query text
            query_embedding = self.generate_embeddings([query_text])[0]
            
            # Query the index
            query_response = index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata
            )
            
            # Filter by score threshold
            filtered_matches = []
            for match in query_response.matches:
                if match.score >= score_threshold:
                    filtered_matches.append({
                        "id": match.id,
                        "score": float(match.score),
                        "metadata": match.metadata if include_metadata else None
                    })
            
            log_with_context(
                self.logger, 20,
                f"Query completed for {index_name}",
                query_length=len(query_text),
                top_k=top_k,
                results_found=len(filtered_matches),
                score_threshold=score_threshold,
                namespace=namespace
            )
            
            return filtered_matches
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to query index {index_name}: {e}",
                query_text=query_text[:50] + "..." if len(query_text) > 50 else query_text
            )
            raise
    
    def delete_vectors(
        self, 
        index_name: str, 
        vector_ids: List[str], 
        namespace: str = ""
    ) -> bool:
        """
        Delete vectors from the index by IDs.
        
        Args:
            index_name: Name of the index
            vector_ids: List of vector IDs to delete
            namespace: Pinecone namespace (optional)
            
        Returns:
            True if deletion was successful
        """
        try:
            index = self.get_index(index_name)
            
            # Delete vectors
            index.delete(ids=vector_ids, namespace=namespace)
            
            log_with_context(
                self.logger, 20,
                f"Deleted {len(vector_ids)} vectors from {index_name}",
                namespace=namespace
            )
            
            return True
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to delete vectors from {index_name}: {e}",
                vector_count=len(vector_ids),
                namespace=namespace
            )
            raise
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Dictionary with index statistics
        """
        try:
            index = self.get_index(index_name)
            
            stats = index.describe_index_stats()
            
            stats_dict = {
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "total_vector_count": stats.total_vector_count,
                "namespaces": {}
            }
            
            # Add namespace statistics
            if stats.namespaces:
                for namespace, ns_stats in stats.namespaces.items():
                    stats_dict["namespaces"][namespace] = {
                        "vector_count": ns_stats.vector_count
                    }
            
            log_with_context(
                self.logger, 10,
                f"Retrieved stats for {index_name}",
                total_vectors=stats_dict["total_vector_count"],
                dimension=stats_dict["dimension"]
            )
            
            return stats_dict
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Failed to get stats for {index_name}: {e}"
            )
            raise
    
    def prepare_chunk_for_upsert(self, chunk_metadata: PineconeChunkMetadata) -> Dict[str, Any]:
        """
        Prepare a chunk for upsert to Pinecone.
        
        Args:
            chunk_metadata: Chunk metadata object
            
        Returns:
            Dictionary formatted for Pinecone upsert
        """
        # Generate embedding for the chunk text
        embedding = self.generate_embeddings([chunk_metadata.text])[0]
        
        # Prepare metadata (exclude text from metadata to avoid duplication)
        metadata = {
            "source_document": chunk_metadata.source_document,
            "chunk_index": chunk_metadata.chunk_index,
            "document_id": chunk_metadata.document_id,
            "char_start": chunk_metadata.char_start,
            "char_end": chunk_metadata.char_end,
            "processed_date": chunk_metadata.processed_date.isoformat(),
            "file_hash": chunk_metadata.file_hash,
            "processing_session": chunk_metadata.processing_session,
            "text": chunk_metadata.text  # Keep text in metadata for retrieval
        }
        
        return {
            "id": chunk_metadata.chunk_id,
            "values": embedding,
            "metadata": metadata
        }


# Convenience functions for common operations
def get_vector_store_client() -> VectorStoreClient:
    """
    Get a configured vector store client using environment variables.
    
    Returns:
        VectorStoreClient instance
    """
    return VectorStoreClient()


def create_chatbot_index(index_name: str = "chatbot-rag-index") -> bool:
    """
    Create the main chatbot index with standard configuration.
    
    Args:
        index_name: Name of the index to create
        
    Returns:
        True if created successfully, False if already exists
    """
    client = get_vector_store_client()
    return client.create_index(
        index_name=index_name,
        cloud="aws",
        region="us-east-1",
        metric="cosine",
        deletion_protection=False
    )
