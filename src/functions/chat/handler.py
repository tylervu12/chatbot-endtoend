"""
RAG Chat API function for AWS Lambda.
Handles chat requests with retrieval-augmented generation using Pinecone and OpenAI.
"""
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import shared modules
import sys
import os
from pathlib import Path

# Import shared modules (copied locally for Lambda deployment)
from utils import setup_logging, log_with_context, get_secret_or_env
from models import ChatRequest, ChatResponse
from vector_store import VectorStoreClient


class RAGChatHandler:
    """
    Handles RAG chat requests with Pinecone retrieval and OpenAI generation.
    """
    
    def __init__(
        self,
        index_name: str = "chatbot-rag-index",
        top_k: int = 3,
        score_threshold: float = 0.5,
        llm_model: str = "gpt-4o"
    ):
        """
        Initialize the RAG chat handler.
        
        Args:
            index_name: Name of the Pinecone index
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score threshold
            llm_model: OpenAI model to use for generation
        """
        self.logger = setup_logging(__name__)
        self.index_name = index_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.llm_model = llm_model
        
        # Initialize vector store client
        self.vector_client = VectorStoreClient()
        
        # Initialize OpenAI client (already done in vector_client, but we need direct access)
        self.openai_client = self.vector_client.openai_client
        
        # Default response for when no relevant information is found
        self.no_info_response = "I don't have information to answer this question based on the available company documents."
        
        log_with_context(
            self.logger, 20,
            "RAG chat handler initialized",
            index_name=index_name,
            top_k=top_k,
            score_threshold=score_threshold,
            llm_model=llm_model
        )
    
    def retrieve_relevant_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks from Pinecone.
        
        Args:
            query: User's question
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Query Pinecone for similar chunks
            matches = self.vector_client.query_vectors(
                index_name=self.index_name,
                query_text=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
                include_metadata=True
            )
            
            # Filter and format results
            relevant_chunks = []
            for match in matches:
                if match["score"] >= self.score_threshold:
                    chunk_info = {
                        "id": match["id"],
                        "score": match["score"],
                        "text": match["metadata"].get("text", ""),
                        "source_document": match["metadata"].get("source_document", ""),
                        "chunk_index": match["metadata"].get("chunk_index", 0)
                    }
                    relevant_chunks.append(chunk_info)
            
            log_with_context(
                self.logger, 20,
                f"Retrieved {len(relevant_chunks)} relevant chunks",
                query_preview=query[:50] + "..." if len(query) > 50 else query,
                scores=[c["score"] for c in relevant_chunks],
                sources=list(set(c["source_document"] for c in relevant_chunks))
            )
            
            return relevant_chunks
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error retrieving chunks: {e}",
                query=query[:100]
            )
            return []
    
    def generate_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate response using OpenAI GPT-4 with retrieved chunks.
        
        Args:
            query: User's question
            chunks: Retrieved relevant chunks
            
        Returns:
            Generated response
        """
        try:
            if not chunks:
                return self.no_info_response
            
            # Prepare context from chunks
            context_parts = []
            sources = set()
            
            for i, chunk in enumerate(chunks, 1):
                context_parts.append(f"Document {i} ({chunk['source_document']}):\n{chunk['text']}\n")
                sources.add(chunk['source_document'])
            
            context = "\n".join(context_parts)
            
            # Create system prompt for grounded responses
            system_prompt = """You are a helpful company chatbot assistant. You must answer questions based ONLY on the provided company document context. 

IMPORTANT RULES:
1. Only use information explicitly stated in the provided documents
2. If the documents don't contain enough information to answer the question, say "I don't have enough information in the company documents to answer this question"
3. Always cite which document(s) you're referencing in your response
4. Be professional and helpful in tone
5. Do not make up or infer information not present in the documents
6. Keep responses concise but complete

The user's question will be followed by relevant excerpts from company documents."""
            
            # Create user prompt with context
            user_prompt = f"""Question: {query}

Company Documents Context:
{context}

Please answer the question based only on the information provided in the company documents above."""
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=500
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            log_with_context(
                self.logger, 20,
                "Generated response using LLM",
                query_preview=query[:50],
                response_length=len(generated_text),
                sources_used=list(sources),
                tokens_used=response.usage.total_tokens
            )
            
            return generated_text
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error generating response: {e}",
                query=query[:100]
            )
            return self.no_info_response
    
    def validate_answer(self, query: str, chunks: List[Dict[str, Any]], answer: str) -> bool:
        """
        Validate that the answer is grounded in the provided context.
        
        Args:
            query: Original user question
            chunks: Retrieved chunks used for context
            answer: Generated answer to validate
            
        Returns:
            True if answer is grounded in context, False otherwise
        """
        try:
            # Skip validation if this is the default "no info" response
            if answer == self.no_info_response:
                return True
            
            if not chunks:
                return False
            
            # Prepare context for validation
            context_text = "\n\n".join([chunk["text"] for chunk in chunks])
            
            # Create validation prompt
            validation_prompt = f"""You are a fact-checker. Your job is to determine if an answer is based solely on the provided context.

Context:
{context_text}

Question: {query}
Answer: {answer}

Is this answer based ONLY on information explicitly stated in the provided context? Consider:
1. Does the answer contain information not in the context?
2. Does the answer make claims beyond what the context supports?
3. Are all facts in the answer directly supported by the context?

Respond with only "true" or "false" - nothing else."""
            
            # Get validation response
            validation_response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.0,  # Very low temperature for consistent validation
                max_tokens=10
            )
            
            validation_result = validation_response.choices[0].message.content.strip().lower()
            is_valid = validation_result == "true"
            
            log_with_context(
                self.logger, 20,
                "Answer validation completed",
                is_valid=is_valid,
                validation_result=validation_result,
                answer_preview=answer[:100] + "..." if len(answer) > 100 else answer
            )
            
            return is_valid
            
        except Exception as e:
            log_with_context(
                self.logger, 40,
                f"Error validating answer: {e}",
                query=query[:100]
            )
            # Default to invalid if validation fails
            return False
    
    def process_chat_request(self, message: str) -> Dict[str, Any]:
        """
        Process a complete chat request through the RAG pipeline.
        
        Args:
            message: User's message/question
            
        Returns:
            Chat response with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            log_with_context(
                self.logger, 20,
                "Processing chat request",
                message_preview=message[:100] + "..." if len(message) > 100 else message
            )
            
            # Step 1: Retrieve relevant chunks
            chunks = self.retrieve_relevant_chunks(message)
            
            # Step 2: Check if we have relevant information
            if not chunks:
                log_with_context(
                    self.logger, 20,
                    "No relevant chunks found above threshold",
                    threshold=self.score_threshold
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                return {
                    "answer": self.no_info_response,
                    "sources": [],
                    "chunks_used": 0,
                    "processing_time_ms": processing_time
                }
            
            # Step 3: Generate response
            answer = self.generate_response(message, chunks)
            
            # Step 4: Validate answer
            is_valid = self.validate_answer(message, chunks, answer)
            
            if not is_valid:
                log_with_context(
                    self.logger, 30,
                    "Answer failed validation - returning no info response",
                    original_answer_preview=answer[:100]
                )
                answer = self.no_info_response
                chunks = []  # Clear chunks since answer is not grounded
            
            # Step 5: Prepare response
            sources = list(set(chunk["source_document"] for chunk in chunks)) if chunks else []
            processing_time = int((time.time() - start_time) * 1000)
            
            response = {
                "answer": answer,
                "sources": sources,
                "chunks_used": len(chunks),
                "processing_time_ms": processing_time
            }
            
            log_with_context(
                self.logger, 20,
                "Chat request processed successfully",
                answer_length=len(answer),
                sources_count=len(sources),
                chunks_used=len(chunks),
                processing_time_ms=processing_time,
                validation_passed=is_valid
            )
            
            return response
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            
            log_with_context(
                self.logger, 40,
                f"Error processing chat request: {e}",
                message=message[:100],
                processing_time_ms=processing_time
            )
            
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "sources": [],
                "chunks_used": 0,
                "processing_time_ms": processing_time
            }


# Global handler instance (for Lambda reuse)
_handler_instance = None


def get_handler() -> RAGChatHandler:
    """Get or create handler instance for Lambda reuse."""
    global _handler_instance
    
    if _handler_instance is None:
        # Get configuration from environment variables
        index_name = os.getenv("PINECONE_INDEX_NAME", "chatbot-rag-index")
        top_k = int(os.getenv("RAG_TOP_K", "3"))
        score_threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))
        llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
        
        _handler_instance = RAGChatHandler(
            index_name=index_name,
            top_k=top_k,
            score_threshold=score_threshold,
            llm_model=llm_model
        )
    
    return _handler_instance


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler for chat API requests.
    
    Args:
        event: API Gateway event
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    logger = setup_logging(__name__)
    
    try:
        # Log request
        log_with_context(
            logger, 20,
            "Received chat request",
            request_id=context.aws_request_id if hasattr(context, 'aws_request_id') else 'unknown'
        )
        
        # Parse request body
        try:
            if isinstance(event.get('body'), str):
                body = json.loads(event['body'])
            else:
                body = event.get('body', {})
            
            # Validate request
            chat_request = ChatRequest(**body)
            
        except Exception as e:
            log_with_context(
                logger, 40,
                f"Invalid request format: {e}",
                body=str(event.get('body', ''))[:200]
            )
            
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, x-api-key"
                },
                "body": json.dumps({
                    "error": "Invalid request format. Expected JSON with 'message' field."
                })
            }
        
        # Get handler and process request
        handler = get_handler()
        response_data = handler.process_chat_request(chat_request.message)
        
        # Create response model for validation
        chat_response = ChatResponse(**response_data)
        
        # Return successful response
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, x-api-key"
            },
            "body": json.dumps(chat_response.model_dump())
        }
        
    except Exception as e:
        log_with_context(
            logger, 40,
            f"Unexpected error in lambda handler: {e}",
            request_id=getattr(context, 'aws_request_id', 'unknown')
        )
        
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, x-api-key"
            },
            "body": json.dumps({
                "error": "Internal server error. Please try again later."
            })
        }


# For local testing
def test_handler_locally():
    """Test the handler locally without Lambda."""
    try:
        # Create a test handler
        handler = RAGChatHandler()
        
        # Test questions
        test_questions = [
            "What is your refund policy?",
            "How do I onboard new customers?",
            "What are the demo best practices?",
            "Tell me about your vacation policy",  # This should return no info
        ]
        
        print("🤖 Testing RAG Chat Handler Locally")
        print("=" * 50)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 30)
            
            response = handler.process_chat_request(question)
            
            print(f"💬 Answer: {response['answer']}")
            print(f"📚 Sources: {response['sources']}")
            print(f"📄 Chunks used: {response['chunks_used']}")
            print(f"⏱️  Processing time: {response['processing_time_ms']}ms")
        
        print("\n✅ Local testing completed!")
        
    except Exception as e:
        print(f"❌ Local testing failed: {e}")


if __name__ == "__main__":
    test_handler_locally()
