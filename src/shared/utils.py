"""
Shared utility functions for AWS, logging, error handling, and file operations.
"""
import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError


# Configure structured logging
def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Setup structured logging with JSON format.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # JSON formatter for structured logging
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add extra fields if present
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
                
            return json.dumps(log_entry)
    
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def log_with_context(logger: logging.Logger, level: int, message: str, **kwargs):
    """
    Log message with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **kwargs: Additional context fields
    """
    # Create a LogRecord with extra fields
    record = logger.makeRecord(
        logger.name, level, '', 0, message, (), None
    )
    record.extra_fields = kwargs
    logger.handle(record)


# AWS Secrets Manager functions
def get_secrets_manager_client(region: str = 'us-east-1') -> boto3.client:
    """
    Get AWS Secrets Manager client.
    
    Args:
        region: AWS region
        
    Returns:
        Secrets Manager client
    """
    return boto3.client('secretsmanager', region_name=region)


def get_secret(secret_arn: str, region: str = 'us-east-1') -> str:
    """
    Retrieve secret value from AWS Secrets Manager.
    
    Args:
        secret_arn: ARN of the secret
        region: AWS region
        
    Returns:
        Secret value as string
        
    Raises:
        ClientError: If secret retrieval fails
        ValueError: If secret value is empty
    """
    logger = setup_logging()
    
    try:
        client = get_secrets_manager_client(region)
        response = client.get_secret_value(SecretId=secret_arn)
        
        secret_value = response.get('SecretString')
        if not secret_value:
            raise ValueError(f"Empty secret value for ARN: {secret_arn}")
            
        log_with_context(
            logger, logging.INFO, 
            "Successfully retrieved secret",
            secret_arn=secret_arn[:20] + "..." if len(secret_arn) > 20 else secret_arn
        )
        
        return secret_value
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        log_with_context(
            logger, logging.ERROR,
            f"Failed to retrieve secret: {error_code}",
            secret_arn=secret_arn,
            error_code=error_code
        )
        raise


def get_secret_or_env(secret_arn: Optional[str], env_var: str, region: str = 'us-east-1') -> str:
    """
    Get secret from AWS Secrets Manager or fall back to environment variable.
    
    Args:
        secret_arn: ARN of the secret (if None, uses env var)
        env_var: Environment variable name as fallback
        region: AWS region
        
    Returns:
        Secret value as string
        
    Raises:
        ValueError: If neither secret nor env var is available
    """
    logger = setup_logging()
    
    # Try environment variable first for local development
    env_value = os.getenv(env_var)
    if env_value:
        log_with_context(
            logger, logging.INFO,
            f"Using environment variable: {env_var}"
        )
        return env_value
    
    # Try AWS Secrets Manager if ARN provided
    if secret_arn:
        try:
            return get_secret(secret_arn, region)
        except Exception as e:
            log_with_context(
                logger, logging.WARNING,
                f"Failed to get secret, trying env var: {e}",
                secret_arn=secret_arn,
                env_var=env_var
            )
    
    # Final fallback attempt with env var
    env_value = os.getenv(env_var)
    if env_value:
        return env_value
        
    raise ValueError(f"Neither secret ARN nor environment variable {env_var} is available")


# File hashing utilities
def get_file_hash(file_path: str) -> str:
    """
    Generate SHA256 hash of a file for duplicate detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash as hex string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    logger = setup_logging()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        file_hash = sha256_hash.hexdigest()
        
        log_with_context(
            logger, logging.DEBUG,
            "Generated file hash",
            file_path=file_path,
            file_hash=file_hash[:16] + "..."
        )
        
        return file_hash
        
    except IOError as e:
        log_with_context(
            logger, logging.ERROR,
            f"Failed to read file for hashing: {e}",
            file_path=file_path
        )
        raise


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get file metadata including size, modification time, and hash.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = os.stat(file_path)
    
    return {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_size': stat.st_size,
        'last_modified': datetime.fromtimestamp(stat.st_mtime),
        'file_hash': get_file_hash(file_path)
    }


# Common error handling functions
class ChatbotError(Exception):
    """Base exception for chatbot-related errors."""
    pass


class DocumentProcessingError(ChatbotError):
    """Exception for document processing errors."""
    pass


class VectorStoreError(ChatbotError):
    """Exception for vector store operations."""
    pass


class APIError(ChatbotError):
    """Exception for API-related errors."""
    pass


def handle_api_error(e: Exception, context: str = "") -> Dict[str, Any]:
    """
    Handle API errors and return standardized error response.
    
    Args:
        e: Exception instance
        context: Additional context about where error occurred
        
    Returns:
        Standardized error response dictionary
    """
    logger = setup_logging()
    
    error_response = {
        'error': True,
        'error_type': type(e).__name__,
        'message': str(e),
        'context': context,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    log_with_context(
        logger, logging.ERROR,
        f"API error in {context}: {str(e)}",
        error_type=type(e).__name__,
        context=context
    )
    
    return error_response


def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Parsed JSON data or default value
    """
    logger = setup_logging()
    
    if not os.path.exists(file_path):
        log_with_context(
            logger, logging.INFO,
            f"JSON file not found, using default: {file_path}"
        )
        return default
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        log_with_context(
            logger, logging.DEBUG,
            f"Successfully loaded JSON file: {file_path}"
        )
        return data
        
    except (json.JSONDecodeError, IOError) as e:
        log_with_context(
            logger, logging.WARNING,
            f"Failed to load JSON file, using default: {e}",
            file_path=file_path
        )
        return default


def safe_json_save(data: Any, file_path: str) -> bool:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logging()
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        log_with_context(
            logger, logging.DEBUG,
            f"Successfully saved JSON file: {file_path}"
        )
        return True
        
    except (IOError, TypeError) as e:
        log_with_context(
            logger, logging.ERROR,
            f"Failed to save JSON file: {e}",
            file_path=file_path
        )
        return False
