import os
import sys
import json
import time
import logging
import argparse
import tiktoken
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any, Iterator
import tempfile
import uuid
import random

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.utils.gcs import GCSFetch
from amici.transformation.prompts import AMICI_EXTRACTION_PROMPT, AMICI_EXTRACTION_SCHEMA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenAI model pricing per 1M tokens
MODEL_PRICING = {
    "gpt-4.1": {"input": 1.00, "output": 4.00},
    "gpt-4.1-2025-04-14": {"input": 1.00, "output": 4.00},
    "gpt-4.1-mini": {"input": 0.20, "output": 0.80},
    "gpt-4.1-mini-2025-04-14": {"input": 0.20, "output": 0.80},
    "gpt-4.1-nano": {"input": 0.05, "output": 0.20},
    "gpt-4.1-nano-2025-04-14": {"input": 0.05, "output": 0.20},
    "gpt-4.5-preview": {"input": 37.50, "output": 75.00},
    "gpt-4.5-preview-2025-02-27": {"input": 37.50, "output": 75.00},
    "gpt-4o": {"input": 1.25, "output": 5.00},
    "gpt-4o-2024-08-06": {"input": 1.25, "output": 5.00},
    "gpt-4o-mini": {"input": 0.075, "output": 0.30},
    "gpt-4o-mini-2024-07-18": {"input": 0.075, "output": 0.30},
    "o1": {"input": 7.50, "output": 30.00},
    "o1-2024-12-17": {"input": 7.50, "output": 30.00},
    "o1-pro": {"input": 75.00, "output": 300.00},
    "o1-pro-2025-03-19": {"input": 75.00, "output": 300.00},
    "o3": {"input": 5.00, "output": 20.00},
    "o3-2025-04-16": {"input": 5.00, "output": 20.00},
    "o4-mini": {"input": 0.55, "output": 2.20},
    "o4-mini-2025-04-16": {"input": 0.55, "output": 2.20},
    "o3-mini": {"input": 0.55, "output": 2.20},
    "o3-mini-2025-01-31": {"input": 0.55, "output": 2.20},
    "o1-mini": {"input": 0.55, "output": 2.20},
    "o1-mini-2024-09-12": {"input": 0.55, "output": 2.20},
    "computer-use-preview": {"input": 1.50, "output": 6.00},
    "computer-use-preview-2025-03-11": {"input": 1.50, "output": 6.00}
}

def list_txt_files_from_gcs(bucket_name: str, prefix: str) -> List[str]:
    """
    List all text files from the specified GCS bucket and prefix.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix of files to list
        
    Returns:
        List of blob names for text files
    """
    logger.info(f"Listing text files from gs://{bucket_name}/{prefix}")
    gcs = GCSFetch(bucket_name)
    txt_files = []
    
    for blob in gcs.list_blobs(prefix):
        if blob.name.endswith('.txt'):
            txt_files.append(blob.name)
    
    logger.info(f"Found {len(txt_files)} text files")
    return txt_files

def read_blob_list_from_file(file_path: str) -> List[str]:
    """
    Read a list of blob names from a file, one blob name per line.
    
    Args:
        file_path: Path to the file containing blob names
        
    Returns:
        List of blob names
    """
    with open(file_path, "r") as f:
        # Read non-empty lines and strip whitespace
        blob_names = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Read {len(blob_names)} blob names from {file_path}")
    return blob_names

def fetch_txt_files_from_gcs(bucket_name: str, prefix: str = None, test_batch_size: int = None, 
                             specific_blobs: List[str] = None) -> Dict[str, str]:
    """
    Fetch text files from the specified GCS bucket.
    If specific_blobs is provided, fetches only those blobs (converting PDFs to corresponding TXT files).
    Otherwise, uses prefix to list and fetch files, with optional test_batch_size limit.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix of files to fetch (used if specific_blobs is None)
        test_batch_size: If set, limits the number of files fetched to this value
        specific_blobs: List of specific blob names to fetch
        
    Returns:
        Dictionary mapping filenames to text content
    """
    gcs = GCSFetch(bucket_name)
    text_files = {}
    
    if specific_blobs:
        logger.info(f"Fetching {len(specific_blobs)} specific text files")
        for blob_name in specific_blobs:
            # If the blob is a PDF, find the corresponding TXT file
            if blob_name.lower().endswith('.pdf'):
                txt_blob_name = blob_name[:-4] + '.txt'
                logger.info(f"Converting PDF blob {blob_name} to txt blob {txt_blob_name}")
            else:
                txt_blob_name = blob_name
            
            if not txt_blob_name.lower().endswith('.txt'):
                logger.warning(f"Skipping non-text blob: {blob_name}")
                continue
            
            try:
                logger.info(f"Fetching {txt_blob_name}")
                content, metadata = gcs.get_from_bucket(txt_blob_name)
                text_files[txt_blob_name] = content.decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to fetch {txt_blob_name}: {e}")
    else:
        # Original functionality for fetching by prefix
        txt_file_names = list_txt_files_from_gcs(bucket_name, prefix)
        
        if not txt_file_names:
            return {}
        
        # If test batch size is specified, randomly sample files
        if test_batch_size and test_batch_size < len(txt_file_names):
            logger.info(f"Sampling {test_batch_size} files for test batch")
            txt_file_names = random.sample(txt_file_names, test_batch_size)
        
        # Now fetch only the required files
        for blob_name in txt_file_names:
            logger.info(f"Fetching {blob_name}")
            content, metadata = gcs.get_from_bucket(blob_name)
            text_files[blob_name] = content.decode('utf-8')
    
    logger.info(f"Fetched {len(text_files)} text files")
    return text_files

def estimate_tokens_and_cost(texts: List[str], prompt: str, model: str = "gpt-4.1-nano") -> Tuple[int, float]:
    """
    Estimate the token count and cost for processing texts with the OpenAI API.
    
    Args:
        texts: List of text content to process
        prompt: The system prompt to use
        model: The model to use for estimation
        
    Returns:
        Tuple of (total_tokens, estimated_cost_usd)
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
    
    prompt_tokens = len(encoding.encode(prompt))
    
    # Calculate input tokens
    input_tokens = 0
    for text in texts:
        # Count tokens for this text + prompt
        text_tokens = len(encoding.encode(text))
        # Add tokens for this request (prompt + text)
        input_tokens += prompt_tokens + text_tokens
    
    # Estimate output tokens (rough estimate for JSON response)
    output_tokens = len(texts) * 500  # Rough estimate: 500 tokens per response
    
    # Get model pricing
    model_base = model.split('-2')[0]  # Strip version date if present
    if model_base in MODEL_PRICING:
        pricing = MODEL_PRICING[model_base]
    else:
        logger.warning(f"Pricing not found for model {model}. Using default pricing.")
        pricing = {"input": 0.05, "output": 0.20}  # Default to gpt-4.1-nano pricing
    
    # Calculate cost per token type
    input_cost = (input_tokens / 1000000) * pricing["input"]
    output_cost = (output_tokens / 1000000) * pricing["output"]
    
    total_tokens = input_tokens + output_tokens
    
    logger.info(f"Cost breakdown: Input tokens: {input_tokens:,}, Output tokens (est.): {output_tokens:,}")
    logger.info(f"Pricing for {model}: ${pricing['input']}/1M input tokens, ${pricing['output']}/1M output tokens")
    logger.info(f"Total cost: ${input_cost + output_cost:.2f}")
    
    return total_tokens, total_cost

def create_batch_requests(files_dict: Dict[str, str], prompt: str, model: str = "gpt-4.1-nano") -> str:
    """
    Create batch requests in JSONL format for OpenAI Batch API
    
    Args:
        files_dict: Dictionary mapping filenames to text content
        prompt: The system prompt to use
        model: OpenAI model to use
        
    Returns:
        JSONL string with batch requests
    """
    batch_lines = []
    
    for filename, text in files_dict.items():
        request_data = {
            "custom_id": filename,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.0
            }
        }
        batch_lines.append(json.dumps(request_data))
    
    return "\n".join(batch_lines)

def split_batch_requests(jsonl_content: str, max_size_bytes: int = 200 * 1024 * 1024) -> List[str]:
    """
    Split batch requests into multiple files if they exceed the maximum allowed size.
    
    Args:
        jsonl_content: JSONL content as a string
        max_size_bytes: Maximum allowed file size in bytes
        
    Returns:
        List of JSONL strings, each below the size limit
    """
    # Convert to bytes to properly check size
    jsonl_bytes = jsonl_content.encode('utf-8')
    
    # If size is within limit, return as is
    if len(jsonl_bytes) <= max_size_bytes:
        return [jsonl_content]
    
    # Split the batch
    logger.info(f"Batch size ({len(jsonl_bytes)/1024/1024:.2f} MB) exceeds limit of {max_size_bytes/1024/1024:.2f} MB. Splitting...")
    
    lines = jsonl_content.strip().split('\n')
    batches = []
    current_batch = []
    current_size = 0
    
    for line in lines:
        line_bytes = (line + '\n').encode('utf-8')
        line_size = len(line_bytes)
        
        # If adding this line would exceed the limit, start a new batch
        if current_size + line_size > max_size_bytes:
            if current_batch:  # Don't create empty batches
                batches.append('\n'.join(current_batch))
                current_batch = []
                current_size = 0
            
            # Handle case where a single line is larger than the limit
            if line_size > max_size_bytes:
                logger.warning(f"Found request larger than max size ({line_size} bytes). Skipping.")
                continue
        
        current_batch.append(line)
        current_size += line_size
    
    # Add the last batch if not empty
    if current_batch:
        batches.append('\n'.join(current_batch))
    
    logger.info(f"Split batch into {len(batches)} smaller batches")
    return batches

def upload_batch_file(jsonl_content: str) -> str:
    """
    Upload a batch file to OpenAI
    
    Args:
        jsonl_content: JSONL content as a string
        
    Returns:
        File ID of the uploaded file
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as temp_file:
        temp_file.write(jsonl_content)
        temp_file_path = temp_file.name
    
    try:
        # Upload file
        with open(temp_file_path, 'rb') as file:
            response = client.files.create(
                file=file,
                purpose="batch"
            )
        
        logger.info(f"Uploaded batch file with ID: {response.id}")
        return response.id
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

def create_batch(file_id: str, endpoint: str = "/v1/chat/completions") -> Dict[str, Any]:
    """
    Create a batch job with OpenAI
    
    Args:
        file_id: File ID of the uploaded batch file
        endpoint: API endpoint to use
        
    Returns:
        Batch information dictionary
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.batches.create(
        input_file_id=file_id,
        endpoint=endpoint,
        completion_window="24h"
    )
    
    return response.dict()

def check_batch_status(batch_id: str, wait: bool = True, polling_interval: int = 60) -> Dict[str, Any]:
    """
    Check status of a batch job
    
    Args:
        batch_id: Batch ID to check
        wait: Whether to wait for completion
        polling_interval: Seconds between status checks when waiting
        
    Returns:
        Batch information dictionary
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    while True:
        response = client.batches.retrieve(batch_id)
        batch_info = response.dict()
        
        status = batch_info["status"]
        logger.info(f"Batch status: {status}")
        
        if status == "completed" or not wait or status in ["failed", "expired", "cancelled"]:
            return batch_info
        
        # Display progress
        completed = batch_info["request_counts"]["completed"]
        total = batch_info["request_counts"]["total"]
        if total > 0:
            logger.info(f"Progress: {completed}/{total} requests completed ({completed/total*100:.1f}%)")
        
        logger.info(f"Waiting {polling_interval} seconds for batch completion...")
        time.sleep(polling_interval)

def download_batch_results(output_file_id: str) -> Dict[str, Any]:
    """
    Download batch results
    
    Args:
        output_file_id: Output file ID
        
    Returns:
        Dictionary mapping filenames to extraction results
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Download file content
    file_response = client.files.content(output_file_id)
    content = file_response.text
    
    # Parse JSONL file
    results = {}
    for line in content.strip().split('\n'):
        result = json.loads(line)
        custom_id = result["custom_id"]
        
        if result.get("error"):
            logger.error(f"Error for {custom_id}: {result['error']}")
            continue
        
        # Get the actual response content
        if result.get("response") and result["response"].get("body"):
            body = result["response"]["body"]
            if isinstance(body, str):
                body = json.loads(body)
            
            # Extract the content from the completion
            if "choices" in body and body["choices"]:
                message = body["choices"][0]["message"]
                if message.get("content"):
                    content_str = message["content"]
                    try:
                        content_json = json.loads(content_str)
                        results[custom_id] = content_json
                    except json.JSONDecodeError:
                        logger.error(f"Could not decode JSON response for {custom_id}")
    
    return results

def save_extraction_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save extraction results to JSON files
    
    Args:
        results: Dictionary mapping filenames to extraction results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename, data in results.items():
        # Convert filename.txt to filename.json
        base_filename = os.path.basename(filename)
        json_filename = os.path.splitext(base_filename)[0] + ".json"
        output_path = os.path.join(output_dir, json_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extraction results to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract entities from amicus brief text files using OpenAI Batch API")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", help="GCS prefix for text files (used if --blob-list-file not provided)")
    parser.add_argument("--blob-list-file", help="File containing blob names (one per line) to process")
    parser.add_argument("--output-dir", default="output", help="Directory to save extraction results")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--model", default="gpt-4.1-nano", help="OpenAI model to use")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for batch completion")
    parser.add_argument("--polling-interval", type=int, default=500, help="Seconds between status checks")
    parser.add_argument("--test-batch", type=int, nargs="?", const=100, 
                       help="Run a test batch with limited number of files (default: 100)")
    args = parser.parse_args()
    
    # Make sure we have either a prefix or a blob list file
    if not args.prefix and not args.blob_list_file:
        logger.error("Either --prefix or --blob-list-file must be provided")
        parser.print_help()
        return
    
    # Get the list of specific blobs if provided
    specific_blobs = None
    if args.blob_list_file:
        specific_blobs = read_blob_list_from_file(args.blob_list_file)
    
    # Fetch text files from GCS
    text_files = fetch_txt_files_from_gcs(
        args.bucket,
        args.prefix,
        args.test_batch,
        specific_blobs
    )
    
    if not text_files:
        logger.error(f"No text files found in gs://{args.bucket}/{args.prefix or 'specified blobs'}")
        return
    
    # Estimate tokens and cost (with 50% Batch API discount)
    texts = list(text_files.values())
    total_tokens, estimated_cost = estimate_tokens_and_cost(texts, AMICI_EXTRACTION_PROMPT, args.model)
    
    # Display information about the batch
    mode = "test batch" if args.test_batch else "full batch"
    logger.info(f"Running {mode} with {len(text_files)} files")
    logger.info(f"Estimated total tokens: {total_tokens:,}")
    logger.info(f"Estimated cost with Batch API (50% discount): ${estimated_cost:.2f}")
    
    # Confirm with user
    if not args.yes:
        confirmation = input(f"Proceed with {mode} extraction for ${estimated_cost:.2f}? (y/N): ")
        if confirmation.lower() != 'y':
            logger.info("Extraction cancelled by user")
            return
    
    # Create batch requests
    logger.info("Creating batch requests...")
    batch_jsonl = create_batch_requests(text_files, AMICI_EXTRACTION_PROMPT, args.model)
    
    # Check if we need to split the batch
    max_size_bytes = 200 * 1024 * 1024  # 200 MB
    batch_parts = split_batch_requests(batch_jsonl, max_size_bytes)
    
    # Process each batch part
    all_results = {}
    for i, batch_part in enumerate(batch_parts):
        if len(batch_parts) > 1:
            logger.info(f"Processing batch part {i+1} of {len(batch_parts)}")
        
        # Upload batch file
        logger.info("Uploading batch file to OpenAI...")
        file_id = upload_batch_file(batch_part)
        
        # Create batch
        logger.info("Creating batch job...")
        batch = create_batch(file_id)
        batch_id = batch["id"]
        logger.info(f"Created batch with ID: {batch_id}")
        
        # Check batch status (with optional wait)
        logger.info("Checking batch status...")
        final_batch = check_batch_status(batch_id, not args.no_wait, args.polling_interval)
        
        if final_batch["status"] != "completed":
            logger.warning(f"Batch part {i+1} did not complete. Final status: {final_batch['status']}")
            if args.no_wait:
                logger.info("You can check the batch status later with the batch ID")
            continue  # Continue with the next batch even if this one fails
        
        # Download and process results
        logger.info("Downloading batch results...")
        output_file_id = final_batch["output_file_id"]
        batch_results = download_batch_results(output_file_id)
        
        # Add results from this batch to the overall results
        all_results.update(batch_results)
        
        logger.info(f"Processed {len(batch_results)} files in batch part {i+1}")
    
    # Save results
    logger.info(f"Saving {len(all_results)} results...")
    save_extraction_results(all_results, args.output_dir)
    
    # Show a summary of test batch results if applicable
    if args.test_batch:
        logger.info(f"Test batch completed: {len(all_results)}/{len(text_files)} files processed successfully")
        if len(all_results) < len(text_files):
            logger.warning(f"Failed to process {len(text_files) - len(all_results)} files")
    
    logger.info("Entity extraction batch processing completed successfully")

if __name__ == "__main__":
    main()
