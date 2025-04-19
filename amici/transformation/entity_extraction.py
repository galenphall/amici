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

def fetch_txt_files_from_gcs(bucket_name: str, prefix: str, test_batch_size: int = None) -> Dict[str, str]:
    """
    Fetch text files from the specified GCS bucket and prefix.
    If test_batch_size is specified, only fetches that many files.
    
    Args:
        bucket_name: Name of the GCS bucket
        prefix: Prefix of files to fetch
        test_batch_size: If set, limits the number of files fetched to this value
        
    Returns:
        Dictionary mapping filenames to text content
    """
    # First, list all matching files
    txt_file_names = list_txt_files_from_gcs(bucket_name, prefix)
    
    if not txt_file_names:
        return {}
    
    # If test batch size is specified, randomly sample files
    if test_batch_size and test_batch_size < len(txt_file_names):
        logger.info(f"Sampling {test_batch_size} files for test batch")
        txt_file_names = random.sample(txt_file_names, test_batch_size)
    
    # Now fetch only the required files
    gcs = GCSFetch(bucket_name)
    text_files = {}
    
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
    
    total_tokens = 0
    for text in texts:
        # Count tokens for this text + prompt
        text_tokens = len(encoding.encode(text))
        # Add tokens for this request (prompt + text + estimated completion)
        total_tokens += prompt_tokens + text_tokens
        # Rough completion estimate (can adjust based on observed completion sizes)
        total_tokens += 500  # Rough estimate for JSON response
    
    # Batch API has 50% discount compared to standard API
    cost_per_1M_tokens = 0.15 * 0.5  # 50% discount for batch API
    estimated_cost = (total_tokens / 1000000) * cost_per_1M_tokens
    
    return total_tokens, estimated_cost

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
    parser.add_argument("--prefix", required=True, help="GCS prefix for text files")
    parser.add_argument("--output-dir", default="output", help="Directory to save extraction results")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--model", default="gpt-4.1-nano", help="OpenAI model to use")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for batch completion")
    parser.add_argument("--polling-interval", type=int, default=500, help="Seconds between status checks")
    parser.add_argument("--test-batch", type=int, nargs="?", const=100, 
                       help="Run a test batch with limited number of files (default: 100)")
    args = parser.parse_args()
    
    # Fetch text files from GCS (with test batch limit directly applied)
    text_files = fetch_txt_files_from_gcs(
        args.bucket, 
        args.prefix,
        args.test_batch  # Pass test batch size directly
    )
    
    if not text_files:
        logger.error(f"No text files found in gs://{args.bucket}/{args.prefix}")
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
