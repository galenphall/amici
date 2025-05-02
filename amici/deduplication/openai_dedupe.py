import os
import sys
import json
import time
import logging
import argparse
import uuid
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any

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

# Define the prompt for entity resolution
ENTITY_RESOLUTION_PROMPT = """
You are a legal assistant helping with entity resolution for Supreme Court amicus briefs. 
Your task is to determine if pairs of named interest groups are actually the same entity.

Guidelines:
1. Consider name variations, acronyms, parent/child organizations, and merged/renamed organizations
2. Be conservative - only answer "yes" if you are confident they're the same entity

For each pair, provide your determination in a JSON array format like this:
[
  {"pair": 1, "are_same_entity": true/false, "confidence": 0.0-1.0},
  {"pair": 2, "are_same_entity": true/false, "confidence": 0.0-1.0},
  ...
]
"""

# Define the JSON schema for structured output
ENTITY_RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "results" : {
            "type":"array",
            "items": {
                "type":"object",
                "properties":{
                    "pair": {
                        "type": "number",
                        "description": "The number of the corresponding entity pair"
                    },
                    "are_same_entity": {
                        "type": "boolean",
                        "description": "Whether the two organizations are the same entity"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score from 0 to 1"
                    }
                },
                "required": ["are_same_entity", "confidence", "pair"],
                "additionalProperties": False
            }
        }        
    },
    "required": ["results"],
    "additionalProperties": False
}

def estimate_tokens_and_cost(pairs: List[Tuple[str, str]], prompt: str, model: str = "gpt-4o-mini") -> Tuple[int, float]:
    """
    Estimate the token count and cost for processing organization pairs with the OpenAI API.
    
    Args:
        pairs: List of (org1, org2) tuples to compare
        prompt: The system prompt to use
        model: The model to use for estimation
        
    Returns:
        Tuple of (total_tokens, estimated_cost_usd)
    """
    # Rough estimate: 1 token ≈ 4 characters
    prompt_tokens = len(prompt) // 4
    
    # Calculate input tokens - batched processing
    batches = (len(pairs) + 9) // 10  # Ceiling division for batch count
    
    # Estimate tokens per batch
    input_tokens = 0
    for start_idx in range(0, len(pairs), 10):
        batch_pairs = pairs[start_idx:start_idx+10]
        
        # Message for this batch
        message = "Compare the following organization pairs and determine if each pair represents the same entity:\n\n"
        for i, (org1, org2) in enumerate(batch_pairs):
            message += f"Pair {i+1}:\nOrg 1: {org1}\nOrg 2: {org2}\n\n"
        message += "For each pair, provide an assessment in a JSON array format. Include only the pair number, whether they are the same entity, and your confidence level."
        
        message_tokens = len(message) // 4
        batch_input_tokens = prompt_tokens + message_tokens
        input_tokens += batch_input_tokens
    
    # Estimate output tokens (rough estimate for JSON response)
    # 50 tokens per pair, but with batch processing overhead
    output_tokens = len(pairs) * 50 + batches * 50
    
    # Get model pricing
    if model in MODEL_PRICING:
        pricing = MODEL_PRICING[model]
    else:
        logger.warning(f"Pricing not found for model {model}. Using default pricing.")
        pricing = MODEL_PRICING["gpt-4o-mini"]
    
    # Calculate cost
    input_cost = (input_tokens / 1000000) * pricing["input"]
    output_cost = (output_tokens / 1000000) * pricing["output"]
    total_cost = input_cost + output_cost
    total_tokens = input_tokens + output_tokens
    
    logger.info(f"Cost breakdown: Input tokens: {input_tokens:,}, Output tokens (est.): {output_tokens:,}")
    logger.info(f"Number of batches: {batches} (10 pairs per batch)")
    logger.info(f"Pricing for {model}: ${pricing['input']}/1M input tokens, ${pricing['output']}/1M output tokens")
    logger.info(f"Total cost: ${total_cost:.4f}")
    
    return total_tokens, total_cost

def create_batch_requests(org_pairs: List[Tuple[str, str]], prompt: str, model: str = "gpt-4o-mini", batch_size: int = 10) -> str:
    """
    Create batch requests in JSONL format for OpenAI Batch API with multiple pairs per request
    
    Args:
        org_pairs: List of (org1, org2) tuples to compare
        prompt: The system prompt to use
        model: OpenAI model to use
        batch_size: Number of pairs to include in a single request
        
    Returns:
        JSONL string with batch requests
    """
    batch_lines = []
    
    # Group pairs into batches
    for batch_idx in range(0, len(org_pairs), batch_size):
        batch_pairs = org_pairs[batch_idx:batch_idx + batch_size]
        request_id = f"batch_{batch_idx // batch_size}"
        
        # Create user message with all pairs in this batch
        user_message = "Compare the following organization pairs and determine if each pair represents the same entity:\n\n"
        for i, (org1, org2) in enumerate(batch_pairs):
            user_message += f"Pair {i+1}:\nOrg 1: {org1}\nOrg 2: {org2}\n\n"
        
        # Add instruction for response format
        user_message += "For each pair, provide an assessment in a JSON array format. Include only the pair number, whether they are the same entity, and your confidence level."
        
        request_data = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_message}
                ],
                "response_format": {"type": "json_schema", "json_schema": {"name": "MatchSchema", "strict": True, "schema": ENTITY_RESOLUTION_SCHEMA}},
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
        Dictionary mapping batch IDs to resolution results
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Download file content
    file_response = client.files.content(output_file_id)
    content = file_response.text

    results = parse_batch_results(content)
    
    return results

def parse_batch_results(content: str) -> dict:

    # Parse JSONL file
    results = {}
    for line in content.strip().split('\n'):
        result = json.loads(line)
        batch_id = result["custom_id"]
        
        if result.get("error"):
            logger.error(f"Error for {batch_id}: {result['error']}")
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
                        results[batch_id] = content_json
                    except json.JSONDecodeError:
                        logger.error(f"Could not decode JSON response for {batch_id}")

    return results

def resume_batch_processing(batch_id: str, output_file: str = None, no_wait: bool = False, polling_interval: int = 60) -> pd.DataFrame:
    """
    Resume processing of an existing batch job
    
    Args:
        batch_id: ID of the batch job to resume
        output_file: Path to save results CSV
        no_wait: Don't wait for batch completion
        polling_interval: Seconds between status checks
        
    Returns:
        DataFrame with entity resolution results
    """
    logger.info(f"Resuming batch processing for batch ID: {batch_id}")
    
    # Check batch status
    final_batch = check_batch_status(batch_id, not no_wait, polling_interval)
    
    if final_batch["status"] != "completed":
        logger.warning(f"Batch did not complete. Final status: {final_batch['status']}")
        if no_wait:
            logger.info("You can check the batch status later with the batch ID")
            return pd.DataFrame()
        return pd.DataFrame()
    
    # Download and process results
    logger.info("Downloading batch results...")
    output_file_id = final_batch["output_file_id"]
    batch_results = download_batch_results(output_file_id)
    
    # Create results dataframe
    results_data = []
    
    # Process the batch results
    for batch_idx, batch_result in batch_results.items():
        # Each batch result is an array of pair assessments
        for result in batch_result:
            if 'pair' not in result:
                logger.warning(f"Missing pair number in result from batch {batch_idx}")
                continue
                
            row_data = {
                'org1': "Unknown",  # We don't have the original data when resuming
                'org2': "Unknown",
                'are_same_entity': result.get('are_same_entity', False),
                'confidence': result.get('confidence', 0.0),
                'pair': result.get('pair'),
                'batch': batch_idx
            }
            
            results_data.append(row_data)
    
    results_df = pd.DataFrame(results_data)
    
    # Save results if output file is specified
    if output_file and not results_df.empty:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
    
    logger.info("Batch processing resumed successfully")
    return results_df

def extract_org_pairs(df: pd.DataFrame,
                    org1_column: str,
                    org2_column: str) -> pd.DataFrame:
    # Extract org pairs from dataframe
    org_pairs = []
    pair_indices = []
    
    for idx, row in df.iterrows():
        org1 = str(row[org1_column])
        org2 = str(row[org2_column])
        
        # Skip if either name is missing
        if pd.isna(org1) or pd.isna(org2):
            continue
            
        org_pairs.append((org1, org2))
        pair_indices.append(idx)

    return org_pairs, pair_indices

def process_dataframe_for_entity_resolution(df: pd.DataFrame, 
                                          org1_column: str, 
                                          org2_column: str, 
                                          model: str = "gpt-4o-mini",
                                          output_file: str = None,
                                          yes: bool = False,
                                          max_pairs: int = None,
                                          no_wait: bool = False,
                                          polling_interval: int = 60,
                                          batch_size: int = 10,
                                          resume_batch: str = None) -> pd.DataFrame:
    """
    Process a dataframe for entity resolution using OpenAI's Batch API
    
    Args:
        df: Input dataframe
        org1_column: Column name for first organization
        org2_column: Column name for second organization
        model: OpenAI model to use
        output_file: Path to save results CSV
        yes: Skip confirmation prompt
        max_pairs: Maximum number of pairs to process (for testing)
        no_wait: Don't wait for batch completion
        polling_interval: Seconds between status checks
        batch_size: Number of org pairs to process in a single API call
        resume_batch: Batch ID to resume processing
        
    Returns:
        DataFrame with entity resolution results
    """
    # If resuming a batch, use dedicated function
    if resume_batch:
        return resume_batch_processing(
            batch_id=resume_batch,
            output_file=output_file,
            no_wait=no_wait,
            polling_interval=polling_interval
        )
    
    org_pairs, pair_indices = extract_org_pairs(df, org1_column, org2_column)

    # Limit number of pairs if max_pairs is specified
    if max_pairs and max_pairs < len(org_pairs):
        logger.info(f"Limiting to {max_pairs} pairs for testing")
        org_pairs = org_pairs[:max_pairs]
        pair_indices = pair_indices[:max_pairs]
    
    if not org_pairs:
        logger.error("No valid organization pairs found in the dataframe")
        return pd.DataFrame()
    
    # Estimate tokens and cost
    logger.info(f"Estimating cost for {len(org_pairs)} organization pairs...")
    total_tokens, estimated_cost = estimate_tokens_and_cost(org_pairs, ENTITY_RESOLUTION_PROMPT, model)
    
    # Display information about the batch
    logger.info(f"Processing {len(org_pairs)} organization pairs in batches of {batch_size}")
    logger.info(f"Estimated total tokens: {total_tokens:,}")
    logger.info(f"Estimated cost: ${estimated_cost:.4f}")
    
    # Confirm with user
    if not yes:
        confirmation = input(f"Proceed with entity resolution for ${estimated_cost:.4f}? (y/N): ")
        if confirmation.lower() != 'y':
            logger.info("Processing cancelled by user")
            return pd.DataFrame()
    
    # Create batch requests
    logger.info("Creating batch requests...")
    batch_jsonl = create_batch_requests(org_pairs, ENTITY_RESOLUTION_PROMPT, model, batch_size)
    
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
        final_batch = check_batch_status(batch_id, not no_wait, polling_interval)
        
        if final_batch["status"] != "completed":
            logger.warning(f"Batch part {i+1} did not complete. Final status: {final_batch['status']}")
            if no_wait:
                logger.info("You can check the batch status later with the batch ID")
            continue
        
        # Download and process results
        logger.info("Downloading batch results...")
        output_file_id = final_batch["output_file_id"]
        batch_results = download_batch_results(output_file_id)
        
        # Add results from this batch to the overall results
        all_results.update(batch_results)
        
        logger.info(f"Processed {len(batch_results)} batches in part {i+1}")
    
    # Create results dataframe
    results_data = []
    
    # Process the batch results
    for batch_idx, batch_result in all_results.items():
        # Each batch result is an array of pair assessments
        for result in batch_result:
            pair_num = result.get('pair')
            if pair_num is None:
                logger.warning(f"Missing pair number in result from batch {batch_idx}")
                continue
                
            # Calculate the actual pair index
            batch_num = int(batch_idx.split('_')[1])
            pair_index = (batch_num * batch_size) + (pair_num - 1)
            
            if pair_index >= len(pair_indices):
                logger.warning(f"Pair index {pair_index} out of range")
                continue
                
            original_idx = pair_indices[pair_index]
            
            row_data = {
                'original_index': original_idx,
                'org1': org_pairs[pair_index][0],
                'org2': org_pairs[pair_index][1],
                'are_same_entity': result.get('are_same_entity', False),
                'confidence': result.get('confidence', 0.0)
            }
            
            results_data.append(row_data)
    
    results_df = pd.DataFrame(results_data)
    
    # Save results if output file is specified
    if output_file and not results_df.empty:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
    
    logger.info("Entity resolution completed successfully")
    return results_df

def link_batch_results_local(df: pd.DataFrame, 
                            org1_column: str, 
                            org2_column: str, 
                            resultsfile: str) -> pd.DataFrame:
    """
    Link batch results in a local file to the dataframe used to submit the batch request.

    The dataframe must *exactly* match the dataframe used in creating the corresponding batch 
    request!!

    Args:
        df: Input DataFrame
        org1_column: column name containing first organizations
        org2_column: column name containing second organizations
        resultsfile: path to the batch results stored locally
    
    Returns:
        DataFrame with pairs and corresponding match probabilities.
    """
    
    if not os.path.exists(resultsfile):
        raise ValueError(f"{resultsfile} is not a valid file path.")

    with open(resultsfile, 'r') as rf:
        results = parse_batch_results(rf.read())

    # Infer batch size based on number of matches in the first result
    batch_size = len(list(list(results.values())[0].values())[0])

    org_pairs, pair_indices = extract_org_pairs(df, org1_column, org2_column)

    linked_pairs = []
    for batch_idx in range(0, len(org_pairs), batch_size):
        batch_pairs = org_pairs[batch_idx:batch_idx + batch_size]

        try:
            batch_output = results[f"batch_{batch_idx // batch_size}"]
        except KeyError as e:
            logger.warning(f"Failed on batch {batch_idx}")
            continue

        k = list(batch_output.keys())[0]
        batch_output = batch_output[k]
        for i, (l, r) in enumerate(batch_pairs):
            output = batch_output[i]
            assert output['pair'] == i+1
            output.update({'left_norm': l, 'right_norm': r})
            linked_pairs.append(output)

    df = pd.DataFrame(linked_pairs)

    return df

def main():
    parser = argparse.ArgumentParser(description="Entity resolution for interest groups using OpenAI Batch API")
    parser.add_argument("--input", help="Input CSV file")
    parser.add_argument("--org1-column", help="Column name for first organization")
    parser.add_argument("--org2-column", help="Column name for second organization")
    parser.add_argument("--output", help="Output CSV file path")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for batch completion")
    parser.add_argument("--polling-interval", type=int, default=60, help="Seconds between status checks")
    parser.add_argument("--max-pairs", type=int, help="Maximum number of pairs to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of pairs to process in a single API call")
    parser.add_argument("--resume-batch", help="Resume processing for an existing batch ID")
    args = parser.parse_args()
    
    # Validate arguments
    if not args.resume_batch and (not args.input or not args.org1_column or not args.org2_column):
        parser.error("--input, --org1-column, and --org2-column are required unless --resume-batch is specified")
    
    # Read input CSV if not resuming
    df = pd.DataFrame()
    if args.input:
        df = pd.read_csv(args.input)
        logger.info(f"Read {len(df)} rows from {args.input}")
    
    # Process for entity resolution
    results_df = process_dataframe_for_entity_resolution(
        df=df,
        org1_column=args.org1_column if not args.resume_batch else None,
        org2_column=args.org2_column if not args.resume_batch else None,
        model=args.model,
        output_file=args.output,
        yes=args.yes,
        max_pairs=args.max_pairs,
        no_wait=args.no_wait,
        polling_interval=args.polling_interval,
        batch_size=args.batch_size,
        resume_batch=args.resume_batch
    )
    
    if results_df.empty:
        logger.warning("No results generated")
    else:
        # Print summary statistics
        match_count = results_df['are_same_entity'].sum()
        total_count = len(results_df)
        match_percent = (match_count / total_count * 100) if total_count > 0 else 0
        
        logger.info(f"Results summary:")
        logger.info(f"  Total pairs processed: {total_count}")
        logger.info(f"  Same entity matches: {match_count} ({match_percent:.1f}%)")
        logger.info(f"  Different entities: {total_count - match_count} ({100 - match_percent:.1f}%)")
        
        # Print high confidence matches
        high_conf_matches = results_df[(results_df['are_same_entity'] == True) & (results_df['confidence'] >= 0.9)]
        if not high_conf_matches.empty:
            logger.info(f"  High confidence matches: {len(high_conf_matches)}")
            for _, row in high_conf_matches.head(5).iterrows():
                logger.info(f"    {row['org1']} = {row['org2']}")
            if len(high_conf_matches) > 5:
                logger.info(f"    ... and {len(high_conf_matches) - 5} more")

def main_local():
    parser = argparse.ArgumentParser(description="Entity resolution for interest groups using OpenAI Batch API")
    parser.add_argument("--df", help="Input CSV file")
    parser.add_argument("--b", help="Path to batch output")
    parser.add_argument("--col1", default="left_norm", help="Column name for first organization")
    parser.add_argument("--col2", default="right_norm", help="Column name for second organization")
    parser.add_argument("--output", help="Output CSV file path")
    args = parser.parse_args()

    # Read input CSV if not resuming
    df = pd.DataFrame()
    if args.df:
        df = pd.read_csv(args.df)
        logger.info(f"Read {len(df)} rows from {args.df}")
    
    linked_df = link_batch_results_local(df, args.col1, args.col2, args.b)

    linked_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()