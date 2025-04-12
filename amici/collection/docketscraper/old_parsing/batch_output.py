import os
import fitz  # PyMuPDF
import time
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import logging
import dotenv
import threading
from tqdm import tqdm
import pandas as pd
import io
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables from .env file
env_file = os.path.join(os.path.dirname(__file__), "../../../env/.env")
dotenv.load_dotenv(env_file)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set the logging output directory
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"batch-extraction-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def check_batch_status(batch_id: str):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    logger.info(f"Batch {batch_id} status: {batch.status}")
    return batch

def retrieve_batch_results(output_file_id: str, save_path: str):
    """Retrieve and save batch results."""
    file_response = client.files.content(output_file_id)
    with open(save_path, 'wb') as f:
        f.write(file_response.content)
    logger.info(f"Retrieved batch results and saved to {save_path}")
    return save_path

def process_batch_results(results_file: str, doc_map: Dict[str, Any]):
    """Process the results from the batch API and map to original documents."""
    from supremecourt.datastructures import Amicus, Lawyer, Brief, ExtractionResult, ProcessedResult

    results = []
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            doc_info = doc_map.get(custom_id, None)
            if not doc_info:
                logger.error(f"Custom ID {custom_id} not found in document map.")
                continue
            
            # Extract the actual filename string, no matter how deep in dictionary structure
            if isinstance(doc_info, dict) and 'source_file' in doc_info:
                source_file = doc_info['source_file']
            else:
                source_file = str(doc_info)
            
            if result.get('error'):
                # Handle error case
                processed_result = ProcessedResult(
                    source_file=source_file,
                    extraction_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                    confidence="error",
                    error=result['error'].get('message', str(result['error']))
                )
            else:
                # Parse successful response
                try:
                    # For structured outputs, response is already in the correct format
                    if 'response' in result and 'body' in result['response']:
                        # Direct JSON parsing from the structured output
                        extraction_data = result['response']['body']['choices'][0]['message']['content']
                        extraction_data = json.loads(extraction_data)
                        
                        # Convert to our Pydantic model
                        processed_result = ProcessedResult(
                            source_file=source_file,
                            extraction_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                            amici=[Amicus.model_validate(amicus) for amicus in extraction_data.get('amici', [])],
                            lawyers=[Lawyer.model_validate(lawyer) for lawyer in extraction_data.get('lawyers', [])],
                            brief=Brief.model_validate(extraction_data.get('brief', {})),
                            confidence=extraction_data.get('confidence', 'medium')
                        )
                    else:
                        # Handle unexpected response format
                        processed_result = ProcessedResult(
                            source_file=source_file,
                            extraction_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                            confidence="error",
                            error="Unexpected response format from batch API"
                        )
                except Exception as e:
                    logger.error(f"Error processing result for {custom_id}: {e}")
                    processed_result = ProcessedResult(
                        source_file=source_file,
                        extraction_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                        confidence="error",
                        error=str(e)
                    )
            
            results.append((custom_id, processed_result))
    
    return results

def retrieve_and_process_results(batch_id: str, output_dir: str, documents: List[Dict[str, Any]]):
    """Retrieve and process batch results once the job is complete."""
    # Check batch status
    batch = check_batch_status(batch_id)
    
    if batch.status != "completed":
        print(f"Batch is not completed yet. Current status: {batch.status}")
        return None
    
    # Check if results already exist locally
    batch_output_file = os.path.join(output_dir, f"batch_output_{batch_id}.jsonl")
    if os.path.exists(batch_output_file):
        print(f"Batch results already downloaded at {batch_output_file}")
    else:
        # Download results
        print(f"Downloading batch results to {batch_output_file}")
        retrieve_batch_results(batch.output_file_id, batch_output_file)
    
    # Process results
    results = process_batch_results(batch_output_file, documents)
    
    # Save individual results
    results_dir = os.path.join(output_dir, f"batch_results_{batch_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a mapping file to track UUID to source_file relationships
    uuid_map = {}
    
    for custom_id, result in results:
        # Generate a UUID for the filename
        file_uuid = str(uuid.uuid4())
        uuid_map[file_uuid] = result.source_file
        
        # Use UUID for filename instead of source_file
        output_file = os.path.join(results_dir, f"{file_uuid}.json")
        with open(output_file, 'w') as f:
            f.write(result.model_dump_json(indent=2))
    
    # Save the UUID to source_file mapping
    with open(os.path.join(results_dir, "uuid_to_source_map.json"), 'w') as f:
        json.dump(uuid_map, f, indent=2)
    
    logger.info(f"Processed {len(results)} results. Saved to {results_dir}")
    print(f"Processed {len(results)} results. Saved to {results_dir}")
    print(f"UUID to source file mapping saved to {os.path.join(results_dir, 'uuid_to_source_map.json')}")
    
    return results

# Example usage (modified version of the original script's main execution)
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process batch results from one or more batch IDs')
    parser.add_argument('--batch_ids', type=str, required=True, 
                      help='The batch ID(s) to process, comma-separated for multiple IDs')
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(__file__), "../data"), 
                      help='Directory containing batch results CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save processed results')
    parser.add_argument('--sample_size', type=int, default=100, 
                      help='Number of random samples to select')
    parser.add_argument('--custom_id_map', type=str, default=None,
                      help='Path to the custom ID map JSON file')
    args = parser.parse_args()

    # Process comma-separated batch IDs
    batch_ids = [bid.strip() for bid in args.batch_ids.split(',')]
    
    # Set output directory
    if not args.output_dir:
        args.output_dir = os.path.join(args.data_dir, "batch_results")
    os.makedirs(args.output_dir, exist_ok=True)

    # Find the custom ID map
    custom_id_map = {}
    if args.custom_id_map and os.path.exists(args.custom_id_map):
        with open(args.custom_id_map, "r") as f:
            custom_id_map = json.load(f)
        logger.info(f"Loaded custom ID map from {args.custom_id_map}")
    else:
        # Try to find the latest custom ID map in the data directory
        map_files = [f for f in os.listdir(args.data_dir) if f.startswith("custom_id_map_") and f.endswith(".json")]
        if map_files:
            latest_map = sorted(map_files, key=lambda x: os.path.getmtime(os.path.join(args.data_dir, x)), reverse=True)[0]
            with open(os.path.join(args.data_dir, latest_map), "r") as f:
                custom_id_map = json.load(f)
            logger.info(f"Loaded latest custom ID map from {latest_map}")
        else:
            logger.error("No custom ID map found. Please provide one with --custom_id_map")
            exit(1)

    # Process each batch ID
    all_results = []
    for batch_id in batch_ids:
        logger.info(f"Processing batch ID: {batch_id}")
        results = retrieve_and_process_results(batch_id, args.output_dir, custom_id_map)
        
        if results:
            all_results.extend(results)
            logger.info(f"Added {len(results)} results from batch {batch_id}")
    
    if not all_results:
        logger.warning("No results processed. Check batch status or custom ID map.")
        exit(0)
        
    # Combine results into a single DataFrame
    combined_results_file = os.path.join(args.output_dir, f"combined_results_{'_'.join(batch_ids[:3])}.csv")
    results_df = pd.DataFrame([result[1].model_dump() for result in all_results])
    
    # Try to add metadata if amicus_briefs.csv exists
    amicus_briefs_path = os.path.join(args.data_dir, "amicus_briefs.csv")
    if os.path.exists(amicus_briefs_path):
        try:
            amicus_briefs = pd.read_csv(amicus_briefs_path)
            blobs = list(amicus_briefs["blob_name"].values)
            
            # Blob names contain the source file names, so we need to map them
            source_file_to_blob_name = {os.path.basename(blob): blob for blob in blobs}
            results_df["blob_name"] = results_df["source_file"].apply(
                lambda x: source_file_to_blob_name.get(x, x)
            )
            results_df = results_df.merge(amicus_briefs, on="blob_name", how="left")
            logger.info("Added metadata from amicus_briefs.csv")
        except Exception as e:
            logger.error(f"Error adding metadata from amicus_briefs.csv: {e}")
    
    # Save combined results
    results_df.to_csv(combined_results_file, index=False)
    logger.info(f"Combined results saved to {combined_results_file}")
    
    # Create random subset
    if len(results_df) > args.sample_size:
        sample_df = results_df.sample(args.sample_size)
        sample_file = os.path.join(args.output_dir, f"sample_results_{'_'.join(batch_ids[:3])}.csv")
        sample_df.to_csv(sample_file, index=False)
        print(f"Created random subset with {len(sample_df)} samples at {sample_file}")
