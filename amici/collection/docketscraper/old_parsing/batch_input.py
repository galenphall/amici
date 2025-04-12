import os
import fitz  # PyMuPDF
import time
import json
from typing import List, Dict, Any, Optional, Literal, Tuple
from pathlib import Path
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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

# Import JSON schema from module (or define it inline if needed)
from supremecourt.datastructures import json_schema

def load_text_from_file(text_file_path: str) -> str:
    """
    Load extracted text from a file.
    
    Args:
        text_file_path: Path to the text file
        
    Returns:
        str: Extracted text content
    """
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if utf-8 fails
        with open(text_file_path, 'r', encoding='latin-1') as f:
            return f.read()


def prepare_appendix_detection_batch(documents: List[Dict[str, Any]], output_file: str) -> str:
    """
    Prepare a batch file to detect if documents have appendices with amici lists.
    
    Args:
        documents: List of document dictionaries
        output_file: Path to save the JSONL batch file
        
    Returns:
        str: Path to the created batch file
    """
    system_prompt = """
    You are an expert legal assistant analyzing amicus briefs. Your task is to determine if this brief 
    contains an appendix or a separate section that lists additional amici curiae not mentioned in the 
    primary text. This is common in briefs with many amici where only a representative subset is 
    mentioned in the introduction.
    """
    
    # Define a simple JSON schema for yes/no response
    appendix_schema = {
        "name": "appendix_detection",
        "schema": {
            "type": "object",
            "properties": {
                "has_appendix": {"type": "boolean"},
                "explanation": {"type": "string"}
            },
            "required": ["has_appendix", "explanation"],
            "additionalProperties": False
        },
        "strict": True
    }
    
    with open(output_file, 'w') as f:
        for doc in documents:
            # Use only first 10k characters for quick detection
            text_preview = doc['text'][:10000]
            custom_id = doc['custom_id']
            
            batch_item = {
                "custom_id": f"appendix-check-{custom_id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",  # Faster, cheaper model for simple detection
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this amicus brief excerpt and determine if there's likely an appendix with additional amici not mentioned in the main text. Look for clues like 'See Appendix for list of amici' or 'Additional amici listed in Appendix A'. Respond with structured JSON:\n\n{text_preview}"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": appendix_schema
                    },
                    "temperature": 0.1
                }
            }
            
            f.write(json.dumps(batch_item) + '\n')
    
    logger.info(f"Created appendix detection batch file with {len(documents)} requests at {output_file}")
    return output_file


def prepare_extraction_batch(documents: List[Dict[str, Any]], 
                           has_appendix: Dict[str, bool], 
                           output_file: str) -> str:
    """
    Prepare a batch file for extracting amici information, using full text for docs with appendices.
    
    Args:
        documents: List of document dictionaries
        has_appendix: Dictionary mapping document IDs to boolean (True if has appendix)
        output_file: Path to save the JSONL batch file
        
    Returns:
        str: Path to the created batch file
    """
    system_prompt = """
    You are an expert legal assistant that extracts information about amici curiae (friends of the court) 
    and their lawyers from amicus briefs. Extract the following:
    
    1. A list of all amici organizations or individuals
    2. For each lawyer: full name, law firm/organization, position (if available), and which amici they represent
    3. The position of the amici (supporting petitioner, supporting respondent, or neutral/other)
    4. The dockets for which the amici are submitting the brief
    
    Be thorough in identifying ALL amici mentioned anywhere in the document, including appendices. 
    Focus on the cover page, "INTEREST OF AMICI CURIAE" section, and any appendices that list additional amici.
    If you're unsure about certain information, set the confidence field accordingly.
    """
    
    with open(output_file, 'w') as f:
        for doc in documents:
            custom_id = doc['custom_id']
            
            # Check if document has appendix
            doc_has_appendix = has_appendix.get(custom_id, False)
            
            # For documents with appendices, use the full text
            # Otherwise, limit to 30k chars to save tokens
            if doc_has_appendix:
                text = doc['text']  # Full text
                logger.debug(f"Using full text for {custom_id} (has appendix)")
            else:
                text = doc['text'][:30000]  # Limited text
                logger.debug(f"Using limited text for {custom_id} (no appendix)")
            
            model = doc.get('model', 'gpt-4o-mini')
            
            batch_item = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract amici and lawyer information from this amicus brief text and respond with structured JSON data:\n\n{text}"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": json_schema
                    },
                    "temperature": 0.1
                }
            }
            
            f.write(json.dumps(batch_item) + '\n')
    
    logger.info(f"Created extraction batch file with {len(documents)} requests at {output_file}")
    return output_file


def check_batch_size(file_path: str, max_size_bytes: int = 200 * 1024 * 1024) -> bool:
    """Check if a batch file is under the size limit."""
    file_size = os.path.getsize(file_path)
    logger.info(f"Batch file size: {file_size / (1024 * 1024):.2f} MB")
    return file_size <= max_size_bytes


def split_batch_file(file_path: str, max_size_bytes: int = 200 * 1024 * 1024, max_docs: int = 10000) -> List[str]:
    """
    Split a large batch file into multiple smaller files.
    
    Args:
        file_path: Path to the original batch file
        max_size_bytes: Maximum size in bytes for each batch file
        max_docs: Maximum number of documents per batch as a fallback
        
    Returns:
        List of paths to the smaller batch files
    """
    with open(file_path, 'r') as f:
        lines = [line for line in f]
    
    batch_files = []
    current_batch = []
    current_size = 0
    batch_count = 1
    
    base_name, ext = os.path.splitext(file_path)
    
    for line in lines:
        line_size = len(line.encode('utf-8'))
        
        # If adding this line would exceed max size or we've hit max docs, start new batch
        if (current_size + line_size > max_size_bytes) or (len(current_batch) >= max_docs):
            # Write current batch to file
            batch_file = f"{base_name}_part{batch_count}{ext}"
            with open(batch_file, 'w') as f:
                f.writelines(current_batch)
            
            batch_files.append(batch_file)
            current_batch = []
            current_size = 0
            batch_count += 1
        
        # Add line to current batch
        current_batch.append(line)
        current_size += line_size
    
    # Write final batch if not empty
    if current_batch:
        batch_file = f"{base_name}_part{batch_count}{ext}"
        with open(batch_file, 'w') as f:
            f.writelines(current_batch)
        batch_files.append(batch_file)
    
    logger.info(f"Split batch file into {len(batch_files)} parts")
    return batch_files


def prepare_extraction_batch_with_splitting(
    documents: List[Dict[str, Any]], 
    has_appendix: Dict[str, bool], 
    output_file: str
) -> List[str]:
    """Prepare extraction batch file and split if needed."""
    # First create the complete batch file
    prepare_extraction_batch(documents, has_appendix, output_file)
    
    # Check if we need to split it
    if check_batch_size(output_file):
        logger.info(f"Batch file size is acceptable")
        return [output_file]
    else:
        logger.warning(f"Batch file exceeds OpenAI size limit, splitting...")
        return split_batch_file(output_file)


def upload_batch_file(file_path: str) -> str:
    """Upload batch input file to OpenAI."""
    with open(file_path, 'rb') as f:
        file = client.files.create(
            file=f,
            purpose="batch"
        )
    logger.info(f"Uploaded file {file_path} with ID: {file.id}")
    return file.id


def create_batch_job(file_id: str, description: str = "Supreme Court amicus analysis") -> str:
    """Create a batch processing job."""
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description}
    )
    logger.info(f"Created batch job with ID: {batch.id}")
    return batch.id


def check_batch_status(batch_id: str) -> Dict:
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    logger.info(f"Batch {batch_id} status: {batch.status}")
    return batch


def retrieve_batch_results(output_file_id: str, save_path: str) -> str:
    """Retrieve and save batch results."""
    file_response = client.files.content(output_file_id)
    with open(save_path, 'wb') as f:
        f.write(file_response.content)
    logger.info(f"Retrieved batch results and saved to {save_path}")
    return save_path


def process_appendix_detection_results(results_file: str) -> Dict[str, bool]:
    """
    Process the results of appendix detection batch job.
    
    Args:
        results_file: Path to the batch results file
        
    Returns:
        Dict mapping document IDs to boolean (True if has appendix)
    """
    has_appendix = {}
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            
            # Extract the original document ID from the custom_id
            custom_id = result['custom_id'].replace('appendix-check-', '')
            
            # Process successful responses
            if 'response' in result and 'body' in result['response']:
                try:
                    body = result['response']['body']
                    # If body is string, parse it as JSON
                    if isinstance(body, str):
                        body = json.loads(body)
                    
                    has_appendix[custom_id] = body.get('has_appendix', False)
                    logger.debug(f"Document {custom_id}: has_appendix={has_appendix[custom_id]}, explanation={body.get('explanation', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error processing appendix detection result for {custom_id}: {e}")
                    has_appendix[custom_id] = False  # Default to False on error
            else:
                logger.warning(f"No valid response for {custom_id}, defaulting to no appendix")
                has_appendix[custom_id] = False
    
    logger.info(f"Processed appendix detection results: {sum(has_appendix.values())} of {len(has_appendix)} documents have appendices")
    return has_appendix


def prepare_documents_from_text_files(text_dir: str, blob_mapping: Dict[str, str], models: List[str] = None) -> List[Dict[str, Any]]:
    """
    Prepare documents from extracted text files.
    
    Args:
        text_dir: Directory containing extracted text files
        blob_mapping: Mapping from document IDs to blob names
        models: List of models to use (optional)
        
    Returns:
        List of document dictionaries for batch processing
    """
    documents = []
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    
    if not models:
        models = ["gpt-4o-mini"] * len(text_files)
    
    logger.info(f"Found {len(text_files)} text files in {text_dir}")
    
    for i, text_file in enumerate(tqdm(text_files, desc="Preparing documents from text files")):
        try:
            # Generate a custom ID for this document
            custom_id = f"doc-{uuid.uuid4()}"
            
            # Get the document ID from the filename (remove .txt extension)
            doc_id = text_file.replace('.txt', '')
            
            # Load the text content
            text_path = os.path.join(text_dir, text_file)
            text_content = load_text_from_file(text_path)
            
            # Get the source file (blob name) from the mapping if available
            source_file = blob_mapping.get(doc_id, None)
            
            # If source file is not found, raise an error
            if not source_file:
                raise ValueError(f"Source file not found for document ID {doc_id}")
            
            # Create document entry
            document = {
                'custom_id': custom_id,
                'source_file': source_file,
                'text': text_content,
                'doc_id': doc_id,
                'model': models[i] if i < len(models) else "gpt-4o-mini"
            }
            
            documents.append(document)
            
        except Exception as e:
            logger.error(f"Error preparing document from {text_file}: {e}")
    
    logger.info(f"Prepared {len(documents)} documents from text files")
    return documents


def two_stage_batch_process(documents: List[Dict[str, Any]], 
                          output_dir: str, 
                          wait_for_completion: bool = False,
                          timeout_seconds: int = 3600) -> Dict[str, Any]:
    """
    Run a two-stage batch processing pipeline: first detect appendices, then extract information.
    
    Args:
        documents: List of document dictionaries
        output_dir: Directory to save output files
        wait_for_completion: Whether to wait for batch jobs to complete
        timeout_seconds: Maximum time to wait for completion
        
    Returns:
        Dictionary with batch job IDs and metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = int(time.time())
    
    # Stage 1: Appendix Detection
    logger.info("Stage 1: Appendix Detection")
    appendix_batch_file = os.path.join(output_dir, f"appendix_detection_{timestamp}.jsonl")
    prepare_appendix_detection_batch(documents, appendix_batch_file)
    
    appendix_file_id = upload_batch_file(appendix_batch_file)
    appendix_batch_id = create_batch_job(appendix_file_id, "Appendix detection")
    
    # Save mapping from custom_id to source_file and doc_id
    custom_id_map = {doc['custom_id']: {'source_file': doc['source_file'], 'doc_id': doc.get('doc_id', '')} 
                    for doc in documents}
    
    with open(os.path.join(output_dir, f"custom_id_map_{timestamp}.json"), "w") as f:
        json.dump(custom_id_map, f)
    
    if wait_for_completion:
        logger.info(f"Waiting for appendix detection batch to complete (timeout: {timeout_seconds}s)")
        
        # Check status periodically until completion or timeout
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            batch = check_batch_status(appendix_batch_id)
            
            if batch.status == "completed":
                logger.info("Appendix detection batch completed")
                
                # Retrieve and process results
                appendix_results_file = os.path.join(output_dir, f"appendix_results_{timestamp}.jsonl")
                retrieve_batch_results(batch.output_file_id, appendix_results_file)
                
                # Process the results
                has_appendix = process_appendix_detection_results(appendix_results_file)
                
                # Stage 2: Extract Amici Information
                logger.info("Stage 2: Amici Information Extraction")
                extraction_batch_file = os.path.join(output_dir, f"extraction_{timestamp}.jsonl")
                prepare_extraction_batch_with_splitting(documents, has_appendix, extraction_batch_file)
                
                extraction_file_id = upload_batch_file(extraction_batch_file)
                extraction_batch_id = create_batch_job(extraction_file_id, "Amici extraction")
                
                # Return both batch IDs
                return {
                    "appendix_detection": {
                        "batch_id": appendix_batch_id,
                        "results_file": appendix_results_file,
                        "status": "completed"
                    },
                    "information_extraction": {
                        "batch_id": extraction_batch_id,
                        "status": "in_progress"
                    },
                    "custom_id_map": os.path.join(output_dir, f"custom_id_map_{timestamp}.json"),
                    "timestamp": timestamp
                }
            
            elif batch.status in ["failed", "cancelled", "expired"]:
                logger.error(f"Appendix detection batch {batch.status}")
                return {
                    "appendix_detection": {
                        "batch_id": appendix_batch_id,
                        "status": batch.status,
                        "error": f"Batch {batch.status}"
                    },
                    "timestamp": timestamp
                }
            
            # Wait before checking again
            time.sleep(30)
        
        # If we get here, we timed out
        logger.warning(f"Timeout waiting for appendix detection batch to complete")
        return {
            "appendix_detection": {
                "batch_id": appendix_batch_id,
                "status": "timeout",
                "error": "Timeout waiting for batch to complete"
            },
            "timestamp": timestamp
        }
    
    # Return immediately if not waiting
    return {
        "appendix_detection": {
            "batch_id": appendix_batch_id,
            "status": "in_progress"
        },
        "timestamp": timestamp,
        "custom_id_map": os.path.join(output_dir, f"custom_id_map_{timestamp}.json")
    }


def process_without_detection(documents: List[Dict[str, Any]], 
                          output_dir: str,
                          use_full_text: bool = False) -> Dict[str, Any]:
    """Process documents without the appendix detection stage."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    
    # Create a mock has_appendix dictionary
    has_appendix = {doc['custom_id']: use_full_text for doc in documents}
    
    # Create the extraction batch with splitting
    extraction_batch_file = os.path.join(output_dir, f"extraction_{timestamp}.jsonl")
    batch_files = prepare_extraction_batch_with_splitting(documents, has_appendix, extraction_batch_file)
    
    # Upload and create batch job for each file
    extraction_batch_ids = []
    for i, batch_file in enumerate(batch_files):
        file_id = upload_batch_file(batch_file)
        batch_id = create_batch_job(file_id, f"Amici extraction batch {i+1}/{len(batch_files)}")
        extraction_batch_ids.append(batch_id)
    
    # Save custom ID mapping
    custom_id_map = {doc['custom_id']: {'source_file': doc['source_file'], 'doc_id': doc.get('doc_id', '')} 
                    for doc in documents}
    map_path = os.path.join(output_dir, f"custom_id_map_{timestamp}.json")
    with open(map_path, "w") as f:
        json.dump(custom_id_map, f)
    
    return {
        "information_extraction": {
            "batch_ids": extraction_batch_ids,
            "status": "in_progress"
        },
        "timestamp": timestamp,
        "custom_id_map": map_path
    }


def load_embedded_text_mapping(data_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load the mapping from document IDs to blob names.
    
    Args:
        data_dir: Directory containing the embedded_text_map.csv file
        
    Returns:
        Tuple of (doc_id_to_blob, blob_to_doc_id) dictionaries
    """
    embedded_map_path = os.path.join(data_dir, "embedded_text_map.csv")
    
    if not os.path.exists(embedded_map_path):
        logger.warning(f"Embedded text mapping file not found: {embedded_map_path}")
        return {}, {}
    
    try:
        embedded_map = pd.read_csv(embedded_map_path, index_col=0)
        
        # Extract the blob name to doc_id mapping
        blob_to_doc_id = {}
        doc_id_to_blob = {}
        
        for blob, row in embedded_map.iterrows():
            doc_id = str(row.get('doc_id', ''))
            if doc_id:
                blob_to_doc_id[blob] = doc_id
                doc_id_to_blob[doc_id] = blob
        
        logger.info(f"Loaded mapping for {len(blob_to_doc_id)} documents")
        return doc_id_to_blob, blob_to_doc_id
    
    except Exception as e:
        logger.error(f"Error loading embedded text mapping: {e}")
        return {}, {}


def continue_from_appendix_batch(appendix_batch_id: str, 
                               output_dir: str,
                               documents: List[Dict[str, Any]],
                               custom_id_map_path: str = None) -> Dict[str, Any]:
    """
    Continue processing from an existing appendix detection batch.
    
    Args:
        appendix_batch_id: Existing appendix detection batch ID
        output_dir: Directory to save output files
        documents: List of document dictionaries
        custom_id_map_path: Path to a custom ID mapping file (optional)
        
    Returns:
        Dictionary with batch job IDs and metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = int(time.time())
    
    # Check batch status
    batch = check_batch_status(appendix_batch_id)
    
    if batch.status != "completed":
        logger.error(f"Appendix detection batch {appendix_batch_id} is not completed (status: {batch.status})")
        return {
            "appendix_detection": {
                "batch_id": appendix_batch_id,
                "status": batch.status,
                "error": f"Batch not completed, current status: {batch.status}"
            },
            "timestamp": timestamp
        }
    
    # Retrieve and process results
    logger.info("Appendix detection batch completed, retrieving results")
    appendix_results_file = os.path.join(output_dir, f"appendix_results_{timestamp}.jsonl")
    retrieve_batch_results(batch.output_file_id, appendix_results_file)
    
    # Process the results
    has_appendix = process_appendix_detection_results(appendix_results_file)
    
    # Load custom ID map if provided, otherwise create a new one
    if custom_id_map_path and os.path.exists(custom_id_map_path):
        with open(custom_id_map_path, 'r') as f:
            custom_id_map = json.load(f)
        logger.info(f"Loaded existing custom ID map from {custom_id_map_path}")
    else:
        # Create new mapping
        custom_id_map = {doc['custom_id']: {'source_file': doc['source_file'], 'doc_id': doc.get('doc_id', '')} 
                        for doc in documents}
        custom_id_map_path = os.path.join(output_dir, f"custom_id_map_{timestamp}.json")
        with open(custom_id_map_path, "w") as f:
            json.dump(custom_id_map, f)
        logger.info(f"Created new custom ID map at {custom_id_map_path}")
    
    # Stage 2: Extract Amici Information
    logger.info("Stage 2: Amici Information Extraction")
    extraction_batch_file = os.path.join(output_dir, f"extraction_{timestamp}.jsonl")
    
    # Get list of batch files (may be split if too large)
    batch_files = prepare_extraction_batch_with_splitting(documents, has_appendix, extraction_batch_file)
    
    # Upload and create batch job for each file
    extraction_batch_ids = []
    for i, batch_file in enumerate(batch_files):
        file_id = upload_batch_file(batch_file)
        batch_id = create_batch_job(file_id, f"Amici extraction batch {i+1}/{len(batch_files)}")
        extraction_batch_ids.append(batch_id)
    
    # Return both batch IDs
    return {
        "appendix_detection": {
            "batch_id": appendix_batch_id,
            "results_file": appendix_results_file,
            "status": "completed"
        },
        "information_extraction": {
            "batch_ids": extraction_batch_ids,
            "status": "in_progress"
        },
        "custom_id_map": custom_id_map_path,
        "timestamp": timestamp
    }


def main():
    """Main function to orchestrate the batch processing pipeline."""
    parser = argparse.ArgumentParser(description='Process amicus briefs using a two-stage batch pipeline.')
    parser.add_argument('--data-dir', type=str, default=None, 
                       help='Directory containing the data files')
    parser.add_argument('--text-dir', type=str, default=None,
                       help='Directory containing extracted text files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files')
    parser.add_argument('--wait', action='store_true',
                       help='Wait for the appendix detection batch to complete')
    parser.add_argument('--skip-detection', action='store_true',
                       help='Skip appendix detection and go straight to extraction')
    parser.add_argument('--use-full-text', action='store_true',
                       help='Use full text for all documents (only with --skip-detection)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit processing to the first N documents')
    parser.add_argument('--continue-from', type=str, default=None,
                       help='Continue from an existing appendix detection batch ID')
    parser.add_argument('--custom-id-map', type=str, default=None,
                       help='Path to an existing custom ID map file (used with --continue-from)')
    args = parser.parse_args()
    
    # Determine directories
    if not args.data_dir:
        args.data_dir = os.path.join(os.path.dirname(__file__), "../../data/SupremeCourtData/")
    
    if not args.text_dir:
        args.text_dir = os.path.join(args.data_dir, "brieftext")
    
    if not args.output_dir:
        args.output_dir = os.path.join(args.data_dir, "batch_results")
    
    # Ensure directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.text_dir):
        logger.error(f"Text directory not found: {args.text_dir}")
        return
    
    # Load the mapping from document IDs to blob names
    doc_id_to_blob, blob_to_doc_id = load_embedded_text_mapping(args.data_dir)
    print(f"Loaded {len(doc_id_to_blob)} document mappings")
    
    # Prepare documents from text files
    documents = prepare_documents_from_text_files(args.text_dir, doc_id_to_blob)
    print(f"Prepared {len(documents)} documents from text files")
    
    if args.limit and args.limit < len(documents):
        documents = documents[:args.limit]
        logger.info(f"Limited processing to {args.limit} documents")
    
    if len(documents) == 0:
        logger.error("No documents found to process")
        return
    
    logger.info(f"Processing {len(documents)} documents")
    
    # Run the batch processing pipeline
    if args.continue_from:
        logger.info(f"Continuing from existing appendix detection batch: {args.continue_from}")
        result = continue_from_appendix_batch(
            args.continue_from, 
            args.output_dir, 
            documents,
            args.custom_id_map
        )
    elif args.skip_detection:
        logger.info("Skipping appendix detection stage")
        result = process_without_detection(documents, args.output_dir, args.use_full_text)
    else:
        logger.info("Running two-stage batch processing")
        result = two_stage_batch_process(documents, args.output_dir, args.wait)
    
    # Print summary
    logger.info("Batch processing pipeline initiated")
    logger.info(f"Results: {json.dumps(result, indent=2)}")
    
    print("\nBatch Processing Summary:")
    print(f"- Documents processed: {len(documents)}")
    
    if not args.skip_detection or args.continue_from:
        print(f"- Appendix detection batch ID: {result['appendix_detection']['batch_id']}")
        print(f"- Appendix detection status: {result['appendix_detection']['status']}")
    
    if 'information_extraction' in result:
        if 'batch_ids' in result['information_extraction']:
            # Multiple batch IDs case
            ids = result['information_extraction']['batch_ids']
            print(f"- Information extraction batch IDs: {', '.join(ids)}")
            print("\nTo check batch statuses, run:")
            for bid in ids:
                print(f"  python -c \"from batch_input import check_batch_status; print(check_batch_status('{bid}'))\"")
        else:
            # Single batch ID case (deprecated)
            print(f"- Information extraction batch ID: {result['information_extraction'].get('batch_id', 'N/A')}")
            print("\nTo check batch status, run:")
            print(f"  python -c \"from batch_input import check_batch_status; print(check_batch_status('{result['information_extraction'].get('batch_id', 'N/A')}'))\"")
    
    print(f"- Custom ID mapping saved to: {result.get('custom_id_map', 'N/A')}")
    print(f"- Timestamp: {result['timestamp']}")
    
    print("\nTo check batch status later, run:")
    
    if not args.skip_detection or args.continue_from:
        print(f"  python -c \"from batch_input import check_batch_status; print(check_batch_status('{result['appendix_detection']['batch_id']}'))\"")
    
    if 'information_extraction' in result:
        print(f"  python -c \"from batch_input import check_batch_status; print(check_batch_status('{result['information_extraction']['batch_id']}'))\"")


if __name__ == "__main__":
    main()