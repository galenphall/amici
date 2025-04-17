#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import argparse
import logging
import pandas as pd
import uuid
import fitz  # PyMuPDF
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed  # Added as_completed
from functools import partial
from typing import Tuple, Dict, List, Optional, Any
from tqdm import tqdm
import tempfile
import time
import io
import json  # Add import for JSON support

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.utils.gcs import GCSFetch


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(f"amicus_ocr_{time.strftime('%Y%m%d-%H%M%S')}.log")  # Log to file
    ]
)
logger = logging.getLogger("amicus_ocr")


def setup_directories(base_dir: Path) -> Tuple[Path, Path, Path]:
    """
    Create necessary directories for processing.
    
    Args:
        base_dir: Base directory for all data
        
    Returns:
        Tuple of (text_dir, ocr_dir, temp_dir) paths
    """
    # Directory for extracted text
    text_dir = base_dir / "brieftext"
    text_dir.mkdir(exist_ok=True)
    
    # Directory for OCR'd PDFs
    ocr_dir = base_dir / "ocr"
    ocr_dir.mkdir(exist_ok=True)
    
    # Temporary directory for downloaded PDFs
    temp_dir = base_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    return text_dir, ocr_dir, temp_dir


def get_text_blob_name(pdf_blob_name: str) -> str:
    """
    Convert a PDF blob name to a text blob name.
    
    Args:
        pdf_blob_name: Name of the PDF blob
        
    Returns:
        Name of the corresponding text blob
    """
    return Path(pdf_blob_name).with_suffix('.txt').as_posix()


def cache_all_blob_names(fetcher: GCSFetch, data_dir: Path, prefix: str = 'SUPREMECOURT/') -> List[str]:
    """
    Download and cache all blob names from the GCS bucket.
    
    Args:
        fetcher: GCSFetch instance for accessing GCS
        data_dir: Directory to save the cache file
        prefix: Prefix to filter blobs (default: 'SUPREMECOURT/')
        
    Returns:
        List of all blob names in the bucket
    """
    logger.info(f"Downloading all blob names from GCS with prefix '{prefix}'...")
    blobs = list(map(lambda b: b.name, fetcher.list_blobs(prefix)))
    
    # Cache the blobs locally
    cache_file = data_dir / "gcs_blobs_cache.json"
    with open(cache_file, "w") as f:
        json.dump(blobs, f)
    
    logger.info(f"Cached {len(blobs)} blob names to {cache_file}")
    return blobs


def check_text_exists_in_gcs(fetcher: GCSFetch, pdf_blob: str, all_blobs: Optional[List[str]] = None) -> bool:
    """
    Check if a text file already exists in GCS for this PDF.
    
    Args:
        fetcher: GCSFetch instance for accessing GCS
        pdf_blob: Name of the PDF blob
        all_blobs: Optional list of all blob names to check against (for efficiency)
        
    Returns:
        True if a text file exists, False otherwise
    """
    text_blob = get_text_blob_name(pdf_blob)
    
    if all_blobs:
        # Use the cached list to check if the blob exists
        return text_blob in all_blobs
    else:
        try:
            # Use list_blobs with the exact path to check if the blob exists
            blobs = list(fetcher.list_blobs(text_blob))
            return len(blobs) > 0
        except Exception as e:
            logger.error(f"Error checking if text exists for {pdf_blob}: {e}")
            return False


def load_or_create_tracking_file(fetcher: GCSFetch, tracking_file: str) -> pd.DataFrame:
    """
    Load an existing tracking file from GCS or create a new one.
    
    Args:
        fetcher: GCSFetch instance for accessing GCS
        tracking_file: Name of the tracking file in GCS
        
    Returns:
        DataFrame with tracking data
    """
    try:
        # Check if tracking file exists using list_blobs
        blobs = list(fetcher.list_blobs(tracking_file))
        if len(blobs) > 0:
            logger.info(f"Loading existing tracking file from GCS: {tracking_file}")
            content, _ = fetcher.get_from_bucket(tracking_file)
            return pd.read_csv(io.BytesIO(content), index_col=0)
        else:
            logger.info(f"Creating new tracking file: {tracking_file}")
            return pd.DataFrame(columns=['has_text', 'needed_ocr', 'error'])
    except Exception as e:
        logger.error(f"Error loading tracking file from GCS: {e}")
        return pd.DataFrame(columns=['has_text', 'needed_ocr', 'error'])


def upload_tracking_file(fetcher: GCSFetch, df: pd.DataFrame, tracking_file: str) -> bool:
    """
    Upload tracking file to GCS.
    
    Args:
        fetcher: GCSFetch instance for accessing GCS
        df: DataFrame to upload
        tracking_file: Name of the tracking file in GCS
        
    Returns:
        True if upload was successful, False otherwise
    """
    try:
        # Convert DataFrame to CSV
        csv_data = df.to_csv().encode('utf-8')
        # Upload to GCS using file-like object
        file_obj = io.BytesIO(csv_data)
        fetcher.upload_to_bucket(tracking_file, file_obj)
        logger.info(f"Uploaded tracking file to GCS: {tracking_file}")
        return True
    except Exception as e:
        logger.error(f"Error uploading tracking file to GCS: {e}")
        return False


def check_has_embedded_text(pdf_content: bytes, min_text_length: int = 100) -> bool:
    """
    Check if a PDF has embedded text.
    
    Args:
        pdf_content: PDF file content as bytes
        min_text_length: Minimum length of text to consider it has embedded text
        
    Returns:
        True if the PDF has embedded text, False otherwise
    """
    try:
        with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
            text = ""
            # Check first 3 pages for text (often enough for cover and intro)
            for page_num in range(min(3, len(pdf))):
                page = pdf[page_num]
                page_text = page.get_text()
                text += page_text
                # Early exit if we've found enough text
                if len(text) > min_text_length:
                    return True
            
            # If we've checked all pages and still don't have enough text
            return len(text) > min_text_length
    except Exception as e:
        logger.error(f"Error checking for embedded text: {e}")
        return False


def extract_text_from_pdf(pdf_path: Path, metadata: Dict[str, Any] = None) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        metadata: Optional metadata to include in the text header
        
    Returns:
        Extracted text
    """
    try:
        with fitz.open(str(pdf_path)) as pdf:
            text = ""
            
            # Add metadata header if provided
            if metadata:
                text += "--- PDF METADATA ---\n"
                for key, value in metadata.items():
                    if value:
                        text += f"{key}: {value}\n"
                text += "--- END METADATA ---\n\n"
            
            pages = list(pdf)
            # Keep the first 4 and last 4 pages.
            for page in pages[:4] + pages[-4:]:
                text += page.get_text()
                text += "\n\n--Page Break--\n\n"
            
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return f"ERROR: Failed to extract text - {str(e)}"


def process_blob_for_text(
    blob_name: str, 
    fetcher: GCSFetch, 
    temp_dir: Path,
    tracking_df: pd.DataFrame,
    force: bool = False
) -> Tuple[str, bool, bool, Optional[str]]:
    """
    Process a single blob to check for embedded text, run OCR if needed, and extract text.
    
    Args:
        blob_name: Name of the blob in GCS
        fetcher: GCSFetch instance for downloading blobs
        temp_dir: Directory to save downloaded PDFs temporarily
        tracking_df: DataFrame to track processing status
        force: Whether to force reprocessing of files
        
    Returns:
        Tuple of (blob_name, has_text, needed_ocr, error_message)
    """
    # Check if we've already processed this blob and it's in the tracking DataFrame
    if blob_name in tracking_df.index and not force:
        if tracking_df.loc[blob_name, 'has_text']:
            logger.debug(f"Blob {blob_name} already processed successfully")
            return blob_name, True, tracking_df.loc[blob_name, 'needed_ocr'], None
    
    # Check if text file already exists in GCS
    if check_text_exists_in_gcs(fetcher, blob_name) and not force:
        logger.debug(f"Text file already exists in GCS for {blob_name}")
        tracking_df.loc[blob_name, 'has_text'] = True
        # We don't know if it needed OCR, so we'll set it to NaN if not already set
        if 'needed_ocr' not in tracking_df.columns or pd.isna(tracking_df.loc[blob_name, 'needed_ocr']):
            tracking_df.loc[blob_name, 'needed_ocr'] = False
        return blob_name, True, tracking_df.loc[blob_name, 'needed_ocr'], None
    
    try:
        # Download the PDF
        pdf_path = temp_dir / f"{Path(blob_name).name}"
        if pdf_path.exists() and not force:
            logger.debug(f"Using cached PDF at {pdf_path}")
            content = pdf_path.read_bytes()
        else:
            # Download the blob
            content, _ = fetcher.get_from_bucket(blob_name)
            
            # Cache the PDF locally
            with open(pdf_path, "wb") as f:
                f.write(content)
            logger.debug(f"Downloaded PDF to {pdf_path}")
        
        # Check if the PDF has embedded text
        has_text = check_has_embedded_text(content)
        needed_ocr = False
        
        if has_text:
            logger.debug(f"PDF {blob_name} has embedded text")
            
            # Extract text from the PDF
            metadata = {
                "original_file": blob_name,
                "needed_ocr": "False"
            }
            text = extract_text_from_pdf(pdf_path, metadata)
            
            # Upload text to GCS
            text_blob_name = get_text_blob_name(blob_name)
            # Create a file-like object for the text content
            text_file = io.BytesIO(text.encode('utf-8'))
            # Use sidecar instead of metadata
            fetcher.upload_to_bucket(
                text_blob_name, 
                text_file,
                sidecar={
                    "original_file": blob_name,
                    "needed_ocr": "False"
                }
            )
            
            logger.info(f"Uploaded extracted text to {text_blob_name}")
            
        else:
            logger.debug(f"PDF {blob_name} needs OCR")
            needed_ocr = True
            
            # Run OCR on the PDF
            ocr_pdf_path = temp_dir / f"ocr_{Path(blob_name).name}"
            
            # Run OCR
            logger.debug(f"Running OCR on {pdf_path}")
            subprocess.run(
                ["ocrmypdf", "--force-ocr", str(pdf_path), str(ocr_pdf_path)],
                check=True,
                capture_output=True
            )
            logger.debug(f"OCR completed for {blob_name}, saved to {ocr_pdf_path}")
            
            # Extract text from the OCR'd PDF
            metadata = {
                "original_file": blob_name,
                "needed_ocr": "True"
            }
            text = extract_text_from_pdf(ocr_pdf_path, metadata)
            
            # Upload text to GCS
            text_blob_name = get_text_blob_name(blob_name)
            # Create a file-like object for the text content
            text_file = io.BytesIO(text.encode('utf-8'))
            # Use sidecar instead of metadata
            fetcher.upload_to_bucket(
                text_blob_name, 
                text_file,
                sidecar={
                    "original_file": blob_name,
                    "needed_ocr": "True"
                }
            )
            
            logger.info(f"Uploaded OCR'd text to {text_blob_name}")
            
            # Upload the OCR'd PDF back to the original blob location
            with open(ocr_pdf_path, 'rb') as f:
                ocr_pdf_file = io.BytesIO(f.read())
                
            # Use sidecar instead of metadata
            fetcher.upload_to_bucket(
                blob_name, 
                ocr_pdf_file,
                sidecar={
                    "ocr_applied": "True",
                    "ocr_date": time.strftime("%Y-%m-%d")
                }
            )
            
            logger.info(f"Uploaded OCR'd PDF back to {blob_name}")
            
            has_text = True
        
        # Update tracking information
        tracking_df.loc[blob_name, 'has_text'] = has_text
        tracking_df.loc[blob_name, 'needed_ocr'] = needed_ocr
        tracking_df.loc[blob_name, 'error'] = None
        
        return blob_name, has_text, needed_ocr, None
    
    except Exception as e:
        error_msg = f"Error processing {blob_name}: {str(e)}"
        logger.error(error_msg)
        
        # Update tracking information
        tracking_df.loc[blob_name, 'has_text'] = False
        tracking_df.loc[blob_name, 'needed_ocr'] = None
        tracking_df.loc[blob_name, 'error'] = str(e)
        
        return blob_name, False, None, error_msg


def main():
    """Main function to orchestrate the OCR processing pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process amicus briefs to extract text via OCR.')
    parser.add_argument('--data-dir', type=str, default=None, 
                        help='Directory for data files (default: ../data/)')
    parser.add_argument('--bucket', type=str, default="interest_groups_raw_documents_2025",
                        help='GCS bucket name')
    parser.add_argument('--parallel', type=int, default=8,
                        help='Number of parallel threads')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of all files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit processing to N files (for testing)')
    args = parser.parse_args()
    
    # Determine the data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent / "data" 
    
    data_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using data directory: {data_dir}")
    
    # Set up directories
    text_dir, ocr_dir, temp_dir = setup_directories(data_dir)
    
    # Initialize GCS fetcher
    fetcher = GCSFetch(args.bucket)
    logger.info(f"Initialized GCS fetcher for bucket: {args.bucket}")
    
    # Define tracking file in GCS
    tracking_file = "amicus_text_extraction_status.csv"
    
    # Load or create tracking DataFrame
    tracking_df = load_or_create_tracking_file(fetcher, tracking_file)
    
    # Load the amicus briefs datafile
    amicus_briefs_path = data_dir / "amicus_briefs.csv"
    if not amicus_briefs_path.exists():
        # Try to load from GCS if not found locally
        logger.info(f"Amicus briefs file not found locally, attempting to download from GCS: {amicus_briefs_path}")
        try:
            content, _ = fetcher.get_from_bucket("amicus_briefs.csv")
            with open(amicus_briefs_path, "wb") as f:
                f.write(content)
            logger.info(f"Downloaded amicus briefs file from GCS: {amicus_briefs_path}")
        except Exception as e:
            logger.error(f"Failed to download amicus briefs file from GCS: {e}")
            sys.exit(1)
            
    logger.info(f"Loading amicus briefs from: {amicus_briefs_path}")
    
    amicus_briefs = pd.read_csv(amicus_briefs_path)
    
    # Add columns for tracking if they don't exist
    if 'transcribed' not in amicus_briefs.columns:
        amicus_briefs['transcribed'] = False
    if 'neededOCR' not in amicus_briefs.columns:
        amicus_briefs['neededOCR'] = False
    
    # Get all blob names
    amicus_blobs = list(amicus_briefs["blob"].values)
    
    if args.limit:
        amicus_blobs = amicus_blobs[:args.limit]
        logger.info(f"Limited processing to {args.limit} files")
    
    logger.info(f"Found {len(amicus_blobs)} amicus briefs to process")
    
    # Download all blob names at once for efficient checking
    logger.info("Downloading all blob names from GCS for efficient checking...")
    all_blobs = cache_all_blob_names(fetcher, data_dir, prefix='SUPREMECOURT/')
    
    # Check which blobs already have text in GCS
    logger.info("Checking for existing text files in GCS...")
    for blob in tqdm(amicus_blobs, desc="Checking existing text"):
        if check_text_exists_in_gcs(fetcher, blob, all_blobs):
            # Update the tracking DataFrame and amicus_briefs DataFrame
            tracking_df.loc[blob, 'has_text'] = True
            amicus_briefs.loc[amicus_briefs['blob'] == blob, 'transcribed'] = True
    
    # Upload initial tracking file to GCS
    upload_tracking_file(fetcher, tracking_df, tracking_file)
    
    # Find blobs that need processing
    blobs_to_process = [blob for blob in amicus_blobs if blob not in tracking_df.index or not tracking_df.loc[blob, 'has_text'] or args.force]
    
    if not blobs_to_process:
        logger.info("All PDFs have already been processed")
        
        # Save the updated amicus_briefs DataFrame locally and upload to GCS
        amicus_briefs.to_csv(amicus_briefs_path, index=False)
        amicus_briefs_file = io.BytesIO(amicus_briefs.to_csv().encode('utf-8'))
        fetcher.upload_to_bucket("amicus_briefs.csv", amicus_briefs_file)
        
        # Generate a summary report
        total_pdfs = len(tracking_df)
        total_with_text = tracking_df['has_text'].sum()
        total_needed_ocr = tracking_df['needed_ocr'].sum()
        
        logger.info(f"Overall summary:")
        logger.info(f"  Total PDFs: {total_pdfs}")
        logger.info(f"  PDFs with text: {total_with_text} ({total_with_text/total_pdfs:.1%})")
        logger.info(f"  PDFs that needed OCR: {total_needed_ocr} ({total_needed_ocr/total_pdfs:.1%})")
        logger.info(f"  PDFs without text: {total_pdfs - total_with_text} ({(total_pdfs - total_with_text)/total_pdfs:.1%})")
        
        return
    
    logger.info(f"Processing {len(blobs_to_process)} PDFs")
    
    # Create a partial function with fixed arguments
    process_fn = partial(
        process_blob_for_text, 
        fetcher=fetcher, 
        temp_dir=temp_dir,
        tracking_df=tracking_df,
        force=args.force
    )
    
    # Process blobs in parallel
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for blob in blobs_to_process:
            futures.append(executor.submit(process_fn, blob))
        
        # Process results as they complete
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs")):
            blob, has_text, needed_ocr, error = future.result()
            
            # Update the amicus_briefs DataFrame
            if blob in amicus_briefs['blob'].values:
                amicus_briefs.loc[amicus_briefs['blob'] == blob, 'transcribed'] = has_text
                if needed_ocr is not None:
                    amicus_briefs.loc[amicus_briefs['blob'] == blob, 'neededOCR'] = needed_ocr
            
            # Upload tracking file periodically
            if i % 10 == 0:
                upload_tracking_file(fetcher, tracking_df, tracking_file)
    
    # Save the updated amicus_briefs DataFrame locally
    amicus_briefs.to_csv(amicus_briefs_path, index=False)
    
    # Upload final tracking file and amicus_briefs to GCS
    upload_tracking_file(fetcher, tracking_df, tracking_file)
    amicus_briefs_file = io.BytesIO(amicus_briefs.to_csv().encode('utf-8'))
    fetcher.upload_to_bucket("amicus_briefs.csv", amicus_briefs_file)
    
    # Generate a summary report
    total_pdfs = len(tracking_df)
    total_with_text = tracking_df['has_text'].sum()
    total_needed_ocr = tracking_df['needed_ocr'].sum()
    total_errors = len(tracking_df[tracking_df['error'].notna()])
    
    logger.info(f"Processing complete!")
    logger.info(f"Overall summary:")
    logger.info(f"  Total PDFs processed: {total_pdfs}")
    logger.info(f"  PDFs with text: {total_with_text} ({total_with_text/total_pdfs:.1%})")
    logger.info(f"  PDFs that needed OCR: {total_needed_ocr} ({total_needed_ocr/total_pdfs:.1%})")
    logger.info(f"  PDFs with errors: {total_errors} ({total_errors/total_pdfs:.1%})")
    logger.info(f"  Results saved to GCS: {tracking_file}")
    
    
if __name__ == "__main__":
    main()