#!/usr/bin/env python3
"""
Script to find amicus briefs and extract text. For those with embedded text,
we get the text directly and store it locally. For those without, we need to 
run OCR. We save the file locations and run OCR locally.

This script provides a robust pipeline that:
1. Checks if PDFs have embedded text
2. Extracts embedded text where available
3. Runs OCR on PDFs without embedded text
4. Saves all extracted text to individual files
5. Tracks processing status for resumability
"""
import sys
import os
from pathlib import Path
import argparse
import logging
import pandas as pd
import uuid
import fitz  # PyMuPDF
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Tuple, Dict, List, Optional, Any
from tqdm import tqdm
import tempfile
import time

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))
from parser.gcs_fetch import GCSFetch


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


def load_or_create_tracking_file(data_dir: Path, filename: str) -> pd.DataFrame:
    """
    Load an existing tracking file or create a new one.
    
    Args:
        data_dir: Directory where tracking files are stored
        filename: Name of the tracking file
        
    Returns:
        DataFrame with tracking data
    """
    file_path = data_dir / filename
    if file_path.exists():
        logger.info(f"Loading existing tracking file: {file_path}")
        return pd.read_csv(file_path, index_col=0)
    else:
        logger.info(f"Creating new tracking file: {file_path}")
        return pd.DataFrame()


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


def process_blob_for_text(
    blob_name: str, 
    fetcher: GCSFetch, 
    text_dir: Path,
    temp_dir: Path,
    cache_pdf: bool = True
) -> Tuple[str, bool, Optional[str], Optional[str]]:
    """
    Process a single blob to check for embedded text and extract it if present.
    
    Args:
        blob_name: Name of the blob in GCS
        fetcher: GCSFetch instance for downloading blobs
        text_dir: Directory to save extracted text
        temp_dir: Directory to save downloaded PDFs temporarily
        cache_pdf: Whether to cache the downloaded PDF locally
        
    Returns:
        Tuple of (blob_name, has_text, doc_id, error_message)
    """
    try:
        # Check if the pdf has already been cached
        pdf_path = temp_dir / f"{Path(blob_name).name}"
        if pdf_path.exists():
            logger.debug(f"Using cached PDF at {pdf_path}")
            content = pdf_path.read_bytes()
        else:
            # Download the blob
            content, _ = fetcher.get_from_bucket(blob_name)
            
            # Cache the PDF if requested
            if cache_pdf:
                pdf_path = temp_dir / f"{Path(blob_name).name}"
                with open(pdf_path, "wb") as f:
                    f.write(content)
                logger.debug(f"Cached PDF at {pdf_path}")
        
        # Check if the PDF has embedded text
        has_text = check_has_embedded_text(content)
        
        if has_text:
            # Generate a unique ID for this document
            doc_id = str(uuid.uuid4())
            
            # Extract and save the text
            with fitz.open(stream=content, filetype="pdf") as pdf:
                text = ""
                pages = list(pdf)
                # Keep the first 4 and last 4 pages.
                for page in pages[:4] + pages[-4:]:
                    text += page.get_text()
                    text += "\n\n--Page Break--\n\n"
            
            # Save the extracted text
            text_path = text_dir / f"{doc_id}.txt"
            with open(text_path, "w", encoding='utf-8') as f:
                f.write(text)
            
            logger.debug(f"Extracted embedded text from {blob_name} to {text_path}")
            return blob_name, True, doc_id, None
        else:
            return blob_name, False, None, None
    
    except Exception as e:
        error_msg = f"Error processing {blob_name}: {str(e)}"
        logger.error(error_msg)
        return blob_name, False, None, error_msg


def run_ocr_on_pdf(
    blob_name: str, 
    doc_id: str,
    fetcher: GCSFetch,
    temp_dir: Path, 
    ocr_dir: Path,
    text_dir: Path,
    force: bool = False
) -> Tuple[str, bool, Optional[str]]:
    """
    Run OCR on a PDF and extract text.
    
    Args:
        blob_name: Name of the blob in GCS
        doc_id: Document ID for this blob
        fetcher: GCSFetch instance for downloading blobs
        temp_dir: Directory for temporary files
        ocr_dir: Directory for OCR'd PDFs
        text_dir: Directory for extracted text
        force: Whether to force OCR even if the file exists
        
    Returns:
        Tuple of (blob_name, success, error_message)
    """
    try:
        input_pdf_path = temp_dir / f"{Path(blob_name).name}"
        ocr_pdf_path = ocr_dir / f"ocr_{Path(blob_name).name}"
        text_path = text_dir / f"{doc_id}.txt"
        
        # Skip if the text file already exists and we're not forcing
        if text_path.exists() and not force:
            logger.debug(f"Text file already exists for {blob_name}: {text_path}")
            return blob_name, True, None
        
        # Download the PDF if it doesn't exist
        if not input_pdf_path.exists():
            content, _ = fetcher.get_from_bucket(blob_name)
            with open(input_pdf_path, "wb") as f:
                f.write(content)
            logger.debug(f"Downloaded PDF to {input_pdf_path}")
        
        # Skip OCR if the OCR'd PDF already exists and we're not forcing
        if ocr_pdf_path.exists() and not force:
            logger.debug(f"OCR'd PDF already exists for {blob_name}: {ocr_pdf_path}")
        else:
            # Run OCR
            logger.debug(f"Running OCR on {input_pdf_path}")
            subprocess.run(
                ["ocrmypdf", "--force-ocr", str(input_pdf_path), str(ocr_pdf_path)],
                check=True,
                capture_output=True
            )
            logger.debug(f"OCR completed for {blob_name}, saved to {ocr_pdf_path}")
        
        # Extract text from the OCR'd PDF
        with fitz.open(str(ocr_pdf_path)) as pdf:
            text = ""
            pages = list(pdf)
            # Keep the first 4 and last 4 pages.
            for page in pages[:4] + pages[-4:]:
                text += page.get_text()
                text += "\n\n--Page Break--\n\n"
        
        # Save the extracted text
        with open(text_path, "w", encoding='utf-8') as f:
            f.write(text)
        
        logger.debug(f"Extracted OCR text from {blob_name} to {text_path}")
        return blob_name, True, None
    
    except Exception as e:
        error_msg = f"Error running OCR on {blob_name}: {str(e)}"
        logger.error(error_msg)
        return blob_name, False, error_msg


def main():
    """Main function to orchestrate the OCR processing pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process amicus briefs to extract text via OCR.')
    parser.add_argument('--data-dir', type=str, default=None, 
                        help='Directory for data files (default: ../../data/SupremeCourtData/)')
    parser.add_argument('--bucket', type=str, default="interest_groups_raw_documents_2025",
                        help='GCS bucket name')
    parser.add_argument('--parallel', type=int, default=8,
                        help='Number of parallel threads')
    parser.add_argument('--check', action='store_true',
                        help='Only check for embedded text, do not run OCR')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of all files')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit processing to N files (for testing)')
    args = parser.parse_args()
    
    # Determine the data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).parent.parent.parent / "data" / "SupremeCourtData"
    
    data_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using data directory: {data_dir}")
    
    # Set up directories
    text_dir, ocr_dir, temp_dir = setup_directories(data_dir)
    
    # Initialize GCS fetcher
    fetcher = GCSFetch(args.bucket)
    logger.info(f"Initialized GCS fetcher for bucket: {args.bucket}")
    
    # Load the amicus briefs datafile
    amicus_briefs_path = data_dir / "amicus_briefs.csv"
    if not amicus_briefs_path.exists():
        logger.error(f"Amicus briefs CSV not found: {amicus_briefs_path}")
        sys.exit(1)
    
    amicus_briefs = pd.read_csv(amicus_briefs_path)
    amicus_blobs = list(amicus_briefs["blob"].values)
    
    if args.limit:
        amicus_blobs = amicus_blobs[:args.limit]
        logger.info(f"Limited processing to {args.limit} files")
    
    logger.info(f"Found {len(amicus_blobs)} amicus briefs to process")
    
    # Load or create tracking DataFrame
    embedded_map_path = data_dir / "embedded_text_map.csv"
    if embedded_map_path.exists() and not args.force:
        logger.info(f"Loading existing embedded text mapping from {embedded_map_path}")
        embedded_map = pd.read_csv(embedded_map_path, index_col=0)
        
        # Convert string 'True'/'False' values to boolean
        if 'has_text' in embedded_map.columns:
            embedded_map['has_text'] = embedded_map['has_text'].astype(bool)
        
        # Check if we need to add any new blobs
        existing_blobs = set(embedded_map.index)
        new_blobs = [blob for blob in amicus_blobs if blob not in existing_blobs]
        
        if new_blobs:
            logger.info(f"Found {len(new_blobs)} new blobs to process")
            
            # Create a partial function with fixed arguments
            process_fn = partial(
                process_blob_for_text, 
                fetcher=fetcher, 
                text_dir=text_dir,
                temp_dir=temp_dir
            )
            
            # Process new blobs in parallel
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                results = list(tqdm(
                    executor.map(process_fn, new_blobs),
                    total=len(new_blobs),
                    desc="Processing new PDFs"
                ))
            
            # Add results to the tracking DataFrame
            new_data = {
                blob: {
                    'has_text': has_text, 
                    'doc_id': doc_id if doc_id else str(uuid.uuid4())
                } 
                for blob, has_text, doc_id, _ in results
            }
            
            new_df = pd.DataFrame.from_dict(new_data, orient='index')
            embedded_map = pd.concat([embedded_map, new_df])
            
            # Save updated mapping
            embedded_map.to_csv(embedded_map_path)
            logger.info(f"Updated embedded text mapping saved to {embedded_map_path}")
    else:
        # Process all blobs to check for embedded text
        logger.info("Processing all PDFs to check for embedded text")
        
        # Create a partial function with fixed arguments
        process_fn = partial(
            process_blob_for_text, 
            fetcher=fetcher, 
            text_dir=text_dir,
            temp_dir=temp_dir
        )
        
        # Process blobs in parallel
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            results = list(tqdm(
                executor.map(process_fn, amicus_blobs),
                total=len(amicus_blobs),
                desc="Processing PDFs"
            ))
        
        # Create tracking DataFrame
        data = {
            blob: {
                'has_text': has_text, 
                'doc_id': doc_id if doc_id else str(uuid.uuid4())
            } 
            for blob, has_text, doc_id, _ in results
        }
        
        embedded_map = pd.DataFrame.from_dict(data, orient='index')

        # Make sure that all blobs and 
        
        # Save mapping
        embedded_map.to_csv(embedded_map_path)
        logger.info(f"Embedded text mapping saved to {embedded_map_path}")
    
    # Summary of PDFs with/without embedded text
    pdfs_with_text = embedded_map['has_text'].sum()
    pdfs_without_text = len(embedded_map) - pdfs_with_text
    logger.info(f"PDFs with embedded text: {pdfs_with_text}")
    logger.info(f"PDFs requiring OCR: {pdfs_without_text}")
    
    # Exit if only checking for embedded text
    if args.check:
        logger.info("Check-only mode, exiting without running OCR")
        return
    
    # Run OCR on PDFs without embedded text
    blobs_for_ocr = embedded_map.index[~embedded_map['has_text']].tolist()
    
    if not blobs_for_ocr:
        logger.info("No PDFs require OCR, exiting")
        return
    
    logger.info(f"Running OCR on {len(blobs_for_ocr)} PDFs without embedded text")
    
    # Load or create OCR results tracking DataFrame
    ocr_results_path = data_dir / "ocr_results.csv"
    if ocr_results_path.exists() and not args.force:
        ocr_results = pd.read_csv(ocr_results_path, index_col=0)
        logger.info(f"Loaded existing OCR results from {ocr_results_path}")
        
        # Only process blobs that haven't been successfully processed
        successful_blobs = set(ocr_results.index[ocr_results['success']])
        blobs_for_ocr = [blob for blob in blobs_for_ocr if blob not in successful_blobs]
    else:
        ocr_results = pd.DataFrame(columns=['success', 'error'])
    
    if not blobs_for_ocr:
        logger.info("All PDFs requiring OCR have already been processed")
        return
    
    logger.info(f"Running OCR on {len(blobs_for_ocr)} remaining PDFs")
    
    # Create a partial function with fixed arguments
    ocr_fn = partial(
        run_ocr_on_pdf,
        fetcher=fetcher,
        temp_dir=temp_dir,
        ocr_dir=ocr_dir,
        text_dir=text_dir,
        force=args.force
    )
    
    # Run OCR in parallel
    with ThreadPoolExecutor(max_workers=max(1, args.parallel // 2)) as executor:
        # OCR is CPU-intensive, so we use fewer workers
        futures = []
        for blob in blobs_for_ocr:
            doc_id = embedded_map.loc[blob, 'doc_id']
            futures.append(executor.submit(ocr_fn, blob, doc_id))
        
        # Process results as they complete
        for future in tqdm(futures, desc="Running OCR"):
            blob, success, error = future.result()
            ocr_results.loc[blob] = [success, error]
            
            # Save results periodically
            if len(ocr_results) % 10 == 0:
                ocr_results.to_csv(ocr_results_path)
    
    # Save final OCR results
    ocr_results.to_csv(ocr_results_path)
    logger.info(f"OCR results saved to {ocr_results_path}")
    
    # Summary of OCR results
    successful_ocr = ocr_results['success'].sum()
    failed_ocr = len(ocr_results) - successful_ocr
    logger.info(f"PDFs successfully processed with OCR: {successful_ocr}")
    logger.info(f"PDFs with OCR errors: {failed_ocr}")
    
    # Update embedded_map to mark successful OCR'd PDFs as having text
    successful_ocr_blobs = ocr_results.index[ocr_results['success']].tolist()
    successful_ocr_blobs = [blob for blob in successful_ocr_blobs if blob in embedded_map.index]
    if successful_ocr_blobs:
        embedded_map.loc[successful_ocr_blobs, 'has_text'] = True
        embedded_map.to_csv(embedded_map_path)
        logger.info(f"Updated embedded text mapping to reflect successful OCR")
    
    # Generate a summary of text extraction
    total_pdfs = len(embedded_map)
    total_with_text = embedded_map['has_text'].sum()
    logger.info(f"Overall summary:")
    logger.info(f"  Total PDFs: {total_pdfs}")
    logger.info(f"  PDFs with text (embedded or OCR'd): {total_with_text} ({total_with_text/total_pdfs:.1%})")
    logger.info(f"  PDFs without text: {total_pdfs - total_with_text} ({(total_pdfs - total_with_text)/total_pdfs:.1%})")
    
    
if __name__ == "__main__":
    main()