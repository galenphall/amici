import os
import fitz  # PyMuPDF
import logging
from ..transformation.ocr import check_has_embedded_text, extract_text_from_pdf, run_ocr_on_pdf

logger = logging.getLogger(__name__)

def format_metadata_header(metadata):
    """
    Format metadata as a header for text files.
    
    Args:
        metadata: Dictionary of metadata fields
        
    Returns:
        Formatted header string
    """
    header = "--- METADATA START ---\n"
    
    for key, value in metadata.items():
        if value:
            header += f"{key}: {value}\n"
    
    header += "--- METADATA END ---\n\n"
    return header

def process_pdf_for_text(pdf_content, metadata=None):
    """
    Process a PDF to extract text, running OCR if needed.
    
    Args:
        pdf_content: PDF file content as bytes
        metadata: Dictionary of metadata to include in the header
        
    Returns:
        Tuple of (extracted_text, needed_ocr)
    """
    if metadata is None:
        metadata = {}
    
    # Check if PDF has embedded text
    has_text = check_has_embedded_text(pdf_content)
    
    if has_text:
        # Extract text directly
        logger.info("PDF has embedded text, extracting directly")
        text = extract_text_from_pdf(pdf_content, extract_all_pages=True)
        return format_metadata_header(metadata) + text, False
    else:
        # Run OCR and then extract text
        logger.info("PDF does not have embedded text, running OCR")
        ocr_pdf_content = run_ocr_on_pdf(pdf_content)
        text = extract_text_from_pdf(ocr_pdf_content, extract_all_pages=True)
        return format_metadata_header(metadata) + text, True