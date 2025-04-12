import os
import fitz  # PyMuPDF
import subprocess
import tempfile
import logging
import io

logger = logging.getLogger(__name__)

def check_has_embedded_text(pdf_content, min_text_length=100):
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
        logger.error(f"Error checking for embedded text: {str(e)}")
        return False

def extract_text_from_pdf(pdf_content, extract_all_pages=False):
    """
    Extract text from a PDF.
    
    Args:
        pdf_content: PDF file content as bytes
        extract_all_pages: Whether to extract all pages or just important ones
        
    Returns:
        Extracted text
    """
    try:
        with fitz.open(stream=pdf_content, filetype="pdf") as pdf:
            text = ""
            
            if extract_all_pages:
                # Extract text from all pages
                for page in pdf:
                    text += page.get_text()
                    text += "\n\n--Page Break--\n\n"
            else:
                # Extract text from just first 5 pages (for appendix detection)
                pages = list(pdf)
                for page in pages[:5]:
                    text += page.get_text()
                    text += "\n\n--Page Break--\n\n"
            
            return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def run_ocr_on_pdf(pdf_content):
    """
    Run OCR on a PDF without embedded text.
    
    Args:
        pdf_content: PDF file content as bytes
        
    Returns:
        OCR'd PDF content as bytes
    """
    try:
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as input_pdf:
            input_pdf.write(pdf_content)
            input_pdf_path = input_pdf.name
            
        output_pdf_path = input_pdf_path + "_ocr.pdf"
        
        # Run OCR
        subprocess.run(
            ["ocrmypdf", "--force-ocr", input_pdf_path, output_pdf_path],
            check=True,
            capture_output=True
        )
        
        # Read the OCR'd PDF
        with open(output_pdf_path, "rb") as f:
            ocr_pdf_content = f.read()
            
        # Clean up temporary files
        try:
            os.unlink(input_pdf_path)
            os.unlink(output_pdf_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")
            
        return ocr_pdf_content
    except Exception as e:
        logger.error(f"Error running OCR on PDF: {str(e)}")
        return pdf_content  # Return original content if OCR fails