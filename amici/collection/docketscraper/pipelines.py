import os
import io
import re
import logging

import itemadapter
import scrapy
import uuid
import pathlib

from google.cloud.storage import Client
from itemadapter import ItemAdapter
from google.cloud import storage
from google.api_core.exceptions import Forbidden as GoogleForbiddenException
from dotenv import load_dotenv

import sys
import os
from pathlib import Path

# Add the project root to the path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from amici.transformation.text_extraction import process_pdf_for_text
from amici.transformation.ocr import check_has_embedded_text, extract_text_from_pdf, run_ocr_on_pdf
from amici.utils.gcs import GCSFetch

logger = logging.getLogger(__name__)

class GCSPipeline:
    """
    A pipeline class for uploading files to GCS. Every time an item is scraped, it's run through this pipeline. Note that
    the pipeline must be enabled in settings.py, which it should get by default.
    """

    def __init__(self):
        # GCS boilerplate code
        # Requires the GOOGLE_APPLICATION_CREDENTIALS environment variable

        load_dotenv("../../../.env")
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "Google auth credentials not set. Make sure to include in the .env file in /env directory.")

        storage_client: storage.Client = storage.Client()
        
        # Initialize and check raw documents bucket
        raw_bucket_name = os.getenv("RAW_BUCKET", "N/A")
        if raw_bucket_name == "N/A":
            raise ValueError("RAW_BUCKET environment variable is not set.")
        self.bucket = storage_client.bucket(raw_bucket_name)

        self.bucket_folder = os.getenv("BUCKET_FOLDER", "N/A")
        self.text_bucket_folder = os.getenv("TEXT_BUCKET_FOLDER", "N/A")


    def process_item(self, item: itemadapter.ItemAdapter.item, spider: scrapy.Spider) -> itemadapter.ItemAdapter.item:
        """
        Uploads the file to the raw documents GCS bucket.
        For amicus briefs, also extracts text and uploads to the text bucket.
        """
        # Transform the item into a dictionary
        adapted = ItemAdapter(item)

        if adapted.get('rawcontent'):
            # Get the file extension
            filetype: str = adapted.get("filetype")[0] if adapted.get("filetype") else None
            raw_content = adapted.get('rawcontent')[0]

            # type check
            if isinstance(raw_content, str):
                file = io.BytesIO(raw_content.encode('utf-8'))
            elif isinstance(raw_content, bytes):
                file = io.BytesIO(raw_content)
            else:
                # Try to convert to bytes
                file = io.BytesIO(str(raw_content).encode('utf-8'))

            # Format the path for the file so it goes into the correct folder.
            # Remove https:// from the URL string
            parsed_url = re.sub("https?://", "", adapted.get("url")[0])

            if parsed_url.endswith('/'):
                parsed_url = parsed_url[:-1]

            # Naming scheme: <state>/<domain>/<sub-pages>/<file><.extension>
            # If the file already has an extension, keep it.
            if re.search("\.[a-z]+$", parsed_url):
                new_file_name = f'{self.bucket_folder}/{parsed_url}'
            # Otherwise, add the inferred extension
            else:
                new_file_name = f'{self.bucket_folder}/{parsed_url}{filetype}'

            blob = self.bucket.blob(new_file_name)

            # Add custom metadata to GCS object, if applicable.
            sidecar = {}

            metadata = adapted.get('metadata')
            if metadata and isinstance(metadata[0], dict):
                sidecar.update(metadata[0])

            for key in ['state', 'doctype', 'accessed', 'url', 'filetype']:
                if adapted.get(key):
                    sidecar[key] = adapted.get(key)[0]

            blob.metadata = {k: str(v) for k, v in sidecar.items()}

            # Upload the raw file
            blob.upload_from_file(file, timeout=900)
            file.seek(0)  # Reset file pointer for potential reuse
            
            # Check if this is an amicus brief and a PDF
            is_amicus = adapted.get('is_amicus_brief') and adapted.get('is_amicus_brief')[0]
            is_pdf = filetype and filetype.lower() == '.pdf'
            
            if is_amicus and is_pdf:
                logger.info(f"Processing amicus brief PDF: {new_file_name}")
                
                # Collect metadata for the text file header
                text_metadata = {
                    "case_title": adapted.get('case_title')[0] if adapted.get('case_title') else None,
                    "docket_page": adapted.get('docket_page')[0] if adapted.get('docket_page') else None,
                    "date": adapted.get('date')[0] if adapted.get('date') else None,
                    "url": adapted.get('url')[0] if adapted.get('url') else None,
                    "original_file": new_file_name
                }
                
                # Extract text from PDF (with OCR if needed)
                pdf_content = raw_content if isinstance(raw_content, bytes) else raw_content.encode('utf-8')
                text_content, needed_ocr = process_pdf_for_text(pdf_content, text_metadata)
                
                if re.search("\.[a-z]+$", parsed_url):
                    text_file_name = f'{self.text_bucket_folder}/{parsed_url}'
                # Otherwise, add the inferred extension
                else:
                    text_file_name = f'{self.text_bucket_folder}/{parsed_url}{filetype}'
                
                text_file_name = Path(text_file_name).with_suffix('.txt').as_posix()
                
                # Upload text to text bucket
                text_blob = self.bucket.blob(text_file_name)
                
                # Add metadata about the text extraction
                text_blob.metadata = {
                    "original_file": new_file_name,
                    "needed_ocr": str(needed_ocr),
                    "case_title": text_metadata.get("case_title", ""),
                    "docket_page": text_metadata.get("docket_page", ""),
                    "date": text_metadata.get("date", "")
                }

                logger.info(f"Uploading text for amicus brief to {text_file_name}")
                
                # Upload the text content
                text_file = io.BytesIO(text_content.encode('utf-8'))
                text_blob.upload_from_file(text_file, content_type='text/plain', timeout=900)
                
                logger.info(f"Uploaded text for amicus brief to {text_file_name}")

        return item