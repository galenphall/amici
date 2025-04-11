import os
import io
import re

import itemadapter
import scrapy
import uuid

from google.cloud.storage import Client
from itemadapter import ItemAdapter
from google.cloud import storage
from google.api_core.exceptions import Forbidden as GoogleForbiddenException
from dotenv import load_dotenv


class GCSPipeline:
    """
    A pipeline class for uploading files to GCS. Every time an item is scraped, it's run through this pipeline. Note that
    the pipeline must be enabled in settings.py, which it should get by default.
    """

    def __init__(self):
        # GCS boilerplate code
        # Requires the GOOGLE_APPLICATION_CREDENTIALS environment variable

        load_dotenv("../../env/.env")
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "Google auth credentials not set. Make sure to include in the .env file in /env directory.")

        storage_client: storage.Client = storage.Client()
        self.bucket = storage_client.bucket("interest_groups_raw_documents_2025")

        # if not os.path.isdir('temp'):
        #     os.mkdir('temp')

    def process_item(self, item: itemadapter.ItemAdapter.item, spider: scrapy.Spider) -> itemadapter.ItemAdapter.item:
        """
        Uploads the file to the raw documents GCS bucket.
        """
        # Transform the item into a dictionary
        adapted = ItemAdapter(item)
        state = adapted.get('state')[0]

        if adapted.get('rawcontent'):

            # Get the file extension
            filetype: str = adapted.get("filetype")[0]
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
                new_file_name = f'{state}/{parsed_url}'

            # Otherwise, add the inferred extension
            else:
                new_file_name = f'{state}/{parsed_url}{filetype}'

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

            blob.upload_from_file(file, timeout=900)

        return item
