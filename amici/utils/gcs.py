import os
from dotenv import load_dotenv
from google.cloud.storage import Client

class GCSFetch:
    def __init__(self, bucketname):
        """
        Initialize a Google Cloud Storage client.
        This constructor creates a new Google Cloud Storage (GCS) client and sets up a connection to a specified bucket.
        It requires proper authentication credentials to be set in the environment.
        Parameters
        ----------
        bucketname : str
            The name of the GCS bucket to connect to.
        Raises
        ------
        ValueError
            If the Google application credentials environment variable is not set.
        Notes
        -----
        The constructor loads environment variables from "../../.env" and expects GOOGLE_APPLICATION_CREDENTIALS
        to be defined there. This environment variable should point to the Google Cloud service account 
        credentials file.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dotenv_path = os.path.normpath(os.path.join(script_dir, '..', '..', '.env'))
        load_dotenv(dotenv_path=dotenv_path)
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            raise ValueError(
                "Google auth credentials not set. Make sure to include in the .env file in /env directory.")
        self.__client = Client()
        self.__bucket_name = bucketname
        self.__bucket = self.__client.bucket(bucketname)

    def get_from_bucket(self, filename) -> (bytes, dict):
        """
        Retrieves a file's contents and metadata from a GCS bucket.
        Args:
            filename (str): The name/path of the file to retrieve from the bucket.
        Returns:
            tuple: A tuple containing:
                - bytes: The file contents as bytes.
                - dict: The file's metadata dictionary, which includes the original metadata 
                       and an additional 'blob_name' key with the filename value.
        Note:
            If the blob has no metadata, an empty dictionary will be returned as metadata.
        """

        blob = self.__bucket.get_blob(filename)
        content_bytes: bytes = blob.download_as_bytes()
        metadata: dict = blob.metadata
        if metadata is None:
            metadata = {}
        metadata['blob_name'] = filename
        return content_bytes, metadata

    def list_blobs(self, prefix) -> list:
        """
        List blobs in a GCS bucket with a specific prefix.
        This method returns an iterable of Google Cloud Storage blob objects that match the given prefix.
        Parameters
        ----------
        prefix : str
            The prefix to filter blobs by. Only blobs with names that start with this prefix will be returned.
        Returns
        -------
        list
            An iterable of blob objects matching the specified prefix.
        """

        blobs = self.__client.list_blobs(self.__bucket_name, prefix=prefix)
        return blobs

    def upload_to_bucket(self, filename, file, sidecar=None):
        """
        Uploads a file to a GCS bucket.
        Args:
            filename (str): The name/path of the file to upload to the bucket.
            file (io.BytesIO): The file content to upload.
            sidecar (dict, optional): A dictionary containing metadata to be associated with the uploaded file.
        Returns:
            None
        """

        blob = self.__bucket.blob(filename)
        blob.upload_from_file(file)
        if sidecar:
            blob.metadata = sidecar
            blob.patch()
        return blob
