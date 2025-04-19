import os
import json
import time
import logging
import tempfile
from typing import Dict, List, Any, Optional, Iterator, Union, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class BatchProcessor:
    """Utility class for processing requests with OpenAI's Batch API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the batch processor
        
        Args:
            api_key: OpenAI API key (if None, loads from environment variable)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
        self.client = OpenAI(api_key=self.api_key)
    
    def create_batch_file(
        self, 
        requests: List[Dict[str, Any]], 
        model: str,
        endpoint: str = "/v1/chat/completions"
    ) -> str:
        """
        Create and upload a batch file
        
        Args:
            requests: List of request dictionaries with 'id' and 'content' keys
            model: OpenAI model to use
            endpoint: API endpoint to use
            
        Returns:
            File ID of the uploaded batch file
        """
        # Create JSONL content
        batch_lines = []
        
        for req in requests:
            request_data = {
                "custom_id": req["id"],
                "method": "POST",
                "url": endpoint,
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": req.get("system_prompt", "")},
                        {"role": "user", "content": req["content"]}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": req.get("temperature", 0.0)
                }
            }
            batch_lines.append(json.dumps(request_data))
        
        jsonl_content = "\n".join(batch_lines)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.jsonl', delete=False) as temp_file:
            temp_file.write(jsonl_content)
            temp_file_path = temp_file.name
        
        try:
            # Upload file
            with open(temp_file_path, 'rb') as file:
                response = self.client.files.create(
                    file=file,
                    purpose="batch"
                )
            
            logger.info(f"Uploaded batch file with ID: {response.id}")
            return response.id
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    def run_batch(
        self,
        requests: List[Dict[str, Any]],
        model: str,
        system_prompt: str,
        wait: bool = True,
        polling_interval: int = 60,
        endpoint: str = "/v1/chat/completions"
    ) -> Dict[str, Any]:
        """
        Run a complete batch job
        
        Args:
            requests: List of request dictionaries with 'id' and 'content' keys
            model: OpenAI model to use
            system_prompt: System prompt to use for all requests
            wait: Whether to wait for batch completion
            polling_interval: Seconds between status checks
            endpoint: API endpoint to use
            
        Returns:
            Dictionary with batch information and results
        """
        # Add system prompt to all requests
        for req in requests:
            req["system_prompt"] = system_prompt
        
        # Upload batch file
        file_id = self.create_batch_file(requests, model, endpoint)
        
        # Create batch
        batch_response = self.client.batches.create(
            input_file_id=file_id,
            endpoint=endpoint,
            completion_window="24h"
        )
        batch_id = batch_response.id
        logger.info(f"Created batch with ID: {batch_id}")
        
        # Check status (with optional wait)
        if wait:
            final_batch = self.wait_for_completion(batch_id, polling_interval)
            
            # Download results if completed
            if final_batch["status"] == "completed":
                results = self.get_results(final_batch["output_file_id"])
                return {
                    "batch_info": final_batch,
                    "results": results,
                    "success": True
                }
            else:
                logger.warning(f"Batch did not complete. Final status: {final_batch['status']}")
                return {
                    "batch_info": final_batch,
                    "results": {},
                    "success": False
                }
        else:
            return {
                "batch_info": batch_response.dict(),
                "results": {},
                "batch_id": batch_id,
                "success": None  # None indicates pending
            }
    
    def wait_for_completion(self, batch_id: str, polling_interval: int = 60) -> Dict[str, Any]:
        """
        Wait for a batch to complete
        
        Args:
            batch_id: ID of the batch
            polling_interval: Seconds between status checks
            
        Returns:
            Final batch information dictionary
        """
        while True:
            response = self.client.batches.retrieve(batch_id)
            batch_info = response.dict()
            
            status = batch_info["status"]
            logger.info(f"Batch status: {status}")
            
            if status in ["completed", "failed", "expired", "cancelled"]:
                return batch_info
            
            # Display progress
            completed = batch_info["request_counts"]["completed"]
            total = batch_info["request_counts"]["total"]
            if total > 0:
                logger.info(f"Progress: {completed}/{total} requests completed ({completed/total*100:.1f}%)")
            
            logger.info(f"Waiting {polling_interval} seconds...")
            time.sleep(polling_interval)
    
    def get_results(self, output_file_id: str) -> Dict[str, Any]:
        """
        Get results from a completed batch
        
        Args:
            output_file_id: Output file ID
            
        Returns:
            Dictionary mapping request IDs to results
        """
        # Download file content
        file_response = self.client.files.content(output_file_id)
        content = file_response.text
        
        # Parse JSONL file
        results = {}
        for line in content.strip().split('\n'):
            result = json.loads(line)
            custom_id = result["custom_id"]
            
            if result.get("error"):
                logger.error(f"Error for {custom_id}: {result['error']}")
                results[custom_id] = {"error": result["error"], "success": False}
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
                            results[custom_id] = {
                                "data": content_json,
                                "success": True,
                                "usage": body.get("usage", {})
                            }
                        except json.JSONDecodeError:
                            logger.error(f"Could not decode JSON response for {custom_id}")
                            results[custom_id] = {
                                "error": "JSON decode error", 
                                "raw_content": content_str,
                                "success": False
                            }
        
        return results
