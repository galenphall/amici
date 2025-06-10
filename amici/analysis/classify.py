import os
import json
import time
import argparse
from openai import OpenAI
from tqdm import tqdm
import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import datetime

from dotenv import load_dotenv

import os.path

# Import Anthropic client
from anthropic import Anthropic

# Ensure path to data is relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))
from utils.normalizers import normalize_interest_group_name
from utils.classes import INDUSTRIES, SECTORS, SECTORS_TO_INDUSTRIES, INDUSTRIES_TO_SECTORS

# Load environment variables from .env file in parent directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(script_dir)), '.env'))

# Initialize clients
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def normalize_name(x):
    return normalize_interest_group_name(x).upper()

def make_schema(interest_groups, allowed_categories):
    """
    Create a JSON schema for the expected output format.
    
    Args:
        interest_groups: List of interest group names
        allowed_categories: List of allowed categories for the enum
    
    Returns:
        dict: JSON schema for the expected output
    """
    schema = {
        "type": "object",
        "properties": {
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "interest_group": {
                            "type": "string",
                            "description": "The name of the interest group being categorized",
                            "enum": interest_groups
                        },
                        "industry": {
                            "type": "string",
                            "description": "The industry category that best matches the interest group",
                            "enum": allowed_categories
                        }
                    },
                    "required": ["interest_group", "industry"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["predictions"],
        "additionalProperties": False
    }
    
    return schema

def prepare_batch_data(interest_group_batches, model_id, allowed_categories, output_file="batch_input.jsonl", use_claude=False):
    """
    Prepare evaluation data for batch processing with the specified schema.
    
    Args:
        interest_groups: List of interest_groups batches
        model_id: ID of the fine-tuned model to use
        allowed_categories: List of allowed categories for the enum
        output_file: Path to save the batch input file
        use_claude: Whether to use Claude API instead of OpenAI
    """
    batch_requests = []
    
    for i, interest_groups in enumerate(interest_group_batches):
        # Format the system message to include allowed categories
        categories_str = "; ".join(allowed_categories)

        prompt = "Categorize these interest groups:\n" + "  ".join([f'({i+1}) {g}' for i, g in enumerate(interest_groups)])
        
        system_message = f"You are an expert at categorizing interest groups into industries. The allowed industries are: {categories_str}. "
        
        if use_claude:
            # Prepare for Claude API with tool use
            request = {
                "custom_id": f"batch-request-{i}",
                "interest_groups": interest_groups,
                "allowed_categories": allowed_categories,
                "system_message": system_message,
                "prompt": prompt
            }
        else:
            # Original OpenAI format
            request = {
                "custom_id": f"batch-request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "categorization", "schema": make_schema(interest_groups, allowed_categories), 'strict': True}
                    },
                    "temperature": 0.0  # Deterministic output for evaluation
                }
            }
        batch_requests.append(request)
    
    # Write to JSONL file
    with open(output_file, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created batch input file with {len(batch_requests)} requests: {output_file}")
    return output_file

def create_claude_tool(interest_groups, allowed_categories):
    """
    Create a tool definition for Claude API to categorize interest groups.
    """
    return {
        "name": "categorize_interest_groups",
        "description": "Categorize interest groups into industry classifications",
        "input_schema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "interest_group": {
                                "type": "string",
                                "description": "The name of the interest group being categorized",
                                "enum": interest_groups
                            },
                            "industry": {
                                "type": "string",
                                "description": "The industry category that best matches the interest group",
                                "enum": allowed_categories
                            }
                        },
                        "required": ["interest_group", "industry"]
                    }
                }
            },
            "required": ["predictions"]
        }
    }

def prepare_claude_batch_requests(test_items, allowed_categories):
    """
    Prepare requests for Claude's Message Batches API.
    
    Args:
        test_items: List of (prompt, interest_groups, expected) tuples
        allowed_categories: List of allowed categories
        
    Returns:
        list: List of request dictionaries for Claude batch API
    """
    requests = []
    
    for i, (prompt, interest_groups, _) in enumerate(test_items):
        categories_str = ", ".join(allowed_categories)
        
        if prompt is None:
            prompt = "Categorize these interest groups:\n" + "\n".join(interest_groups)
        
        system_message = (
            """You are an expert political finance analyst specializing in categorizing lobbying entities into standardized industry classifications. Your task is to analyze each interest group name and assign it to the most appropriate industry category according to the National Institute for Money in Politics classification system.

            INSTRUCTIONS:
            1. Review each interest group name carefully
            2. Consider the likely business activities, policy focus, or stakeholders represented
            3. Select the SINGLE most appropriate industry category from the list below
            4. Provide your classification in the tool call with the exact format specified
            5. If truly uncertain, classify as "Unknown/Other" rather than guessing

            CLASSIFICATION GUIDELINES:
            - Focus on the primary function of the organization, not secondary activities
            - Consider parent companies or industry affiliations when relevant
            - Be consistent with similar organizations you've classified previously
            - Trade associations should be classified by the industry they represent, unless they are general business trade associations

            CATEGORIES
            """
            f"The allowed industries are: {categories_str}. "
        )
        
        # Create tool for this batch
        tool = create_claude_tool(interest_groups, allowed_categories)
        
        request = {
            "custom_id": f"batch-request-{i}",
            "params": {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4096,
                "temperature": 0.0,
                "system": system_message,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": "categorize_interest_groups"}
            }
        }
        requests.append(request)
    
    return requests

def submit_claude_batch(requests, model_id="claude-3-5-sonnet-20241022"):
    """
    Submit a batch job to Claude's Message Batches API.
    
    Args:
        requests: List of request dictionaries
        model_id: Claude model to use
        
    Returns:
        MessageBatch object or None if error
    """
    try:
        # Update model in all requests
        for req in requests:
            req["params"]["model"] = model_id
        
        # Create batch
        message_batch = claude_client.messages.batches.create(
            requests=requests
        )
        
        print(f"Claude batch submitted successfully!")
        print(f"Batch ID: {message_batch.id}")
        print(f"Status: {message_batch.processing_status}")
        print(f"Created at: {message_batch.created_at}")
        
        return message_batch
        
    except Exception as e:
        print(f"Error submitting Claude batch: {e}")
        return None

def monitor_claude_batch(batch_id):
    """
    Monitor the status of a Claude batch until completion.
    
    Args:
        batch_id: ID of the batch
        
    Returns:
        MessageBatch object when complete, None if error
    """
    try:
        while True:
            batch = claude_client.messages.batches.retrieve(batch_id)
            
            print(f"Batch status: {batch.processing_status}")
            print(f"Completed: {batch.request_counts.succeeded}/{batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored}")
            
            if batch.processing_status == "ended":
                print("Batch processing complete!")
                return batch
            
            # Wait before checking again
            time.sleep(60)
            
    except Exception as e:
        print(f"Error monitoring Claude batch: {e}")
        return None

def download_claude_batch_results(batch_id, save_path="claude_batch_results.jsonl"):
    """
    Download and process results from a completed Claude batch.
    
    Args:
        batch_id: ID of the completed batch
        save_path: Path to save results
        
    Returns:
        list: Parsed results
    """
    try:
        results = []
        
        # Stream results
        for result in claude_client.messages.batches.results(batch_id):
            results.append(result)
        
        # Save to file in OpenAI-compatible format for consistency
        with open(save_path, 'w') as f:
            for result in results:
                formatted_result = {
                    "custom_id": result.custom_id,
                    "response": {
                        "body": {
                            "choices": [{
                                "message": {
                                    "content": ""
                                }
                            }]
                        }
                    }
                }
                
                if result.result.type == "succeeded":
                    # Extract tool use content
                    for content in result.result.message.content:
                        if hasattr(content, 'input'):  # Tool use block
                            formatted_result["response"]["body"]["choices"][0]["message"]["content"] = json.dumps(content.input)
                            break
                elif result.result.type == "errored":
                    formatted_result["error"] = {
                        "type": result.result.error.type,
                        "message": str(result.result.error)
                    }
                
                f.write(json.dumps(formatted_result) + '\n')
        
        print(f"Downloaded {len(results)} results to {save_path}")
        return results
        
    except Exception as e:
        print(f"Error downloading Claude batch results: {e}")
        return None

def submit_openai_batch(client, batch_file):
    """
    Submit a batch job to OpenAI and return the job ID.
    
    Returns:
        tuple: (batch_id, file_id) - IDs of the batch job and uploaded file
    """
    try:
        # Upload the batch file first
        with open(batch_file, 'rb') as f:
            batch_file_obj = client.files.create(
                file=f,
                purpose="batch"
            )
        
        file_id = batch_file_obj.id
        print(f"Batch file uploaded with ID: {file_id}")
        
        # Create batch job using the file ID
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        batch_id = batch.id
        print(f"Batch job submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"Status: {batch.status}")
        print(f"Created at: {batch.created_at}")
        
        return batch_id, file_id
    except Exception as e:
        print(f"Error submitting batch job: {e}")
        return None, None

def monitor_openai_batch(client, batch_id):
    """
    Monitor the status of a batch job until completion.
    
    Args:
        client: OpenAI client
        batch_id: ID of the batch job
    
    Returns:
        str: Output file ID if job completed successfully, None otherwise
    """
    try:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        
        print(f"Starting monitoring of batch {batch_id}")
        print(f"Initial status: {status}")
        
        while status not in ["completed", "failed", "expired", "cancelled"]:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            print(f"Batch status: {status}, Completed: {batch.request_counts.completed}/{batch.request_counts.total}")
            
            if status == "completed":
                print("Batch processing complete!")
                return batch.output_file_id
            elif status in ["failed", "expired", "cancelled"]:
                print(f"Batch processing ended with status: {status}")
                return None
            
            # Wait before checking status again
            time.sleep(60)
            
    except Exception as e:
        print(f"Error monitoring batch job: {e}")
        return None

def download_openai_batch_results(client, output_file_id, save_path="batch_results.jsonl"):
    """
    Download the results of a completed batch job from OpenAI.
    
    Args:
        client: OpenAI client
        output_file_id: ID of the output file from the batch job
        save_path: Path to save the downloaded results
        
    Returns:
        list: The parsed batch results
    """
    try:
        # Download the content
        response = client.files.content(output_file_id)
        content = response.read().decode('utf-8')
        
        # Save to file
        with open(save_path, 'w') as f:
            f.write(content)
        
        print(f"Downloaded batch results to {save_path}")
        
        # Parse the results
        results = []
        with open(save_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        return results
    
    except Exception as e:
        print(f"Error downloading batch results: {e}")
        return None

def main_run(input_file="../data/unique_amicus_merged_names.csv", name_col="merged_name", batch_size=50, use_claude=False, claude_model="claude-3-5-sonnet-20241022"):
    """
    Main entry point for running the batch job.
    
    Args:
        input_file: Path to the input CSV file containing names
        name_col: Name of the column in the CSV file containing names
        use_claude: Whether to use Claude API instead of OpenAI
        claude_model: Claude model to use if use_claude is True
    """
    # Load input data
    df = pd.read_csv(input_file)
    
    # Extract names from the specified column
    names = df[name_col].tolist()
    names = [normalize_name(name) for name in names]
    names = list(set(names))  # Remove duplicates
    print(f"Loaded {len(names)} unique names from {input_file}")
    
    # Prepare batch data
    batched_names = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    print(f"Prepared {len(batched_names)} items for batch processing")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if use_claude:
        print(f"Using Claude Message Batches API with model {claude_model}...")
        
        # Prepare requests for Claude batch API
        requests = prepare_claude_batch_requests(test_items, INDUSTRIES)
        
        # Submit batch
        batch = submit_claude_batch(requests, model_id=claude_model)
        
        if batch:
            print(f"\nBatch submitted! You can monitor it at:")
            print(f"https://console.anthropic.com/settings/workspaces/default/batches/{batch.id}")
            print(f"\nOr continue monitoring here...")
            
            # Monitor until complete
            completed_batch = monitor_claude_batch(batch.id)
            
            if completed_batch:
                # Download results
                output_file = f"claude_batch_results_{timestamp}.jsonl"
                results = download_claude_batch_results(batch.id, output_file)
                
                if results:
                    print(f"\nBatch processing statistics:")
                    print(f"Total requests: {len(requests)}")
                    print(f"Succeeded: {completed_batch.request_counts.succeeded}")
                    print(f"Errored: {completed_batch.request_counts.errored}")
                    print(f"Canceled: {completed_batch.request_counts.canceled}")
                    print(f"Expired: {completed_batch.request_counts.expired}")
        
    else:
        print("Using OpenAI API...")
        batch_data = prepare_batch_data(
            batched_names, 
            model_id="ft:gpt-4.1-mini-2025-04-14:personal:igclassify-v2:BXRi9hNg",
            allowed_categories=INDUSTRIES, 
            output_file=f"batch_input_{timestamp}.jsonl"
        )
        print(f"Batch data prepared and saved to {batch_data}")

        cont = input("Continue? [Y/n]: ")
        if cont.lower() == 'y':
        
            # Submit batch job
            batch_id, output_file_id = submit_openai_batch(openai_client, batch_data)
            
            if batch_id:
                print(f"Batch job submitted successfully! Batch ID: {batch_id}")
                print(f"Output file ID: {output_file_id}")
                print("Use this ID to monitor or retrieve results.")
        else:
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process interest groups using OpenAI or Claude API')
    parser.add_argument('--use-claude', action='store_true', help='Use Claude API instead of OpenAI')
    parser.add_argument('--claude-model', default='claude-3-5-sonnet-20241022', help='Claude model to use')
    parser.add_argument('--input-file', default='../data/unique_amicus_merged_names.csv', help='Input CSV file')
    parser.add_argument('--name-col', default='merged_name', help='Column name containing names')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    parser.add_argument('--output-id', type=str)
    
    args = parser.parse_args()
    
    if 'output_id' in args and args.output_id is not None:
        results = download_openai_batch_results(openai_client, args.output_id)
        results_map = []

        for result in results:
            # Check if the result was successful
            if 'error' not in result:
                try:
                    # Parse the content which should be JSON from the tool call
                    content = result['response']['body']['choices'][0]['message']['content']
                    parsed_content = json.loads(content)
                    
                    # Extract predictions
                    if 'predictions' in parsed_content:
                        for prediction in parsed_content['predictions']:
                            if isinstance(prediction, dict):
                                results_map.append({
                                    'interest_group': prediction['interest_group'].lower(),
                                    'industry': prediction['industry']
                                })
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error parsing result for {result.get('custom_id', 'unknown')}: {e}")
            else:
                print(f"Error in batch result {result.get('custom_id', 'unknown')}: {result['error']}")

        results_df = pd.DataFrame(results_map)
        output_csv = f"openai_classifications.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"Saved {len(results_df)} classifications to {output_csv}")
    else:
        main_run(
            input_file=args.input_file,
            name_col=args.name_col,
            batch_size=args.batch_size,
            use_claude=args.use_claude,
            claude_model=args.claude_model
        )

    # timestamp = '2025-06-10 00:57:27.409306+00:00'
    # batch_id = 'msgbatch_01G8jFcQkRJMJFJar75bv299'
    # batch = completed_batch = claude_client.messages.batches.retrieve(batch_id)
    # output_file = f"classification/claude_batch_results_{timestamp}.jsonl"
    # results = download_claude_batch_results(batch.id, output_file)
    
    # if results:
    #     print(f"\nBatch processing statistics:")
    #     print(f"Succeeded: {completed_batch.request_counts.succeeded}")
    #     print(f"Errored: {completed_batch.request_counts.errored}")
    #     print(f"Canceled: {completed_batch.request_counts.canceled}")
    #     print(f"Expired: {completed_batch.request_counts.expired}")

    # results_map = []
    # # extract interest group to industry mapping
    # # Load the results file
    # with open(output_file, 'r') as f:
    #     for line in f:
    #         result = json.loads(line)
            
    #         # Check if the result was successful
    #         if 'error' not in result:
    #             try:
    #                 # Parse the content which should be JSON from the tool call
    #                 content = result['response']['body']['choices'][0]['message']['content']
    #                 parsed_content = json.loads(content)
                    
    #                 # Extract predictions
    #                 if 'predictions' in parsed_content:
    #                     for prediction in parsed_content['predictions']:
    #                         if isinstance(prediction, dict):
    #                             results_map.append({
    #                                 'interest_group': prediction['interest_group'],
    #                                 'industry': prediction['industry']
    #                             })
    #             except (json.JSONDecodeError, KeyError, IndexError) as e:
    #                 print(f"Error parsing result for {result.get('custom_id', 'unknown')}: {e}")
    #         else:
    #             print(f"Error in batch result {result.get('custom_id', 'unknown')}: {result['error']}")

    # print(f"Extracted {len(results_map)} interest group classifications")

    # # Convert to DataFrame and save to CSV
    # results_df = pd.DataFrame(results_map)
    # output_csv = f"claude_classifications.csv"
    # results_df.to_csv(output_csv, index=False)
    # print(f"Saved {len(results_df)} classifications to {output_csv}")