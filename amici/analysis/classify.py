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

# Ensure path to data is relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))
from utils.normalizers import normalize_interest_group_name
from utils.classes import INDUSTRIES, SECTORS, SECTORS_TO_INDUSTRIES, INDUSTRIES_TO_SECTORS

# Load environment variables from .env file in parent directory
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(script_dir)), '.env'))

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def normalize_name(x):
    return normalize_interest_group_name(x).upper()

def load_test_data(test_file_path):
    """
    Load the test data from a JSONL file.
    """
    test_data = []
    with open(test_file_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Loaded {len(test_data)} test examples from {test_file_path}")
    return test_data

def load_categories(categories_file_path):
    """
    Load the allowed categories from a file with one category per line.
    """
    categories = []
    with open(categories_file_path, 'r') as f:
        for line in f:
            category = line.strip()
            if category:
                categories.append(category)
    
    print(f"Loaded {len(categories)} allowed categories from {categories_file_path}")
    return categories

def extract_test_prompts_and_expected(test_data):
    """
    Extract user prompts, interest group names, and expected outputs from the test data.
    Returns a list of tuples (prompt, interest_groups, expected_output)
    """
    test_items = []
    for example in test_data:
        # Find user message
        user_prompt = None
        for message in example["messages"]:
            if message["role"] == "user":
                user_prompt = message["content"]
                break
        
        # Extract interest group names from the prompt
        # Assuming format: "Categorize these interest groups:\n" + "\n".join(batch["interest_groups"])
        interest_groups = []
        if user_prompt:
            lines = user_prompt.split("\n")
            if len(lines) > 1:
                interest_groups = [line for line in lines[1:] if line.strip()]
        
        # Get expected output (assistant message)
        expected_output = json.loads(example["messages"][-1]["content"])
        
        if user_prompt and interest_groups:
            test_items.append((user_prompt, interest_groups, expected_output))
    
    return test_items

def extract_categories_from_test_data(test_data):
    """
    Extract all unique industry categories from the test data to use as the enum.
    """
    categories = set()
    
    for example in test_data:
        try:
            # Get assistant message (the last message)
            assistant_msg = example["messages"][-1]["content"]
            expected = json.loads(assistant_msg)
            
            # Extract all industry categories
            for prediction in expected["predictions"]:
                if "industry" in prediction:
                    categories.add(prediction["industry"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error extracting categories: {e}")
    
    return sorted(list(categories))

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

def prepare_batch_data(test_items, model_id, allowed_categories, output_file="batch_input.jsonl"):
    """
    Prepare evaluation data for batch processing with the specified schema.
    
    Args:
        test_items: List of (prompt, interest_groups, expected) tuples
        model_id: ID of the fine-tuned model to use
        allowed_categories: List of allowed categories for the enum
        output_file: Path to save the batch input file
    """
    batch_requests = []
    
    for i, (prompt, interest_groups, _) in enumerate(test_items):
        # Format the system message to include allowed categories
        categories_str = ", ".join(allowed_categories)

        if prompt is None:
            prompt = "Categorize these interest groups:\n" + "\n".join(interest_groups)
        
        system_message = (
            """
            You are an expert political finance analyst specializing in categorizing lobbying entities into standardized industry classifications. Your task is to analyze each interest group name and assign it to the most appropriate industry category according to the National Institute for Money in Politics classification system.

            INSTRUCTIONS:
            1. Review each interest group name carefully
            2. Consider the likely business activities, policy focus, or stakeholders represented
            3. Select the SINGLE most appropriate industry category from the list below
            4. Provide your classification in a consistent format: "Interest Group Name: [CATEGORY]"
            5. If truly uncertain, classify as "Unknown/Other" rather than guessing

            CLASSIFICATION GUIDELINES:
            - Focus on the primary function of the organization, not secondary activities
            - Consider parent companies or industry affiliations when relevant
            - Be consistent with similar organizations you've classified previously
            - Trade associations should be classified by the industry they represent, unless they are general business trade associations
            - Follow the provided output schema

            SCHEMA
            {"predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "interest_group": {"type": "string"},
                            "industry": {"type": "string"}
                        }
                    }
                }
            }

            CATEGORIES
            """
            f"The allowed industries are: {categories_str}. "
        )
        
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

def submit_batch_job(client, batch_file):
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

def monitor_batch_job(client, batch_id):
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

def download_batch_results(client, output_file_id, save_path="batch_results.jsonl"):
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

def match_predictions_with_expected(batch_results, test_items):
    """
    Match the batch results with the expected outputs to create evaluation pairs.
    
    Args:
        batch_results: List of batch response objects from OpenAI
        test_items: List of (prompt, interest_groups, expected) tuples
        
    Returns:
        list: List of (interest_group, predicted_industry, expected_industry, predicted_sector, expected_sector) tuples
    """
    evaluation_pairs = []
    errors = []
    
    # Create a mapping from custom_id to expected output
    expected_map = {}
    for i, (_, _, expected) in enumerate(test_items):
        expected_map[f"batch-request-{i}"] = expected
    
    for result in batch_results:
        # Get the custom_id to match with the expected output
        custom_id = result.get("custom_id")
        
        if custom_id in expected_map:
            # Check if the result has a response field
            if "response" in result:
                response_data = result["response"]
                
                # Handle the case where status is unknown but body contains valid JSON
                if isinstance(response_data, dict) and "body" in response_data:
                    try:
                        # The body could be a string or already parsed JSON
                        body_content = response_data["body"]
                        if isinstance(body_content, str):
                            try:
                                body_content = json.loads(body_content)
                            except json.JSONDecodeError:
                                # Some bodies might be already parsed JSON objects
                                pass
                        
                        # Extract the content from choices
                        if isinstance(body_content, dict) and "choices" in body_content:
                            choices = body_content["choices"]
                            if choices and isinstance(choices, list) and len(choices) > 0:
                                message = choices[0].get("message", {})
                                content = message.get("content", "")
                                
                                # Parse the content as JSON
                                if isinstance(content, str):
                                    predictions = json.loads(content)
                                else:
                                    predictions = content
                                
                                # Get the expected output
                                expected = expected_map[custom_id]
                                
                                # Match predictions with expected values
                                if "predictions" in predictions and "predictions" in expected:
                                    pred_dict = {p["interest_group"]: p["industry"] for p in predictions["predictions"]}
                                    exp_dict = {e["interest_group"]: e["industry"] for e in expected["predictions"]}
                                    
                                    # Create evaluation pairs with both industry and sector mappings
                                    for group in pred_dict:
                                        if group in exp_dict:
                                            pred_industry = pred_dict[group]
                                            exp_industry = exp_dict[group]
                                            
                                            # Map industries to sectors using the imported mappings
                                            pred_sector = INDUSTRIES_TO_SECTORS.get(pred_industry, "Unknown")
                                            exp_sector = INDUSTRIES_TO_SECTORS.get(exp_industry, "Unknown")
                                            
                                            evaluation_pairs.append((group, pred_industry, exp_industry, pred_sector, exp_sector))
                                        else:
                                            errors.append(f"Warning: Group '{group}' in predictions not found in expected output")
                                else:
                                    errors.append(f"Missing 'predictions' in result or expected output for {custom_id}")
                        else:
                            errors.append(f"No choices found in response body for {custom_id}")
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        errors.append(f"Error processing result {custom_id}: {e}")
                # Handle the case where status is a number (normal response)
                elif isinstance(response_data, dict) and "status" in response_data and response_data["status"] == 200:
                    # Original code for handling successful responses
                    try:
                        response_body = json.loads(response_data["body"])
                        predictions = response_body["choices"][0]["message"]["content"]
                        
                        # If content is a string, parse it as JSON
                        if isinstance(predictions, str):
                            predictions = json.loads(predictions)
                        
                        # Get the expected output
                        expected = expected_map[custom_id]
                        
                        # Match predictions with expected values
                        if "predictions" in predictions and "predictions" in expected:
                            pred_dict = {p["interest_group"]: p["industry"] for p in predictions["predictions"]}
                            exp_dict = {e["interest_group"]: e["industry"] for e in expected["predictions"]}
                            
                            # Create evaluation pairs with both industry and sector mappings
                            for group in pred_dict:
                                if group in exp_dict:
                                    pred_industry = pred_dict[group]
                                    exp_industry = exp_dict[group]
                                    
                                    # Map industries to sectors using the imported mappings
                                    pred_sector = INDUSTRIES_TO_SECTORS.get(pred_industry, "Unknown")
                                    exp_sector = INDUSTRIES_TO_SECTORS.get(exp_industry, "Unknown")
                                    
                                    evaluation_pairs.append((group, pred_industry, exp_industry, pred_sector, exp_sector))
                                else:
                                    errors.append(f"Warning: Group '{group}' in predictions not found in expected output")
                        else:
                            errors.append(f"Missing 'predictions' in result or expected output for {custom_id}")
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        errors.append(f"Error processing result {custom_id}: {e}")
                else:
                    # Log error if response was not successful
                    status = response_data.get("status", "unknown")
                    error = response_data.get("body", "No error details")
                    errors.append(f"Request {custom_id} failed with status {status}: {error}")
            else:
                errors.append(f"No response field found for {custom_id}")
        else:
            errors.append(f"No matching expected output for custom_id: {custom_id}")
    
    if errors:
        print(f"Encountered {len(errors)} errors during matching:")
        for i, error in enumerate(errors[:5]):  # Print first 5 errors
            print(f"  {i+1}. {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    print(f"Created {len(evaluation_pairs)} evaluation pairs")
    return evaluation_pairs

def calculate_metrics(evaluation_pairs):
    """
    Calculate precision, recall, accuracy, and F1 score for the predictions at both industry and sector levels.
    
    Args:
        evaluation_pairs: List of (group, pred_industry, exp_industry, pred_sector, exp_sector) tuples
        
    Returns:
        dict: Dictionary containing the calculated metrics for both industry and sector levels
    """
    if not evaluation_pairs:
        empty_metrics = {
            "accuracy": 0,
            "macro_precision": 0,
            "macro_recall": 0,
            "macro_f1": 0,
            "per_category": {}
        }
        return {
            "industry": empty_metrics,
            "sector": empty_metrics
        }
    
    # Extract predictions and expected labels for both industry and sector
    groups = [p[0] for p in evaluation_pairs]
    
    # Industry level
    y_pred_industry = [p[1] for p in evaluation_pairs]
    y_true_industry = [p[2] for p in evaluation_pairs]
    
    # Sector level
    y_pred_sector = [p[3] for p in evaluation_pairs]
    y_true_sector = [p[4] for p in evaluation_pairs]
    
    # Calculate metrics for both levels
    metrics = {}
    
    # Function to calculate metrics for a specific level (industry or sector)
    def calc_level_metrics(y_true, y_pred, level_name):
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y_true, y_pred)
        
        # Get unique labels to calculate per-category metrics
        unique_labels = sorted(set(y_true + y_pred))
        
        # Calculate per-category metrics
        per_category = {}
        for label in unique_labels:
            # Create binary arrays for this category
            y_true_bin = [1 if y == label else 0 for y in y_true]
            y_pred_bin = [1 if y == label else 0 for y in y_pred]
            
            # Count occurrences in true labels
            true_count = sum(y_true_bin)
            
            # Skip categories with no true samples
            if true_count == 0:
                continue
                
            # Calculate metrics for this category
            precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            
            per_category[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": true_count
            }
        
        # Calculate macro-averaged metrics
        macro_precision = np.mean([per_category[label]["precision"] for label in per_category])
        macro_recall = np.mean([per_category[label]["recall"] for label in per_category])
        macro_f1 = np.mean([per_category[label]["f1"] for label in per_category])
        
        # Calculate weighted-averaged metrics
        total_support = sum([per_category[label]["support"] for label in per_category])
        weighted_precision = sum([per_category[label]["precision"] * per_category[label]["support"] for label in per_category]) / total_support
        weighted_recall = sum([per_category[label]["recall"] * per_category[label]["support"] for label in per_category]) / total_support
        weighted_f1 = sum([per_category[label]["f1"] * per_category[label]["support"] for label in per_category]) / total_support
        
        return {
            "accuracy": overall_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "per_category": per_category
        }
    
    # Calculate metrics for industry level
    metrics["industry"] = calc_level_metrics(y_true_industry, y_pred_industry, "industry")
    
    # Calculate metrics for sector level
    metrics["sector"] = calc_level_metrics(y_true_sector, y_pred_sector, "sector")
    
    return metrics

def generate_confusion_matrix(evaluation_pairs, output_dir="evaluation_outputs"):
    """
    Generate and save confusion matrix visualizations for both industry and sector levels.
    
    Args:
        evaluation_pairs: List of (group, pred_industry, exp_industry, pred_sector, exp_sector) tuples
        output_dir: Directory to save the confusion matrix images
        
    Returns:
        dict: Dictionary containing paths to the generated confusion matrix images
    """
    if not evaluation_pairs:
        print("No evaluation pairs to generate confusion matrices")
        return {"industry": None, "sector": None}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {}
    
    # Generate confusion matrix for industry level
    y_pred_industry = [p[1] for p in evaluation_pairs]
    y_true_industry = [p[2] for p in evaluation_pairs]
    
    # Get unique industry labels
    unique_industry_labels = sorted(set(y_true_industry + y_pred_industry))
    
    # Create industry confusion matrix
    cm_industry = confusion_matrix(y_true_industry, y_pred_industry, labels=unique_industry_labels)
    
    # Plot industry confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_industry, annot=True, fmt="d", cmap="Blues", 
                xticklabels=unique_industry_labels, yticklabels=unique_industry_labels)
    plt.xlabel('Predicted Industry')
    plt.ylabel('True Industry')
    plt.title('Industry-Level Confusion Matrix')
    plt.tight_layout()
    
    # Save industry confusion matrix
    industry_cm_path = os.path.join(output_dir, "industry_confusion_matrix.png")
    plt.savefig(industry_cm_path, dpi=100)
    plt.close()
    print(f"Saved industry confusion matrix to {industry_cm_path}")
    output_files["industry"] = industry_cm_path
    
    # Generate confusion matrix for sector level
    y_pred_sector = [p[3] for p in evaluation_pairs]
    y_true_sector = [p[4] for p in evaluation_pairs]
    
    # Get unique sector labels
    unique_sector_labels = sorted(set(y_true_sector + y_pred_sector))
    
    # Create sector confusion matrix
    cm_sector = confusion_matrix(y_true_sector, y_pred_sector, labels=unique_sector_labels)
    
    # Plot sector confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_sector, annot=True, fmt="d", cmap="Blues", 
                xticklabels=unique_sector_labels, yticklabels=unique_sector_labels)
    plt.xlabel('Predicted Sector')
    plt.ylabel('True Sector')
    plt.title('Sector-Level Confusion Matrix')
    plt.tight_layout()
    
    # Save sector confusion matrix
    sector_cm_path = os.path.join(output_dir, "sector_confusion_matrix.png")
    plt.savefig(sector_cm_path, dpi=100)
    plt.close()
    print(f"Saved sector confusion matrix to {sector_cm_path}")
    output_files["sector"] = sector_cm_path
    
    return output_files

def analyze_errors(evaluation_pairs):
    """
    Analyze common error patterns in the model predictions at both industry and sector levels.
    
    Args:
        evaluation_pairs: List of (group, pred_industry, exp_industry, pred_sector, exp_sector) tuples
        
    Returns:
        dict: Dictionary containing error analysis for both industry and sector levels
    """
    if not evaluation_pairs:
        empty_analysis = {"error_count": 0, "error_rate": 0, "common_errors": []}
        return {
            "industry": empty_analysis,
            "sector": empty_analysis
        }
    
    # Count total predictions
    total = len(evaluation_pairs)
    
    # Function to analyze errors for a specific level (industry or sector)
    def analyze_level_errors(pred_idx, exp_idx):
        # Find errors at this level
        errors = [(p[0], p[pred_idx], p[exp_idx]) for p in evaluation_pairs if p[pred_idx] != p[exp_idx]]
        error_count = len(errors)
        error_rate = error_count / total if total > 0 else 0
        
        # Analyze common misclassifications
        misclassifications = defaultdict(int)
        for _, pred, exp in errors:
            misclassifications[(exp, pred)] += 1
        
        # Sort by frequency
        common_errors = [
            {
                "true_category": true,
                "predicted_category": pred,
                "count": count,
                "percentage": (count / error_count) * 100 if error_count > 0 else 0
            }
            for (true, pred), count in sorted(misclassifications.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Group errors by expected category
        category_errors = defaultdict(int)
        category_counts = defaultdict(int)
        
        for p in evaluation_pairs:
            category_counts[p[exp_idx]] += 1
            if p[pred_idx] != p[exp_idx]:
                category_errors[p[exp_idx]] += 1
        
        # Calculate error rate by category
        error_by_category = {
            category: {
                "error_count": errors,
                "total": category_counts[category],
                "error_rate": errors / category_counts[category] if category_counts[category] > 0 else 0
            }
            for category, errors in category_errors.items()
        }
        
        return {
            "error_count": error_count,
            "total_predictions": total,
            "error_rate": error_rate,
            "common_errors": common_errors[:10],  # Top 10 most common errors
            "error_by_category": error_by_category
        }
    
    # Analyze errors at industry level (indices 1 and 2)
    industry_analysis = analyze_level_errors(1, 2)
    
    # Analyze errors at sector level (indices 3 and 4)
    sector_analysis = analyze_level_errors(3, 4)
    
    return {
        "industry": industry_analysis,
        "sector": sector_analysis
    }

def save_prediction_results(evaluation_pairs, metrics, error_analysis, cm_paths, output_file="prediction_results.json"):
    """
    Save the prediction results, metrics, and error analysis to a JSON file.
    
    Args:
        evaluation_pairs: List of (group, pred_industry, exp_industry, pred_sector, exp_sector) tuples
        metrics: Dictionary of calculated metrics for industry and sector
        error_analysis: Dictionary of error analysis results for industry and sector
        cm_paths: Dictionary of paths to confusion matrix images
        output_file: Path to save the results
    """
    # Format evaluation pairs for output
    formatted_pairs = [
        {
            "interest_group": group,
            "industry": {
                "predicted": pred_industry,
                "expected": exp_industry,
                "correct": pred_industry == exp_industry
            },
            "sector": {
                "predicted": pred_sector,
                "expected": exp_sector,
                "correct": pred_sector == exp_sector
            }
        }
        for group, pred_industry, exp_industry, pred_sector, exp_sector in evaluation_pairs
    ]
    
    # Create results object
    results = {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "confusion_matrix_paths": cm_paths,
        "predictions": formatted_pairs
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved prediction results to {output_file}")
    return output_file

def print_metrics_summary(metrics, error_analysis):
    """
    Print a summary of the evaluation metrics to the console for both industry and sector levels.
    
    Args:
        metrics: Dictionary of calculated metrics for industry and sector
        error_analysis: Dictionary of error analysis results for industry and sector
    """
    print("\n" + "="*80)
    print(" "*30 + "EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    # Print industry-level metrics
    print("\n" + "="*40)
    print(" "*10 + "INDUSTRY-LEVEL METRICS")
    print("="*40)
    
    print(f"\nOverall Industry Metrics:")
    print(f"  Accuracy:            {metrics['industry']['accuracy']:.4f}")
    print(f"  Macro Precision:     {metrics['industry']['macro_precision']:.4f}")
    print(f"  Macro Recall:        {metrics['industry']['macro_recall']:.4f}")
    print(f"  Macro F1 Score:      {metrics['industry']['macro_f1']:.4f}")
    print(f"  Weighted Precision:  {metrics['industry']['weighted_precision']:.4f}")
    print(f"  Weighted Recall:     {metrics['industry']['weighted_recall']:.4f}")
    print(f"  Weighted F1 Score:   {metrics['industry']['weighted_f1']:.4f}")
    
    print(f"\nIndustry Error Analysis:")
    print(f"  Total Predictions:     {error_analysis['industry']['total_predictions']}")
    print(f"  Correct Predictions:   {error_analysis['industry']['total_predictions'] - error_analysis['industry']['error_count']}")
    print(f"  Incorrect Predictions: {error_analysis['industry']['error_count']}")
    print(f"  Error Rate:            {error_analysis['industry']['error_rate']:.4f}")
    
    print(f"\nTop 5 Most Common Industry Errors:")
    for i, error in enumerate(error_analysis['industry']['common_errors'][:5]):
        print(f"  {i+1}. True: '{error['true_category']}' → Predicted: '{error['predicted_category']}'")
        print(f"     Count: {error['count']} ({error['percentage']:.1f}% of errors)")
    
    # Print sector-level metrics
    print("\n" + "="*40)
    print(" "*10 + "SECTOR-LEVEL METRICS")
    print("="*40)
    
    print(f"\nOverall Sector Metrics:")
    print(f"  Accuracy:            {metrics['sector']['accuracy']:.4f}")
    print(f"  Macro Precision:     {metrics['sector']['macro_precision']:.4f}")
    print(f"  Macro Recall:        {metrics['sector']['macro_recall']:.4f}")
    print(f"  Macro F1 Score:      {metrics['sector']['macro_f1']:.4f}")
    print(f"  Weighted Precision:  {metrics['sector']['weighted_precision']:.4f}")
    print(f"  Weighted Recall:     {metrics['sector']['weighted_recall']:.4f}")
    print(f"  Weighted F1 Score:   {metrics['sector']['weighted_f1']:.4f}")
    
    print(f"\nSector Error Analysis:")
    print(f"  Total Predictions:     {error_analysis['sector']['total_predictions']}")
    print(f"  Correct Predictions:   {error_analysis['sector']['total_predictions'] - error_analysis['sector']['error_count']}")
    print(f"  Incorrect Predictions: {error_analysis['sector']['error_count']}")
    print(f"  Error Rate:            {error_analysis['sector']['error_rate']:.4f}")
    
    print(f"\nTop 5 Most Common Sector Errors:")
    for i, error in enumerate(error_analysis['sector']['common_errors'][:5]):
        print(f"  {i+1}. True: '{error['true_category']}' → Predicted: '{error['predicted_category']}'")
        print(f"     Count: {error['count']} ({error['percentage']:.1f}% of errors)")
    
    # Print sector improvement over industry
    industry_error_rate = error_analysis['industry']['error_rate']
    sector_error_rate = error_analysis['sector']['error_rate']
    
    if industry_error_rate > 0:
        improvement = (industry_error_rate - sector_error_rate) / industry_error_rate * 100
        print(f"\nSector-Level Improvement:")
        print(f"  Error rate reduction: {improvement:.2f}% compared to industry-level")
        print(f"  Sector accuracy ({metrics['sector']['accuracy']:.4f}) vs. Industry accuracy ({metrics['industry']['accuracy']:.4f})")
    
    print("\n" + "="*80)

def run_batch_evaluation(client, output_file_id, test_items, output_dir="evaluation_outputs"):
    """
    Complete end-to-end evaluation of a batch job with both industry and sector level analysis.
    
    Args:
        client: OpenAI client
        output_file_id: ID of the output file from the batch job
        test_items: List of (prompt, interest_groups, expected) tuples
        output_dir: Directory to save evaluation outputs
        
    Returns:
        dict: Summary of evaluation results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download batch results
    results_path = os.path.join(output_dir, "batch_results.jsonl")
    batch_results = download_batch_results(client, output_file_id, results_path)
    
    if not batch_results:
        print("Failed to download batch results. Evaluation aborted.")
        return None
    
    # Match predictions with expected outputs (now includes sector mapping)
    evaluation_pairs = match_predictions_with_expected(batch_results, test_items)
    
    if not evaluation_pairs:
        print("No valid evaluation pairs found. Evaluation aborted.")
        return None
    
    # Calculate metrics for both industry and sector
    metrics = calculate_metrics(evaluation_pairs)
    
    # Analyze errors for both industry and sector
    error_analysis = analyze_errors(evaluation_pairs)
    
    # Generate confusion matrices for both industry and sector
    cm_paths = generate_confusion_matrix(evaluation_pairs, output_dir)
    
    # Save detailed results
    results_json_path = os.path.join(output_dir, "prediction_results.json")
    save_prediction_results(evaluation_pairs, metrics, error_analysis, cm_paths, results_json_path)
    
    # Print summary
    print_metrics_summary(metrics, error_analysis)
    
    # Return summary for reference
    return {
        "metrics": {
            "industry": {
                "accuracy": metrics["industry"]["accuracy"],
                "macro_precision": metrics["industry"]["macro_precision"],
                "macro_recall": metrics["industry"]["macro_recall"],
                "macro_f1": metrics["industry"]["macro_f1"]
            },
            "sector": {
                "accuracy": metrics["sector"]["accuracy"],
                "macro_precision": metrics["sector"]["macro_precision"],
                "macro_recall": metrics["sector"]["macro_recall"],
                "macro_f1": metrics["sector"]["macro_f1"]
            }
        },
        "error_rate": {
            "industry": error_analysis["industry"]["error_rate"],
            "sector": error_analysis["sector"]["error_rate"]
        },
        "outputs": {
            "results_file": results_json_path,
            "confusion_matrices": cm_paths
        }
    }

def main_test():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned OpenAI model on batch test data")
    parser.add_argument("--test_file", default="../data/openai_finetune/test_batch.jsonl", 
                       help="Path to the test batch JSONL file")
    parser.add_argument("--model_id", default="ft:gpt-4.1-mini-2025-04-14:personal:igclassify-v2:BXRi9hNg", 
                       help="ID of the fine-tuned model")
    parser.add_argument("--api_key", default=None, 
                       help="OpenAI API key (if not set in environment)")
    parser.add_argument("--output", default="batch_input.jsonl", 
                       help="Path to save batch input file")
    parser.add_argument("--categories", default=None, 
                       help="Path to a file with allowed categories (one per line)")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor the batch job until completion")
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Extract test prompts, interest groups, and expected outputs
    test_items = extract_test_prompts_and_expected(test_data)
    print(f"Extracted {len(test_items)} test items")

    print(f"Example test item: {test_items[0]}")
    
    # Determine allowed categories
    if args.categories:
        allowed_categories = load_categories(args.categories)
    else:
        # Extract categories from test data
        allowed_categories = INDUSTRIES
    
    print(f"Using {len(allowed_categories)} categories for classification:")
    for cat in allowed_categories:
        print(f"  - {cat}")
    
    # Prepare batch data
    batch_file = prepare_batch_data(test_items, args.model_id, allowed_categories, args.output)
    
    # Submit batch job
    batch_id, file_id = submit_batch_job(client, batch_file)
    
    if batch_id:
        if args.monitor:
            print("\nMonitoring batch job until completion...")
            output_file_id = monitor_batch_job(client, batch_id)
            
            if output_file_id:
                print(f"\nJob completed successfully!")
                print(f"Output file ID: {output_file_id}")
                print(f"To retrieve results: client.files.content('{output_file_id}')")
                print(f"To analyze results, download and compare with expected outputs")
        else:
            print("\nNext steps:")
            print(f"1. Monitor batch job status: client.batches.retrieve('{batch_id}')")
            print(f"2. Once complete, retrieve output file ID from: client.batches.retrieve('{batch_id}').output_file_id")
            print(f"3. Download results with: client.files.content(output_file_id)")
            print(f"4. Analyze the results by comparing with expected outputs")

def main_run(input_file="../data/unique_amicus_merged_names.csv", name_col="merged_name", batch_size=50):
    """
    Main entry point for running the OpenAI batch job.
    
    Args:
        input_file: Path to the input CSV file containing names
        name_col: Name of the column in the CSV file containing names
    """
    # Load input data
    df = pd.read_csv(input_file)
    
    # Extract names from the specified column
    names = df[name_col].tolist()
    names = [normalize_name(name) for name in names]
    names = list(set(names))  # Remove duplicates
    print(f"Loaded {len(names)} unique names from {input_file}")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Prepare batch data
    batched_names = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    test_items = [(names, None, None) for names in batched_names]
    print(f"Prepared {len(test_items)} test items for batch processing")

    batch_data = prepare_batch_data(test_items, 
        model_id="ft:gpt-4.1-mini-2025-04-14:personal:igclassify-v2:BXRi9hNg",
        allowed_categories=INDUSTRIES, 
        output_file=f"batch_input_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    print(f"Batch data prepared and saved to {batch_data}")
    
    # Submit batch job
    batch_id, output_file_id = submit_batch_job(client, batch_data)
    
    if batch_id:
        print(f"Batch job submitted successfully! Batch ID: {batch_id}")
        print(f"Output file ID: {output_file_id}")
        print("Use this ID to monitor or retrieve results.")

def main_analyze(batch_id=None, output_file_id=None, test_file="../data/openai_finetune/test_batch.jsonl"):
    """
    Main entry point for downloading and evaluating batch results at both industry and sector levels.
    
    Args:
        batch_id: ID of the batch job (if output_file_id is not provided)
        output_file_id: ID of the output file (if already known)
        test_file: Path to the test batch JSONL file
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Get output file ID if not provided
    if not output_file_id and batch_id:
        try:
            batch = client.batches.retrieve(batch_id)
            if batch.status == "completed":
                output_file_id = batch.output_file_id
                print(f"Retrieved output file ID: {output_file_id}")
            else:
                print(f"Batch job is not completed (status: {batch.status})")
                return
        except Exception as e:
            print(f"Error retrieving batch job: {e}")
            return
    
    if not output_file_id:
        print("No output file ID provided. Please provide either batch_id or output_file_id.")
        return
    
    # Load test data
    test_data = load_test_data(test_file)
    
    # Extract test items
    test_items = extract_test_prompts_and_expected(test_data)
    
    # Run evaluation (now includes sector-level analysis)
    evaluation_results = run_batch_evaluation(client, output_file_id, test_items)
    
    if evaluation_results:
        print("\nEvaluation completed successfully!")
        print(f"Industry Accuracy: {evaluation_results['metrics']['industry']['accuracy']:.4f}")
        print(f"Industry Macro F1: {evaluation_results['metrics']['industry']['macro_f1']:.4f}")
        print(f"Sector Accuracy:   {evaluation_results['metrics']['sector']['accuracy']:.4f}")
        print(f"Sector Macro F1:   {evaluation_results['metrics']['sector']['macro_f1']:.4f}")
        print(f"See detailed results in: {evaluation_results['outputs']['results_file']}")

if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Download and evaluate OpenAI batch results at industry and sector levels")
    # parser.add_argument("--batch_id", help="ID of the batch job")
    # parser.add_argument("--output_file_id", help="ID of the output file from a completed batch job")
    # parser.add_argument("--test_file", default="../data/openai_finetune/test_batch.jsonl", 
    #                   help="Path to the test batch JSONL file")
    # parser.add_argument("--output_dir", default="evaluation_outputs",
    #                   help="Directory to save evaluation outputs")
    
    # args = parser.parse_args()
    
    # if not args.batch_id and not args.output_file_id:
    #     print("Please provide either --batch_id or --output_file_id")
    # else:
    #     main_analyze(args.batch_id, args.output_file_id, args.test_file)
    # main_submit()
    main_run()