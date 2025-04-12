import pandas as pd
import os
import json
import re
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter

uuid_to_source = pd.read_csv("../data/SupremeCourtData/embedded_text_map.csv")
uuid_to_source.columns = ['source_file', 'has_text', 'uuid']
uuid_to_source_map = uuid_to_source.set_index('uuid').source_file
source_to_uuid_map = uuid_to_source.set_index('source_file').uuid

def normalize_docket_number(docket: str) -> str:
    """
    Normalize a docket number to the YY-N{1,4} format.
    Examples:
    - "No. 18-271" -> "18-271"
    - "Case No. 18-271" -> "18-271"
    - "18-271" -> "18-271"
    """
    # Remove prefix like "No." and any whitespace
    match = re.search(r'(\d{1,2}-\d{1,4})', docket)
    if match:
        return match.group(1)
    return None  # Return original if no match found

def docket_number_from_source_file_path(filepath: str) -> str:
    """
    Extract the docket number from a pdf file location.
    Example:
    - SUPREMECOURT/www.supremecourt.gov/DocketPDF/20/20-843/[...].pdf -> 20-843
    """
    match = re.search(r'/(\d{1,2}-\d{1,4})/', filepath)
    if match: 
        return match.group(1)
    return None

def process_result_file(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process a single result file and return:
    1. List of row dictionaries
    2. Statistics about the file
    """
    stats = {
        'processed': 0,
        'errors': 0,
        'amici_count': 0,
        'docket_count': 0,
    }
    
    try:
        with open(file_path, 'r') as f:
            result = json.load(f)
        
        stats['processed'] = 1
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON in {file_path}")
        stats['errors'] = 1
        return [], stats
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        stats['errors'] = 1
        return [], stats
    
    rows = []
    
    # Skip files with errors
    if result.get('error'):
        print(f"Skipping file with error: {file_path}")
        stats['errors'] = 1
        return [], stats
    
    # Get the source file (brief filename)
    source_file = result.get('source_file', 'unknown')
    # if '.txt' in source_file:
    #     text_file = os.path.join("../data/SupremeCourtData/brieftext", source_file)
    #     source_file = uuid_to_source_map.get(source_file.replace(".txt", ""), source_file)
    # else:
    #     text_file = os.path.join("../data/SupremeCourtData/brieftext/", source_to_uuid_map[source_file] + ".txt")

    # if not source_file:
    #     print(f"No source file found for {file_path}")
    #     stats['errors'] = 1
    #     return [], stats
    
    # Get the brief position
    brief = result.get('brief', {})
    position = brief.get('position', 'unknown')
    
    # Get the docket numbers and normalize them
    dockets = brief.get('dockets', [])
    normalized_dockets = [normalize_docket_number(docket) for docket in dockets]
    normalized_dockets = [d for d in normalized_dockets if d]

    # At least one of the dockets should be in the source_file path
    if any((d not in source_file) or (d[:2] < "17") for d in normalized_dockets):
        normalized_dockets = []

    # If there are no properly normalized dockets present, use the filename
    if len(normalized_dockets) == 0:
        normalized_docket_from_file = docket_number_from_source_file_path(source_file)
        if normalized_docket_from_file:
            normalized_dockets = [normalized_docket_from_file]

    # Last resort: try opening the source file and regexing the docket numbers
    # if len(normalized_dockets) == 0:
    #     with open(text_file, 'r') as f:
    #         text = f.read().strip()
    #     normalized_dockets = re.findall(r'[^\d](\d{1,2}-\d{1,4})[^\d]', text[:500])

    if len(normalized_dockets) == 0:
        print(f"Error reading {file_path}: No dockets found")
        stats['errors'] = 1
        return [], stats

    stats['docket_count'] = len(normalized_dockets)
    
    # Get the amici
    amici = result.get('amici', [])
    stats['amici_count'] = len(amici)
    
    # Create a row for each combination of amicus and docket number
    for amicus in amici:
        amicus_name = amicus.get('name', '')
        amicus_type = amicus.get('type', '')
        
        for docket in normalized_dockets:
            row = {
                'source_file': source_file,
                'amicus_name': amicus_name,
                'amicus_type': amicus_type,
                'docket_number': docket,
                'position': position,
                'confidence': result.get('confidence', 'unknown')
            }
            rows.append(row)
    
    return rows, stats

def find_json_files(results_dirs: List[str]) -> List[str]:
    """Find all JSON files in the given directories, recursively."""
    json_files = []
    
    for results_dir in results_dirs:
        # First, check if the directory exists
        if not os.path.exists(results_dir):
            print(f"Warning: Directory {results_dir} does not exist.")
            continue
        
        # Find all JSON files recursively
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.json') and file != "uuid_to_source_map.json":
                    json_files.append(os.path.join(root, file))
    
    return json_files

def process_batch_results(results_dirs: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process all JSON result files in the specified directories and create a table
    with each amicus, docket number, and position on its own row.
    
    Returns:
    1. DataFrame with the results
    2. Statistics about the processing
    """
    all_rows = []
    
    # Statistics dictionary
    stats = {
        'files_processed': 0,
        'files_with_errors': 0,
        'total_amici': 0,
        'total_dockets': 0,
        'position_counts': Counter(),
        'amicus_type_counts': Counter(),
    }
    
    # Find all JSON files
    json_files = find_json_files(results_dirs)

    # Load the mapping between file uuids and pdf names
    uuid_to_source_map_path = os.path.join(results_dirs[0], "uuid_to_source_map.json")
    
    if not json_files:
        print("No JSON files found in the specified directories.")
        return pd.DataFrame(), stats
    
    print(f"Processing {len(json_files)} JSON files...")
    
    for file_path in json_files:
        rows, file_stats = process_result_file(file_path)
        all_rows.extend(rows)
        
        # Update statistics
        stats['files_processed'] += file_stats['processed']
        stats['files_with_errors'] += file_stats['errors']
        stats['total_amici'] += file_stats['amici_count']
        stats['total_dockets'] += file_stats['docket_count']
    
    # Create DataFrame
    if not all_rows:
        print("Warning: No data rows created. Check input files.")
        return pd.DataFrame(), stats
    
    df = pd.DataFrame(all_rows)
    
    # Update more statistics from the data
    if 'position' in df.columns:
        stats['position_counts'].update(df['position'].value_counts().to_dict())
    
    if 'amicus_type' in df.columns:
        stats['amicus_type_counts'].update(df['amicus_type'].value_counts().to_dict())
    
    # Sort by docket number and amicus name
    if 'docket_number' in df.columns and 'amicus_name' in df.columns:
        df = df.sort_values(['docket_number', 'amicus_name'])
    
    return df, stats

def print_statistics(stats: Dict[str, Any]):
    """Print statistics about the processed data."""
    print("\n===== Processing Statistics =====")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files with errors: {stats['files_with_errors']}")
    print(f"Total amici found: {stats['total_amici']}")
    print(f"Total dockets found: {stats['total_dockets']}")
    
    print("\nPosition counts:")
    for position, count in stats['position_counts'].most_common():
        print(f"  {position}: {count}")
    
    print("\nAmicus type counts:")
    for amicus_type, count in stats['amicus_type_counts'].most_common():
        print(f"  {amicus_type}: {count}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create a table of amici, docket numbers, and positions from batch results')
    parser.add_argument('--results_dirs', nargs='+', required=True, 
                        help='Directories containing processed results')
    parser.add_argument('--output', type=str, default="amici_dockets_positions.csv",
                        help='Output CSV file path')
    parser.add_argument('--include_source', action='store_true',
                        help='Include source file in output')
    parser.add_argument('--include_confidence', action='store_true',
                        help='Include confidence level in output')
    
    args = parser.parse_args()
    
    # Process results into a DataFrame
    print("Processing results...")
    df, stats = process_batch_results(args.results_dirs)
    
    if df.empty:
        print("No data to save.")
        return
    
    # Remove columns if not requested
    columns_to_keep = ['amicus_name', 'amicus_type', 'docket_number', 'position']
    
    if args.include_source:
        columns_to_keep.append('source_file')
    
    if args.include_confidence:
        columns_to_keep.append('confidence')
    
    # Keep only requested columns that exist in the DataFrame
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Created table with {len(df)} rows and saved to {args.output}")
    
    # Print statistics
    print_statistics(stats)
    
    # Print a sample
    print("\nSample of data:")
    print(df.head())

if __name__ == "__main__":
    main()