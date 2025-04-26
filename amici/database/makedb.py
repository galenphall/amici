import json, os, re
import pandas as pd 
import numpy as np
import sqlite3
from datetime import datetime
import argparse
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress warnings
logging.getLogger("tqdm").setLevel(logging.CRITICAL)

def ensure_json_serializable(obj):
    """
    Ensure that the object is JSON serializable.
    """
    if isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None:
        return "None"
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def map_urls_to_json(input_dir='../data/extracted_entities', output_dir='../data/extracted_entities_linked'):
    """
    Iterate through the JSON output files from the OpenAI API and add the corresponding URLs and 
    metadata from the original pdfs to the JSON files.
    """
    df = pd.read_csv('../data/amicus_briefs.csv')
    json_filenames = os.listdir(input_dir)
    json_filenames = [f for f in json_filenames if f.endswith('.json')]

    df['json_filename'] = df['url'].apply(lambda x: x.split('/')[-1].replace('.pdf', '.json'))

    logging.info(f"Found {len(json_filenames)} JSON files to process.")
    logging.info(f"Found {len(df)} rows in the CSV file.")
    for f in tqdm(json_filenames):
        if f.endswith('.json'):
            # Find the corresponding row in the DataFrame
            row = df[df['json_filename'] == f]
            if not row.empty:
                # Read the JSON file
                with open(os.path.join(input_dir, f), 'r') as file:
                    data = json.load(file)
                
                # Add the URL and metadata to the JSON data
                data['url'] = row.iloc[0]['url']
                for key in row.columns:
                    if key != 'json_filename':
                        data[key] = row.iloc[0][key]
                
                try:
                    # Write the updated JSON data back to the file
                    with open(os.path.join(output_dir, f), 'w') as file:
                        json.dump(ensure_json_serializable(data), file, indent=4)
                except Exception as e:
                    print(f"Error writing to file {f}: {e}")
                    print(f"Data: {data}")
                    raise e

def create_database(folder_path, db_name='supreme_court_docs.db', batch_size=100):
    """
    Create a relational database from JSON files in the given folder.
    
    Args:
        folder_path: Path to folder containing JSON files
        db_name: Name of the database file to create
        batch_size: Number of files to process in a single transaction
    """
    # Connect to SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create tables and indexes
    create_tables(cursor)
    
    # Process JSON files
    process_files(folder_path, cursor, conn, batch_size)
    
    # Close connection
    conn.close()
    
    print(f"Database '{db_name}' created successfully.")

def create_tables(cursor):
    """Create tables and indexes"""
    
    # Documents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        document_id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        docket_url TEXT,
        date TEXT,
        date_formatted DATE,
        label TEXT,
        doc_title TEXT,
        blob TEXT,
        transcribed BOOLEAN,
        neededOCR BOOLEAN,
        complete_amici_list BOOLEAN,
        counsel_of_record TEXT
    )
    ''')
    
    # Dockets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dockets (
        docket_id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        year INTEGER,
        number INTEGER,
        position TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (document_id)
    )
    ''')
    
    # Amici table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS amici (
        amicus_id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        name TEXT,
        category TEXT,
        FOREIGN KEY (document_id) REFERENCES documents (document_id)
    )
    ''')
    
    # Lawyers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lawyers (
        lawyer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        name TEXT,
        role TEXT,
        employer TEXT,
        is_counsel_of_record BOOLEAN,
        FOREIGN KEY (document_id) REFERENCES documents (document_id)
    )
    ''')
    
    # Create indexes for commonly queried columns
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_date ON documents (date_formatted)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_counsel ON documents (counsel_of_record)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_dockets_year_number ON dockets (year, number)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_lawyers_name ON lawyers (name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_amici_name ON amici (name)')

def process_files(folder_path, cursor, conn, batch_size=100):
    """Process JSON files in batches"""
    # Get all JSON files
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    total_files = len(json_files)
    
    print(f"Found {total_files} JSON files to process.")
    
    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = json_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files+batch_size-1)//batch_size}")
        
        # Start a transaction for the batch
        cursor.execute('BEGIN TRANSACTION')
        
        for file_name in batch_files:
            file_path = os.path.join(folder_path, file_name)
            
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    
                    # Insert into documents table
                    cursor.execute('''
                    INSERT INTO documents (
                        url, docket_url, date, date_formatted, label, doc_title, 
                        blob, transcribed, neededOCR, complete_amici_list, counsel_of_record
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        data.get('url', ''),
                        data.get('docket_url', ''),
                        data.get('date', ''),
                        format_date(data.get('date', '')),
                        data.get('label', ''),
                        data.get('doc_title', ''),
                        data.get('blob', ''),
                        data.get('transcribed', False),
                        data.get('neededOCR', False),
                        data.get('complete_amici_list', False),
                        data.get('counsel_of_record', '')
                    ))
                    
                    # Get the inserted document ID
                    document_id = cursor.lastrowid
                    
                    # Insert into dockets table
                    for docket in data.get('dockets', []):
                        cursor.execute('''
                        INSERT INTO dockets (document_id, year, number, position)
                        VALUES (?, ?, ?, ?)
                        ''', (
                            document_id,
                            docket.get('year', 0),
                            docket.get('number', 0),
                            docket.get('position', '')
                        ))
                    
                    # Insert into amici table
                    for amicus in data.get('amici', []):
                        cursor.execute('''
                        INSERT INTO amici (document_id, name, category)
                        VALUES (?, ?, ?)
                        ''', (
                            document_id,
                            amicus.get('name', ''),
                            amicus.get('category', '')
                        ))
                    
                    # Insert into lawyers table
                    for lawyer in data.get('lawyers', []):
                        # Check if this lawyer is the counsel of record
                        is_counsel = lawyer.get('name', '') == data.get('counsel_of_record', '')
                        
                        cursor.execute('''
                        INSERT INTO lawyers (document_id, name, role, employer, is_counsel_of_record)
                        VALUES (?, ?, ?, ?, ?)
                        ''', (
                            document_id,
                            lawyer.get('name', ''),
                            lawyer.get('role', ''),
                            lawyer.get('employer', ''),
                            is_counsel
                        ))
                    
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
        
        # Commit the transaction for this batch
        conn.commit()
        print(f"Processed {min(i+batch_size, total_files)}/{total_files} files")

def format_date(date_str):
    """Convert date string like 'Jul 27 2018' to a formatted date '2018-07-27'."""
    if not date_str:
        return ''
    
    try:
        # Parse the date string
        date_obj = datetime.strptime(date_str, '%b %d %Y')
        return date_obj.strftime('%Y-%m-%d')
    except ValueError:
        return ''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a database from JSON files.')
    parser.add_argument('folder', help='Path to the folder containing JSON files')
    parser.add_argument('--db', default='supreme_court_docs.db', help='Database filename')
    parser.add_argument('--batch', type=int, default=100, help='Batch size for processing')
    parser.add_argument('--map-urls', action='store_true', help='Run map_urls_to_json before creating the database')
    
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: Folder '{args.folder}' not found.")
        exit(1)

    source_folder = args.folder

    if args.map_urls:
        if not os.path.isdir(args.folder.strip('/') + '_linked/'):
            os.makedirs(args.folder.strip('/') + '_linked/')
        else:
            print(f"Warning: Folder '{args.folder.strip('/') + '_linked/'}' already exists. Overwriting files.")

        map_urls_to_json(args.folder, args.folder.strip('/') + '_linked/')
        print("Mapped URLs to JSON files.")

        source_folder = args.folder.strip('/') + '_linked/'
    
    create_database(source_folder, args.db, args.batch)
