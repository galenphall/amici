import sqlite3
import pandas as pd
import argparse
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_db(db_path):
    """Connect to the SQLite database"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file {db_path} not found")
    
    conn = sqlite3.connect(db_path)
    return conn

def get_table_data(conn, table_name):
    """Query a table and return the data as a DataFrame"""
    logging.info(f"Getting data from {table_name} table")
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    logging.info(f"Retrieved {len(df)} rows from {table_name}")
    return df

def export_to_excel(db_path, output_path):
    """Export all tables from the database to an Excel workbook"""
    # Connect to the database
    conn = connect_to_db(db_path)
    
    # Get a list of all tables in the database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [table[0] for table in cursor.fetchall()]
    logging.info(f"Found {len(tables)} tables in the database: {tables}")
    
    # Create Excel writer
    logging.info(f"Creating Excel workbook at {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Process each table
        for table in tqdm(tables, desc="Exporting tables"):
            df = get_table_data(conn, table)
            
            # Write to Excel sheet
            df.to_excel(writer, sheet_name=table, index=False)
            logging.info(f"Wrote {len(df)} rows to {table} sheet")
    
    conn.close()
    logging.info(f"Excel workbook created successfully at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Export database to Excel workbook.')
    parser.add_argument('--db', default='supreme_court_docs.db', help='Database filename')
    parser.add_argument('--output', default='supreme_court_docs.xlsx', help='Output Excel filename')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db):
        logging.error(f"Database file {args.db} not found")
        exit(1)
    
    export_to_excel(args.db, args.output)
    print(f"Database exported to {args.output} successfully")

if __name__ == "__main__":
    main()
