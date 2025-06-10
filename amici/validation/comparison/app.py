import os
import sys
import sqlite3
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gcs import GCSFetch

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'supreme_court_docs.db')
SPREADSHEET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'handcoded.xlsx')
BUCKET_NAME = 'interest_groups_raw_documents_2025'

# Add this after DATABASE_PATH definition
if not os.path.exists(DATABASE_PATH):
    print(f"ERROR: Database not found at {DATABASE_PATH}")
else:
    print(f"Database found at {DATABASE_PATH}")

# Initialize GCS client
gcs_client = GCSFetch(BUCKET_NAME)

# Load spreadsheet data
spreadsheet_df = pd.read_excel(SPREADSHEET_PATH)

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize validation tables"""
    with open(os.path.join(os.path.dirname(__file__), 'schema.sql'), 'r') as f:
        schema = f.read()
    
    conn = get_db()
    conn.executescript(schema)
    conn.commit()
    conn.close()

def populate_initial_matches():
    """Populate initial document matches based on case ID and date order"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get all documents with docket info, ordered by docket and date
    cursor.execute("""
        SELECT d.document_id, d.date_formatted, dk.year, dk.number
        FROM documents d
        JOIN dockets dk ON d.document_id = dk.document_id
        WHERE d.complete_amici_list = 1
        ORDER BY dk.year, dk.number, d.date_formatted
    """)
    
    documents = cursor.fetchall()
    
    # Group documents by docket
    docket_groups = {}
    for doc in documents:
        # Convert to spreadsheet case_id format (e.g., "2018-001")
        case_id = f"20{doc['year']}-{str(doc['number']).zfill(3)}"
        if case_id not in docket_groups:
            docket_groups[case_id] = []
        docket_groups[case_id].append(doc['document_id'])
    
    # Match with spreadsheet rows
    for case_id, doc_ids in docket_groups.items():
        # Find matching rows in spreadsheet
        matching_rows = spreadsheet_df[spreadsheet_df['caseId'] == case_id]
        
        # Match documents to rows by order
        for i, (idx, row) in enumerate(matching_rows.iterrows()):
            if i < len(doc_ids):
                cursor.execute("""
                    INSERT OR REPLACE INTO document_matches 
                    (document_id, spreadsheet_row_index, case_id, is_confirmed)
                    VALUES (?, ?, ?, ?)
                """, (doc_ids[i], idx, case_id, False))
    
    conn.commit()
    conn.close()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/documents')
def get_documents():
    """Get all documents with match status"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            d.document_id,
            d.label,
            d.date_formatted,
            dk.year,
            dk.number,
            dm.case_id,
            dm.is_confirmed,
            dm.spreadsheet_row_index,
            COUNT(DISTINCT a.amicus_id) as my_amici_count
        FROM documents d
        JOIN dockets dk ON d.document_id = dk.document_id
        LEFT JOIN document_matches dm ON d.document_id = dm.document_id
        LEFT JOIN amici a ON d.document_id = a.document_id
        -- WHERE d.complete_amici_list = 1  -- Comment this out temporarily
        WHERE dk.year = 18 AND dm.is_confirmed
        GROUP BY d.document_id
        ORDER BY dk.year DESC, dk.number, d.date_formatted
        -- LIMIT 100  -- Add a limit to avoid loading too many
    """)
    
    documents = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(documents)

@app.route('/api/document/<int:doc_id>')
def get_document(doc_id):
    """Get document details with comparison state"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get document info
    cursor.execute("""
        SELECT d.*, dk.year, dk.number, dm.spreadsheet_row_index, dm.case_id
        FROM documents d
        JOIN dockets dk ON d.document_id = dk.document_id
        LEFT JOIN document_matches dm ON d.document_id = dm.document_id
        WHERE d.document_id = ?
    """, (doc_id,))
    
    document = dict(cursor.fetchone())
    
    # Get my amici
    cursor.execute("""
        SELECT name FROM amici WHERE document_id = ? ORDER BY name
    """, (doc_id,))
    my_amici = [row['name'] for row in cursor.fetchall()]
    
    # Get their amici from spreadsheet
    their_amici = []
    if document['spreadsheet_row_index'] is not None:
        row = spreadsheet_df.iloc[document['spreadsheet_row_index']]
        their_amici = [name.strip() for name in row['Cosign_group_n'].split(';')]
    
    # Get marked entities
    cursor.execute("""
        SELECT entity_name, entity_type 
        FROM comparison_results 
        WHERE document_id = ?
    """, (doc_id,))
    marked_entities = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return jsonify({
        'document': document,
        'my_amici': my_amici,
        'their_amici': their_amici,
        'marked_entities': marked_entities
    })

@app.route('/api/document/<int:doc_id>/pdf')
def get_document_pdf(doc_id):
    """Get PDF content from GCS"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get blob name from document
    cursor.execute("SELECT blob FROM documents WHERE document_id = ?", (doc_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'error': 'Document not found'}), 404
    
    blob_name = result['blob']
    
    try:
        content, metadata = gcs_client.get_from_bucket(blob_name)
        return Response(content, mimetype='application/pdf')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/mark_entity', methods=['POST'])
def mark_entity():
    """Mark an entity as missed"""
    data = request.json
    doc_id = data['document_id']
    entity_name = data['entity_name']
    entity_type = data['entity_type']  # 'missed_by_me' or 'missed_by_them'
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Check if already marked
    cursor.execute("""
        SELECT id FROM comparison_results 
        WHERE document_id = ? AND entity_name = ? AND entity_type = ?
    """, (doc_id, entity_name, entity_type))
    
    existing = cursor.fetchone()
    
    if not existing:
        cursor.execute("""
            INSERT INTO comparison_results (document_id, entity_name, entity_type)
            VALUES (?, ?, ?)
        """, (doc_id, entity_name, entity_type))
        conn.commit()
    
    conn.close()
    return jsonify({'success': True})

@app.route('/api/unmark_entity', methods=['POST'])
def unmark_entity():
    """Remove entity marking"""
    data = request.json
    doc_id = data['document_id']
    entity_name = data['entity_name']
    entity_type = data['entity_type']
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM comparison_results 
        WHERE document_id = ? AND entity_name = ? AND entity_type = ?
    """, (doc_id, entity_name, entity_type))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/cycle_match/<int:doc_id>')
def cycle_match(doc_id):
    """Get next potential match for document"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Get current document's docket info
    cursor.execute("""
        SELECT dk.year, dk.number, dm.spreadsheet_row_index
        FROM documents d
        JOIN dockets dk ON d.document_id = dk.document_id
        LEFT JOIN document_matches dm ON d.document_id = dm.document_id
        WHERE d.document_id = ?
    """, (doc_id,))
    
    result = cursor.fetchone()
    if not result:
        return jsonify({'error': 'Document not found'}), 404
    
    # Convert to case_id format
    case_id = f"20{result['year']}-{str(result['number']).zfill(3)}"
    current_index = result['spreadsheet_row_index']
    
    # Find all rows with this case_id
    matching_indices = spreadsheet_df[spreadsheet_df['caseId'] == case_id].index.tolist()
    
    if not matching_indices:
        return jsonify({'error': 'No matches found'}), 404
    
    # Find next index
    if current_index is None:
        next_index = matching_indices[0]
    else:
        try:
            current_pos = matching_indices.index(current_index)
            next_index = matching_indices[(current_pos + 1) % len(matching_indices)]
        except ValueError:
            next_index = matching_indices[0]
    
    # Get the row data
    row = spreadsheet_df.iloc[next_index]
    their_amici = [name.strip() for name in row['Cosign_group_n'].split(';')]
    
    conn.close()
    
    return jsonify({
        'spreadsheet_row_index': int(next_index),
        'case_id': case_id,
        'their_amici': their_amici,
        'support_n': int(row['Support_n'])
    })

@app.route('/api/update_match', methods=['POST'])
def update_match():
    """Update document match"""
    data = request.json
    doc_id = data['document_id']
    row_index = data['spreadsheet_row_index']
    case_id = data['case_id']
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO document_matches 
        (document_id, spreadsheet_row_index, case_id, is_confirmed)
        VALUES (?, ?, ?, ?)
    """, (doc_id, row_index, case_id, True))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/api/export_results')
def export_results():
    """Export comparison results as JSON"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            d.document_id,
            d.label,
            dk.year,
            dk.number,
            dm.case_id,
            cr.entity_name,
            cr.entity_type,
            cr.marked_at
        FROM comparison_results cr
        JOIN documents d ON cr.document_id = d.document_id
        JOIN dockets dk ON d.document_id = dk.document_id
        LEFT JOIN document_matches dm ON d.document_id = dm.document_id
        ORDER BY d.document_id, cr.entity_type, cr.entity_name
    """)
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(results)

@app.route('/api/debug')
def debug():
    """Debug endpoint to check database state"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Check documents count
    cursor.execute("SELECT COUNT(*) as count FROM documents WHERE complete_amici_list = 1")
    doc_count = cursor.fetchone()['count']
    
    # Check if dockets table exists and has data
    cursor.execute("SELECT COUNT(*) as count FROM dockets")
    docket_count = cursor.fetchone()['count']
    
    # Check if amici table exists and has data
    cursor.execute("SELECT COUNT(*) as count FROM amici")
    amici_count = cursor.fetchone()['count']
    
    # Get sample data
    cursor.execute("""
        SELECT d.document_id, d.label, dk.year, dk.number 
        FROM documents d 
        JOIN dockets dk ON d.document_id = dk.document_id 
        WHERE d.complete_amici_list = 1 
        LIMIT 5
    """)
    sample_docs = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return jsonify({
        'documents_count': doc_count,
        'dockets_count': docket_count,
        'amici_count': amici_count,
        'sample_documents': sample_docs
    })

if __name__ == '__main__':
    # Initialize database tables
    init_db()
    
    # Populate initial matches if needed
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as count FROM document_matches")
    if cursor.fetchone()['count'] == 0:
        populate_initial_matches()
    conn.close()
    
    app.run(debug=True, port=5000)