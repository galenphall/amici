from flask import Flask, render_template, jsonify, request
import sqlite3
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.utils.gcs import GCSFetch

app = Flask(__name__)
app.config['DATABASE'] = Path(__file__).parent.parent / 'database' / 'supreme_court_docs.db'

# Initialize GCS client
gcs = GCSFetch("interest_groups_raw_documents_2025")

def get_db():
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db

def init_validation_tables():
    """Create validation tracking tables if they don't exist"""
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS validation_results (
            validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            amicus_id INTEGER,
            is_correct BOOLEAN,
            validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(document_id),
            FOREIGN KEY (amicus_id) REFERENCES amici(amicus_id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS missed_amici (
            missed_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            name TEXT,
            category TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(document_id)
        )
    ''')
    db.commit()
    db.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/documents')
def get_documents():
    db = get_db()
    documents = db.execute('''
        SELECT d.document_id, d.doc_title, d.date_formatted, 
               COUNT(DISTINCT a.amicus_id) as amici_count
        FROM documents d
        LEFT JOIN amici a ON d.document_id = a.document_id
        GROUP BY d.document_id
        ORDER BY d.date_formatted DESC
    ''').fetchall()
    db.close()
    return jsonify([dict(doc) for doc in documents])

@app.route('/api/document/<int:doc_id>')
def get_document_details(doc_id):
    db = get_db()
    
    # Get document info
    doc = db.execute('SELECT * FROM documents WHERE document_id = ?', (doc_id,)).fetchone()
    if not doc:
        return jsonify({'error': 'Document not found'}), 404
    
    # Get docket info
    docket = db.execute('SELECT * FROM dockets WHERE document_id = ?', (doc_id,)).fetchone()
    
    # Get amici
    amici = db.execute('SELECT * FROM amici WHERE document_id = ? ORDER BY name', (doc_id,)).fetchall()
    
    # Get lawyers
    lawyers = db.execute('SELECT * FROM lawyers WHERE document_id = ? ORDER BY name', (doc_id,)).fetchall()
    
    # Get validation stats
    stats = db.execute('''
        SELECT 
            COUNT(CASE WHEN is_correct = 1 THEN 1 END) as correct,
            COUNT(CASE WHEN is_correct = 0 THEN 1 END) as incorrect,
            COUNT(DISTINCT amicus_id) as validated
        FROM validation_results
        WHERE document_id = ?
    ''', (doc_id,)).fetchone()
    
    missed_count = db.execute('SELECT COUNT(*) as count FROM missed_amici WHERE document_id = ?', 
                              (doc_id,)).fetchone()
    
    db.close()
    
    return jsonify({
        'document': dict(doc),
        'docket': dict(docket) if docket else None,
        'amici': [dict(a) for a in amici],
        'lawyers': [dict(l) for l in lawyers],
        'validation_stats': {
            'correct': stats['correct'],
            'incorrect': stats['incorrect'],
            'validated': stats['validated'],
            'total': len(amici),
            'missed': missed_count['count']
        }
    })

@app.route('/api/document/<int:doc_id>/pdf')
def get_pdf(doc_id):
    db = get_db()
    doc = db.execute('SELECT blob FROM documents WHERE document_id = ?', (doc_id,)).fetchone()
    db.close()
    
    if not doc or not doc['blob']:
        return jsonify({'error': 'PDF not found'}), 404
    
    try:
        pdf_bytes, _ = gcs.get_from_bucket(doc['blob'])
        return pdf_bytes, 200, {'Content-Type': 'application/pdf'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate_amicus():
    data = request.json
    db = get_db()
    
    # Check if already validated
    existing = db.execute('''
        SELECT validation_id FROM validation_results 
        WHERE document_id = ? AND amicus_id = ?
    ''', (data['document_id'], data['amicus_id'])).fetchone()
    
    if existing:
        # Update existing validation
        db.execute('''
            UPDATE validation_results 
            SET is_correct = ?, validated_at = CURRENT_TIMESTAMP
            WHERE validation_id = ?
        ''', (data['is_correct'], existing['validation_id']))
    else:
        # Insert new validation
        db.execute('''
            INSERT INTO validation_results (document_id, amicus_id, is_correct)
            VALUES (?, ?, ?)
        ''', (data['document_id'], data['amicus_id'], data['is_correct']))
    
    db.commit()
    db.close()
    return jsonify({'success': True})

@app.route('/api/add-missed', methods=['POST'])
def add_missed_amicus():
    data = request.json
    db = get_db()
    db.execute('''
        INSERT INTO missed_amici (document_id, name, category)
        VALUES (?, ?, ?)
    ''', (data['document_id'], data['name'], data.get('category', '')))
    db.commit()
    db.close()
    return jsonify({'success': True})

@app.route('/api/missed/<int:doc_id>')
def get_missed_amici(doc_id):
    db = get_db()
    missed = db.execute('''
        SELECT * FROM missed_amici 
        WHERE document_id = ? 
        ORDER BY added_at DESC
    ''', (doc_id,)).fetchall()
    db.close()
    return jsonify([dict(m) for m in missed])

if __name__ == '__main__':
    init_validation_tables()
    app.run(debug=True, port=5000)