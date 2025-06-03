-- Table to store document-to-spreadsheet matches
CREATE TABLE IF NOT EXISTS document_matches (
    document_id INTEGER PRIMARY KEY,
    spreadsheet_row_index INTEGER,
    case_id TEXT,
    is_confirmed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

-- Table to store comparison results
CREATE TABLE IF NOT EXISTS comparison_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    entity_name TEXT,
    entity_type TEXT, -- 'missed_by_me' or 'missed_by_them'
    marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_comparison_results_document 
ON comparison_results(document_id);

CREATE INDEX IF NOT EXISTS idx_document_matches_case_id 
ON document_matches(case_id);