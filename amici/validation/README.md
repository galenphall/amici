# Amici Validation Tool

This tool helps validate extracted amici by comparing our database extractions with hand-coded data from a spreadsheet.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following files in place:
   - `../database/supreme_court_docs.db` - SQLite database with extracted amici
   - `../data/handcoded.xlsx` - Spreadsheet with hand-coded amici
   - `../../.env` - Environment file with Google Cloud credentials

3. Run the application:
```bash
python app.py
```

4. Open http://localhost:5000 in your browser

## Usage

### Main Interface
- **Left Panel**: Shows entities from the hand-coded spreadsheet
- **Center Panel**: Displays the PDF document from Google Cloud Storage
- **Right Panel**: Shows entities extracted by our system

### Marking Entities
- **Single-click checkbox**: Select multiple entities to mark in bulk
- **Double-click entity**: Toggle individual entity marking
- **"Mark Selected" buttons**: Mark selected entities as missed

### Entity Types
- **Missed by Me** (red background): Entities in their data that we didn't extract
- **Missed by Them** (blue background): Entities we extracted that they didn't include

### Navigation
- Use the dropdown or Previous/Next buttons to navigate documents
- PDF navigation: Use the page controls or arrow keys
- Keyboard shortcuts:
  - `←/→`: Previous/Next PDF page
  - `Ctrl+←/→`: Previous/Next document
  - `Ctrl+C`: Cycle match

### Match Correction
- Click "Wrong Match?" to cycle through potential matches
- The system will show a preview before updating
- Confirmed matches are marked with ✓ in the dropdown

### Export
- Click "Export Results" to download all marked entities as JSON

## Database Schema

The tool creates two additional tables:

1. `document_matches`: Stores document-to-spreadsheet row mappings
2. `comparison_results`: Stores marked entities (missed by either side)

## Notes

- Initial matches are created automatically based on case ID and date order
- All markings and match corrections are persisted in the database
- The tool handles PDF loading errors gracefully
- Entity comparisons are case-sensitive