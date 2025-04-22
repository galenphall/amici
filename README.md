# amici
Code for scraping and constructing a database of amicus brief signees in the US Supreme Court

# Data Collection
## Extraction

All amicus briefs submitted since 2018 are accessible via searching the database exposed at [this US Government webpage](https://www.supremecourt.gov/docket/docket.aspx). Alternatively, one can cycle through urls of the following format:

`https://www.supremecourt.gov/search.aspx?filename=/docket/docketfiles/html/public/{YY}-{N:1-5}.html`

where `{N:1-5}` indicates the docket number, ranging from `1` to somewhere in the `10000`s.

Each such webpage contains one html table with `id=docketinfo` containing the metadata about the docket (title, date docketed, lower courts) and another table with `id=proceedings` containing the (dated) documents submitted in the course of the proceedings. The `proceedings` table may contain rows corresponding to amicus briefs; the descriptions in these rows always begin with the phrase 'Brief amicus curiae' or 'Brief amici curiae' and the pdf brief is contained in the hyperlink with text 'Main Document'. 

We save all docket pages and all pdfs in a Google Cloud Bucket for persistent access.

Note that the codebase assumes both bucket access keys and OpenAI API keys are stored in a `.env` file in the outer directory.

## Transformation

### OCR
Most of the amicus brief pdfs have embedded text. A small number (~1%) do not. We use the [`ocrmypdf`](https://github.com/ocrmypdf/OCRmyPDF) library to create embedded text in these files as well, and we save the results to our Google Cloud Bucket.

### Text extraction
Once all pdfs have embedded text, we extract it and store the text in a new set of txt files in Google Cloud. We include the original file metadata in that text file for redundancy, and also store a table mapping original pdf bucket locations to their corresponding text files.

### Appendix detection
Most briefs contain the full list of amici on the title page, but some contain long appendices with lists of amici. Since we are using pay-per-token LLM calls to extract data, we only want to automaticall process those briefs with amici on the title page. So we need to detect which briefs have relevant appendices. In all such cases, the title page contains some obvious shortening of the amici (e.g. "23 US States") and the table of contents references the appendix. The exact textual and syntactic layouts of these sections vary, making regex parsing unweildy. We therefore use LLM calls with structured output to generate a boolean `appendix` field for every brief, telling us whether we can get the list of amici from the front page or not. We feed only the first 5 pages worth of text to these API calls.

### Amici extraction with structured output (no appendix)
We use LLM structured output APIs to get lists of amici from the title page of all non-appendix briefs. We also extract the names, roles, and firms for each lawyer listed on the first few pages, as well as the docket year-number-position tuples `List<(int, int, Position)>` given on the brief, since one brief may refer to multiple dockets. Here `Position` is an Enum capturing the Amici's supporting either the petitioner or respondent on a given docket. We again feed only the first 5 pages worth of text.

### Amici extraction with structured output (appendix)
Documents with appendices are harder to deal with, because we don't know _a priori_ how many pages we will need to scan, or even which appendix the amici will be listed in, and attempting to extract this information reliably is difficult. Instead we feed the entire text of these documents through a separate structured response query to a more powerful (and thus more expensive) LLM capable of reliably processing entire briefs worth of text.

## Loading
Once the amici are extracted, we create a tabular dataset. See next section.

# Database Schema

The database is structured to capture relationships between documents (amicus briefs), dockets, amici curiae, and lawyers. This relational structure enables efficient querying across all dimensions of the data.

## Tables

### documents

Primary table storing metadata about each amicus brief.

| Column | Type | Description |
|--------|------|-------------|
| document_id | INTEGER | Primary key, auto-incremented |
| url | TEXT | URL to the original document on supremecourt.gov |
| docket_url | TEXT | URL to the docket page |
| date | TEXT | Original date string (e.g., "Jul 27 2018") |
| date_formatted | DATE | Normalized date (YYYY-MM-DD format) |
| label | TEXT | Brief's label from the docket page |
| doc_title | TEXT | Document title |
| blob | TEXT | Path to document in storage |
| transcribed | BOOLEAN | Whether text was successfully extracted |
| neededOCR | BOOLEAN | Whether OCR was required |
| complete_amici_list | BOOLEAN | Whether all amici are listed or abbreviated |
| counsel_of_record | TEXT | Name of counsel of record |

### dockets

Cases referenced by each document. A single brief may refer to multiple dockets.

| Column | Type | Description |
|--------|------|-------------|
| docket_id | INTEGER | Primary key, auto-incremented |
| document_id | INTEGER | Foreign key to documents table |
| year | INTEGER | Two-digit year identifier (e.g., 18 for 2018) |
| number | INTEGER | Docket number within the year |
| position | TEXT | Position supported ("P" for Petitioner, "R" for Respondent) |

### amici

Organizations or individuals filing as amici curiae.

| Column | Type | Description |
|--------|------|-------------|
| amicus_id | INTEGER | Primary key, auto-incremented |
| document_id | INTEGER | Foreign key to documents table |
| name | TEXT | Name of the amicus curiae |
| category | TEXT | Category (e.g., "organization", "individual", "government") |

### lawyers

Legal representatives associated with each document.

| Column | Type | Description |
|--------|------|-------------|
| lawyer_id | INTEGER | Primary key, auto-incremented |
| document_id | INTEGER | Foreign key to documents table |
| name | TEXT | Lawyer's name |
| role | TEXT | Role (e.g., "Counsel of Record") |
| employer | TEXT | Law firm or organization |
| is_counsel_of_record | BOOLEAN | Whether this lawyer is the primary counsel |

## Relationships

- **One-to-Many**: A document can reference multiple dockets
- **One-to-Many**: A document can have multiple amici
- **One-to-Many**: A document can have multiple lawyers
- **One-to-One**: Each document has at most one counsel of record

## Indexes

| Table | Index | Columns | Purpose |
|-------|-------|---------|---------|
| documents | idx_document_date | date_formatted | Optimize queries by date |
| documents | idx_document_counsel | counsel_of_record | Optimize lookups by counsel |
| dockets | idx_dockets_year_number | year, number | Optimize docket lookups |
| lawyers | idx_lawyers_name | name | Optimize lawyer name searches |
| amici | idx_amici_name | name | Optimize amici name searches |

## Example Queries

### Find all briefs filed by a specific organization

```sql
SELECT d.* FROM documents d
JOIN amici a ON d.document_id = a.document_id
WHERE a.name = 'American Civil Liberties Union' AND a.category = 'organization';
```

### Find all lawyers who have served as counsel of record
```sql
sqlSELECT DISTINCT name, COUNT(*) as brief_count 
FROM lawyers 
WHERE is_counsel_of_record = 1 
GROUP BY name 
ORDER BY brief_count DESC;
```
