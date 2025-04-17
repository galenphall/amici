"""
This script processes docket files from the Supreme Court to find links to amicus briefs.
It downloads the docket files from a Google Cloud Storage bucket, extracts the links to the amicus briefs,
and saves the information to a local file. At the end of the script the local file is uploaded to the GCS bucket.
"""
import os
from concurrent.futures import ThreadPoolExecutor
import json
from selectolax.parser import HTMLParser
from tqdm import tqdm
import urllib.parse
from datetime import datetime
import concurrent.futures
import pandas as pd
import logging
import dotenv
from pathlib import Path

import sys
import os
sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.utils.gcs import GCSFetch

# Load environment variables from .env file
env_file = os.path.join(os.path.dirname(__file__), "../../.env")
dotenv.load_dotenv(env_file)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set the logging output directory
log_dir = os.path.join(os.path.dirname(__file__), "../logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"extraction-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False

def process_docket_file(docket_html: str, docket_url: str, output_directory: str) -> None:
    """Process a single docket file to find links to amicus briefs."""
    parser = HTMLParser(docket_html)

    # Get the table containing the docket information, which has id 'proceedings'
    proceeding_table = parser.css("table#proceedings")
    if not proceeding_table:
        logger.warning(f"No proceedings table found in {docket_url}")
        return False
    # Get all rows in the table
    rows = proceeding_table[0].css("tr")
    # Remove the first row (header)
    rows.pop(0)
    # Match label rows and link rows (2n, 2n+1)
    for i in range(0, len(rows), 2):
        label_row = rows[i]
        link_row = rows[i + 1]

        # Get the date and label
        date = label_row.css_first("td").text(strip=True)
        label = label_row.css_first("td:nth-child(2)").text(strip=True)
        # Get the link from the <a> element with text "Main Document"
        link = None
        for a_tag in link_row.css('a'):
            if a_tag.text(strip=True) == "Main Document":
                link = a_tag
                break
        if link:
            href = link.attributes["href"]
            doc_title = link.text(strip=True)
            # Check if the link is a PDF
            if href.endswith(".pdf"):
                # Get the full URL
                url = urllib.parse.urljoin(docket_url, href)
                # Save the URL to a file
                with open(os.path.join(output_directory, "document_links.txt"), "a") as f:
                    f.write(f"{docket_url}\t{date}\t{label}\t{doc_title}\t{url}\n")
                logger.info(f"Found document: {url}")
    return True

# Process files concurrently using ThreadPoolExecutor
def process_docket_blob(blob, fetcher, output_directory):
    try:
        # Get the docket number from the blob name
        year, docket_num = blob.split("/")[-1].strip(".html").split("-")
        year = int(year)
        docket_num = int(docket_num)
        
        content_bytes, metadata = fetcher.get_from_bucket(blob)
        docket_html = content_bytes.decode("utf-8")
        docket_url = f"{metadata['blob_name']}"
        has_docket = process_docket_file(docket_html, docket_url, output_directory)
        
        return year, docket_num, has_docket
    except Exception as e:
        logger.error(f"Error processing {blob}: {e}")
        return None, None, False

def batch_gcs_docket_process(fetcher: GCSFetch, blobs: list, output_directory: str) -> None:
    """Process multiple docket files from GCS to find links to amicus briefs."""
    os.makedirs(output_directory, exist_ok=True)
    
    logger.info(f"Found {len(blobs)} docket files to process")
    
    # Process files concurrently
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks and get future objects
        futures = [executor.submit(process_docket_blob, blob, fetcher, output_directory) 
                  for blob in blobs]
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing docket files"):
            year, docket_num, has_docket = future.result()
    
    logger.info(f"Processing complete. Results saved to {output_directory}")

def collect_document_links(blobs: list, data_dir: str, fetcher: GCSFetch) -> None:
    docket_table_blobs = [b for b in blobs if 'filename=/docket/docketfiles/html/' in b]
    batch_gcs_docket_process(fetcher, docket_table_blobs, data_dir)

def cache_supreme_court_blob_names(fetcher, data_dir):
    blobs = list(map(lambda b: b.name, fetcher.list_blobs('SUPREMECOURT/')))
    # cache the blobs locally
    with open(os.path.join(data_dir, "supremecourt_blobs.json"), "w") as f:
        json.dump(blobs, f)
    return blobs

# Main execution
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "../data/")
    os.makedirs(data_dir, exist_ok=True)

    fetcher = GCSFetch("interest_groups_raw_documents_2025")

    # Check if the blobs are already cached
    if os.path.exists(os.path.join(data_dir, "supremecourt_blobs.json")):
        recache = input("Blobs already cached. Do you want to recache? (y/n): ")
        if recache.lower() == "y":
            blobs = cache_supreme_court_blob_names(fetcher, data_dir)
        else:
            # load the blobs from the cache
            with open(os.path.join(data_dir, "supremecourt_blobs.json"), "r") as f:
                blobs = json.load(f)
    else:
        print("No blobs cached. Caching now...")
        blobs = cache_supreme_court_blob_names(fetcher, data_dir)
    print(f"Found {len(blobs)} blobs in the bucket")

    # Step 1. Download all docket tables and use them to find the amicus briefs
    collect = input("Do you want to collect the document links? (y/n): ")
    if collect.lower() == "y":
        collect_document_links(blobs, data_dir, fetcher)
    else:
        logger.info("Skipping document link collection.")
    
    document_links_df = pd.read_csv(os.path.join(data_dir, "document_links.txt"), sep="\t", header=None)
    document_links_df.columns = ["docket_url", "date", "label", "doc_title", "url"]

    ## Make sure docket_url is in correct format
    document_links_df["docket_url"] = document_links_df["docket_url"].apply(lambda x: x.split("SUPREMECOURT/", 1)[-1] if isinstance(x, str) else "")

    ## Filter for amicus briefs
    amicus_briefs = document_links_df[document_links_df["label"].str.contains("^Brief (amicus|amici)", case=False)]
    
    # Step 2. Download all amicus briefs and extract the information
    amicus_briefs["blob"] = amicus_briefs["url"].apply(lambda x: x.replace("http://", "SUPREMECOURT/") if isinstance(x, str) else "")

    ## Keep only the briefs from '18 onwards
    amicus_briefs = amicus_briefs[amicus_briefs["blob"].str.contains(r"DocketPDF/(18|19|20|21|22|23|24)")].copy()

    ## Save the amicus briefs to a file
    amicus_briefs.to_csv(os.path.join(data_dir, "amicus_briefs.csv"), index=False)

    print(f"Found {len(amicus_briefs)} amicus briefs")
    print(amicus_briefs.head())

    ## Upload the amicus briefs to GCS
    amicus_briefs_blob = "SUPREMECOURT/amicus_briefs.csv"
    fetcher.upload_to_bucket(amicus_briefs_blob, os.path.join(data_dir, "amicus_briefs.csv"))