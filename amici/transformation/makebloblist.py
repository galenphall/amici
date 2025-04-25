import pandas as pd
import os
from pathlib import Path
import re

if __name__ == "__main__":
    # Get the current directory
    current_dir = Path(__file__).parent
    # Get the parent directory
    parent_dir = current_dir.parent
    # Get the data directory
    data_dir = parent_dir / "data"
    # Get the path to the csv file
    csv_file_path = data_dir / "amicus_briefs.csv"

    df = pd.read_csv(csv_file_path)

    blobs = df.blob.tolist()

    # Replace .pdf endings with .txt
    blobs = [re.sub(r'\.pdf$', '.txt', blob) for blob in blobs]

    # Save the blobs to a text file
    blobs_file_path = data_dir / "alltextblobs.txt"
    with open(blobs_file_path, 'w') as f:
        for blob in blobs:
            f.write(blob + '\n')
    print(f"Blobs saved to {blobs_file_path}")

