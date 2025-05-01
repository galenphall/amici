# Example script to run the HITL process
import sys
from pathlib import Path
import logging
import argparse
import re

# Add parent directory to path to allow relative imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.deduplication.dbdedupe import DbDeduplicator


if __name__ == "__main__":

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Argument parser for command line options
    parser = argparse.ArgumentParser(description="Run the HITL deduplication process.")
    parser.add_argument('--batch_size', type=int, default=10, help='Size of each batch for HITL')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for the web server')
    parser.add_argument('--port', type=int, default=5000, help='Port for the web server')
    parser.add_argument('--load_state', type=str, default=None, help='Path to load previous HITL state')
    parser.add_argument('--featpath', type=str, default=None, help='Path to feature file for training')
    parser.add_argument('--dbpath', type=str, default='../database/supreme_court_docs.db', help='Path to the database')
    parser.add_argument('--output_dir', type=str, default='../data/', help='Directory to save output files')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.featpath:
        if not args.featpath.endswith(('.csv', '.zarr')):
            raise ValueError("Feature file must be in CSV or Zarr format.")
        file, filetype = re.match(r"(.+)\.(csv|zarr)$", args.featpath).groups()
        # Load features from a file if provided
        deduplicator, features_df = DbDeduplicator.load_features(file, filetype)
    else:
        deduplicator = DbDeduplicator(args.dbpath)
        deduplicator.blocking()
        deduplicator.compute_similarity_scores()
        # Save features with metadata
        csv_path = output_dir / "features"
        
        # Save in both formats for demonstration
        deduplicator.save_features(str(csv_path), format='csv')

    # Run the HITL process (this will open a web interface)
    hitl_manager = deduplicator.run_hitl_process(
        batch_size=10,
        host='127.0.0.1',
        port=5000,
        load_state=None  # or 'hitl_state.pkl' to resume a previous session
    )

    # After completion, get the final mapping
    mapping_df = deduplicator.apply_confirmed_matches(output_path="final_mapping.csv")
    print(f"Deduplication complete. Merged {len(mapping_df)} entities.")