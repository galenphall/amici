# Example script to run the HITL process
import sys
from pathlib import Path
import logging
import argparse
import re
import os

# Add parent directory to path to allow relative imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.deduplication.dbdedupe import DbDeduplicator, HITLManager, HITLGui


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
    parser.add_argument('--no_resume', action='store_true', help='Force start new process, ignoring saved state')
    parser.add_argument('--autosave_interval', type=int, default=5, help='Auto-save after this many decisions')
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
        # Initialize deduplicator with output directory for persistence
        deduplicator = DbDeduplicator(
            args.dbpath, 
            output_dir=str(output_dir)
        )
        
        # Check if we should resume from existing state
        resume = not args.no_resume
        if resume:
            logger.info("Attempting to load saved state")
            if deduplicator.load_state():
                logger.info("Successfully loaded saved state")
            else:
                logger.info("No saved state found or loading failed, computing features")
                deduplicator.blocking()
                deduplicator.compute_similarity_scores()
                # Save features with metadata
                deduplicator.save_state()
        else:
            logger.info("Computing new features (ignoring any saved state)")
            deduplicator.blocking()
            deduplicator.compute_similarity_scores()
            # Save initial state
            deduplicator.save_state()

    # Run the HITL process (this will open a web interface)
    hitl_state_path = os.path.join(args.output_dir, "hitl_state.pkl") if args.load_state is None else args.load_state
    
    if not args.no_resume and os.path.exists(hitl_state_path):
        # Load previous HITL state if available and resume is not disabled
        logger.info(f"Loading HITL state from {hitl_state_path}")
        hitl_manager = HITLManager.load_state(hitl_state_path, deduplicator)
    else:
        # Initialize the HITL manager with the deduplicator
        logger.info("Creating new HITL manager")
        hitl_manager = HITLManager(
            deduplicator, 
            save_path=hitl_state_path,
            batch_size=args.batch_size,
            autosave_interval=args.autosave_interval
        )
        hitl_manager.initialize_candidates()
        hitl_manager.save_state()  # Initial save
    
    # Setup automatic save on exit
    import atexit
    def save_on_exit():
        logger.info("Process ending, saving final state")
        hitl_manager.save_state()
        deduplicator.save_state()
    atexit.register(save_on_exit)
        
    gui = HITLGui(hitl_manager)

    app = gui._create_app()
    
    # Then run it with the guard
    app.run(host=args.host, port=args.port, debug=False)

    # After completion, get the final mapping
    mapping_df = deduplicator.apply_confirmed_matches(output_path=os.path.join(args.output_dir, "final_mapping.csv"))
    print(f"Deduplication complete. Merged {len(mapping_df)} entities.")