import argparse
import os
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dedupe_hitl.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Deduplication Human-in-the-Loop Management')
    parser.add_argument('--db-path', required=True, help='Path to SQLite database')
    parser.add_argument('--output-dir', help='Output directory for all files (default: beside database)')
    parser.add_argument('--port', type=int, default=5000, help='Port for web interface (default: 5000)')
    parser.add_argument('--host', default='127.0.0.1', help='Host for web interface (default: 127.0.0.1)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for review (default: 10)')
    parser.add_argument('--autosave', type=int, default=5, help='Autosave interval in decisions (default: 5)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a new HITL process')
    start_parser.add_argument('--no-resume', action='store_true', help='Force start new process, ignoring saved state')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export current mapping')
    export_parser.add_argument('--output', required=True, help='Output CSV file path')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate and repair data integrity')
    validate_parser.add_argument('--repair', action='store_true', help='Attempt to repair issues')
    
    args = parser.parse_args()
    
    # Import your modules here to avoid circular imports
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from amici.deduplication.dbdedupe import DbDeduplicator, HITLManager
    
    # Initialize deduplicator
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.db_path), "dedupe_output")
    deduplicator = DbDeduplicator(
        db_path=args.db_path,
        output_dir=output_dir
    )
    
    # Handle commands
    if args.command == 'start':
        # Load existing state if available and not overridden
        resume = not args.no_resume
        logger.info(f"Starting HITL process (resume={resume})")
        
        if resume:
            deduplicator.load_state()
        
        # Run the HITL process
        hitl_manager = deduplicator.run_hitl_process(
            batch_size=args.batch_size,
            host=args.host,
            port=args.port,
            resume=resume
        )
        
    elif args.command == 'export':
        # Load state first
        logger.info("Loading state for export")
        if not deduplicator.load_state():
            logger.error("Failed to load deduplicator state")
            return 1
            
        # Generate mapping
        logger.info(f"Exporting mapping to {args.output}")
        mapping_df = deduplicator.apply_confirmed_matches(output_path=args.output)
        logger.info(f"Exported {len(mapping_df)} entity mappings")
        
    elif args.command == 'validate':
        # Load state first
        logger.info("Loading state for validation")
        if not deduplicator.load_state():
            logger.error("Failed to load deduplicator state")
            return 1
            
        # Create a temporary HITL manager to validate
        hitl_state_path = os.path.join(output_dir, "hitl_state.pkl")
        if os.path.exists(hitl_state_path):
            hitl_manager = HITLManager.load_state(hitl_state_path, deduplicator)
            
            # Validate state
            logger.info("Validating state integrity")
            is_valid = hitl_manager.validate_state()
            
            if is_valid:
                logger.info("Validation successful - no issues found")
            elif args.repair:
                logger.info("Validation found issues - repairs attempted")
                # Save repaired state
                hitl_manager.save_state()
                deduplicator.save_state()
            else:
                logger.warning("Validation found issues - run with --repair to attempt fixes")
        else:
            logger.warning(f"No HITL state found at {hitl_state_path}")
    
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
