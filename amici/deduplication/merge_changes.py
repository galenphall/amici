#!/usr/bin/env python3
"""
Script to update amicus names in database based on deduplication mapping CSV.
Usage: python update_amicus_names.py <mapping_csv_path> <database_path>
"""

import sqlite3
import csv
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

sys.path.append(str(Path(__file__).parent.parent.parent))
from amici.utils.normalizers import normalize_interest_group_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AmicusNameUpdater:
    def __init__(self, db_path: str, mapping_csv_path: str):
        self.db_path = db_path
        self.mapping_csv_path = mapping_csv_path
        self.mapping: Dict[str, str] = {}
        self.changes_log: List[Tuple[int, str, str]] = []
        
    def backup_database(self):
        """Create a backup of the database before making changes."""
        backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Creating database backup at: {backup_path}")
        shutil.copy2(self.db_path, backup_path)
        return backup_path
    
    def load_mapping(self):
        """Load the deduplication mapping from CSV."""
        logger.info(f"Loading mapping from: {self.mapping_csv_path}")
        
        try:
            with open(self.mapping_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Verify required columns exist
                if 'original_name' not in reader.fieldnames or 'mapped_name' not in reader.fieldnames:
                    raise ValueError("CSV must contain 'original_name' and 'mapped_name' columns")
                
                for row in reader:
                    original = normalize_interest_group_name(row['original_name'].strip())
                    mapped = normalize_interest_group_name(row['mapped_name'].strip())
                    
                    if original and mapped:
                        self.mapping[original] = mapped
                        
            logger.info(f"Loaded {len(self.mapping)} name mappings")
            
        except Exception as e:
            logger.error(f"Error loading mapping CSV: {e}")
            raise
    
    def get_amici_to_update(self, conn: sqlite3.Connection) -> List[Tuple[int, str]]:
        """Get all amici that need updating based on the mapping."""
        cursor = conn.cursor()
        
        # Get all amici
        cursor.execute("SELECT amicus_id, name FROM amici")
        all_amici = cursor.fetchall()
        
        amici_to_update = []
        
        for amicus_id, current_name in all_amici:
            # Normalize the current name
            normalized_name = normalize_interest_group_name(current_name)
            
            # Check if this normalized name exists in our mapping
            if normalized_name in self.mapping:
                new_name = normalize_interest_group_name(self.mapping[normalized_name])
                
                # Only update if the names are actually different
                if current_name != new_name:
                    amici_to_update.append((amicus_id, current_name, new_name))
                    
        logger.info(f"Found {len(amici_to_update)} amici names to update")
        return amici_to_update
    
    def update_amici_names(self, conn: sqlite3.Connection, amici_to_update: List[Tuple[int, str, str]]):
        """Update the amici names in the database."""
        cursor = conn.cursor()
        
        try:
            # Check if merged_name column exists, if not add it
            cursor.execute("PRAGMA table_info(amici)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'merged_name' not in columns:
                logger.info("Adding merged_name column to amici table")
                cursor.execute("ALTER TABLE amici ADD COLUMN merged_name TEXT")
                # Populate merged_name column with existing name values
                cursor.execute("UPDATE amici SET merged_name = name WHERE merged_name IS NULL")
                conn.commit()

            # Normalize *all* merged names
            cursor.execute("SELECT DISTINCT merged_name FROM amici WHERE merged_name IS NOT NULL")
            all_merged_names = cursor.fetchall()

            for (merged_name,) in all_merged_names:
                normalized_merged = normalize_interest_group_name(merged_name)
                if normalized_merged != merged_name:
                    cursor.execute(
                        "UPDATE amici SET merged_name = ? WHERE merged_name = ?",
                        (normalized_merged, merged_name)
                    )

            # Also normalize all merged names in amici_embeddings
            cursor.execute("SELECT DISTINCT merged_name FROM amici_embeddings WHERE merged_name IS NOT NULL")
            all_embedding_merged_names = cursor.fetchall()

            for (merged_name,) in all_embedding_merged_names:
                normalized_merged = normalize_interest_group_name(merged_name)
                if normalized_merged != merged_name:
                    cursor.execute(
                        "UPDATE amici_embeddings SET merged_name = ? WHERE merged_name = ?",
                        (normalized_merged, merged_name)
                    )

            conn.commit()
            logger.info("Normalized all existing merged_name values")

            # Update amici table
            for amicus_id, old_name, new_name in amici_to_update:
                cursor.execute(
                    "UPDATE amici SET merged_name = ? WHERE amicus_id = ?",
                    (new_name, amicus_id)
                )
                self.changes_log.append((amicus_id, old_name, new_name))
                logger.debug(f"Updated amicus_id {amicus_id}: '{old_name}' -> '{new_name}'")
            
            # Also update amici_embeddings table if the name exists there
            # Check if merged_name column exists in amici_embeddings, if not add it
            cursor.execute("PRAGMA table_info(amici_embeddings)")
            embedding_columns = [row[1] for row in cursor.fetchall()]
            if 'merged_name' not in embedding_columns:
                logger.info("Adding merged_name column to amici_embeddings table")
                cursor.execute("ALTER TABLE amici_embeddings ADD COLUMN merged_name TEXT")
                # Populate merged_name column with existing name values
                cursor.execute("UPDATE amici_embeddings SET merged_name = name WHERE merged_name IS NULL")
                conn.commit()

            unique_name_changes = {}
            for _, old_name, new_name in amici_to_update:
                normalized_old = normalize_interest_group_name(old_name)
                if normalized_old not in unique_name_changes:
                    unique_name_changes[normalized_old] = new_name
            
            embeddings_updated = 0
            for normalized_old, new_name in unique_name_changes.items():
                # Check if the old name exists in amici_embeddings
                cursor.execute(
                    "SELECT COUNT(*) FROM amici_embeddings WHERE name = ?",
                    (normalized_old,)
                )
                
                if cursor.fetchone()[0] > 0:
                    cursor.execute(
                        "UPDATE amici_embeddings SET merged_name = ? WHERE name = ?",
                        (new_name, normalized_old)
                    )
                    embeddings_updated += 1
            
            logger.info(f"Updated {embeddings_updated} entries in amici_embeddings table")
            
            conn.commit()
            logger.info(f"Successfully updated {len(amici_to_update)} amicus names")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating names: {e}")
            raise
    
    def write_changes_log(self):
        """Write a log of all changes made."""
        log_path = f"amicus_name_updates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['amicus_id', 'old_name', 'new_name'])
            writer.writerows(self.changes_log)
            
        logger.info(f"Changes log written to: {log_path}")
        return log_path
    
    def run(self):
        """Execute the full update process."""
        logger.info("Starting amicus name deduplication update")
        
        # Step 1: Backup database
        backup_path = self.backup_database()
        
        try:
            # Step 2: Load mapping
            self.load_mapping()
            
            # Step 3: Connect to database and perform updates
            with sqlite3.connect(self.db_path) as conn:
                # Get amici to update
                amici_to_update = self.get_amici_to_update(conn)
                
                if amici_to_update:
                    # Perform updates
                    self.update_amici_names(conn, amici_to_update)
                    
                    # Write changes log
                    log_path = self.write_changes_log()
                    
                    logger.info(f"Update completed successfully!")
                    logger.info(f"Backup saved at: {backup_path}")
                    logger.info(f"Changes log saved at: {log_path}")
                else:
                    logger.info("No amici names needed updating")
                    
        except Exception as e:
            logger.error(f"Error during update process: {e}")
            logger.error(f"Database backup available at: {backup_path}")
            raise


def main():
    if len(sys.argv) != 3:
        print("Usage: python update_amicus_names.py <mapping_csv_path> <database_path>")
        sys.exit(1)
    
    mapping_csv_path = sys.argv[1]
    database_path = sys.argv[2]
    
    # Validate inputs
    if not Path(mapping_csv_path).exists():
        logger.error(f"Mapping CSV file not found: {mapping_csv_path}")
        sys.exit(1)
        
    if not Path(database_path).exists():
        logger.error(f"Database file not found: {database_path}")
        sys.exit(1)
    
    # Run the updater
    updater = AmicusNameUpdater(database_path, mapping_csv_path)
    
    try:
        updater.run()
    except Exception as e:
        logger.error(f"Update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()