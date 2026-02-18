"""
genome_tracker.py

Master registry of all genomes and their current species_id.
Single source of truth for genome distribution in the speciation pipeline.

Genome tracker maintains:
- species_id > 0: Genome belongs to a species (in elites.json)
- species_id == 0: Genome is in reserves (in reserves.json)
- species_id == -1: Genome is archived (in archive.json)
"""

import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class GenomeTracker:
    """
    Master registry of all genomes and their current species_id.
    
    Single source of truth for genome distribution:
    - species_id > 0: In elites.json (belongs to a species)
    - species_id == 0: In reserves.json (cluster 0 / reserves)
    - species_id == -1: In archive.json (archived)
    
    The tracker is authoritative - if files show different species_id,
    the tracker's value is correct and files should be updated.
    """
    
    def __init__(self, logger=None):
        """
        Initialize genome tracker.
        
        Args:
            logger: Optional logger instance
        """
        self.genomes: Dict[str, Dict[str, Any]] = {}
        self.logger = logger or get_logger("GenomeTracker")
        self._dirty = False  # Track if tracker has unsaved changes
    
    def register(self, genome_id: Union[str, int], species_id: int, generation: int) -> bool:
        """
        Register a new genome or update existing genome.
        
        Args:
            genome_id: Unique genome identifier (str or int, converted to str internally)
            species_id: Species ID (>0 for species, 0 for reserves, -1 for archived)
            generation: Generation number
            
        Returns:
            True if registered successfully, False otherwise
        """
        # Convert to string to ensure type consistency (accepts both str and int)
        genome_id = str(genome_id)
        try:
            if genome_id in self.genomes:
                # Update existing
                self.genomes[genome_id]["species_id"] = species_id
                self.genomes[genome_id]["last_updated_generation"] = generation
                self.genomes[genome_id]["last_updated_timestamp"] = datetime.now().isoformat()
            else:
                # Register new
                self.genomes[genome_id] = {
                    "species_id": species_id,
                    "created_generation": generation,
                    "last_updated_generation": generation,
                    "last_updated_timestamp": datetime.now().isoformat()
                }
            self._dirty = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to register genome {genome_id}: {e}")
            return False
    
    def update_species_id(self, genome_id: Union[str, int], new_species_id: int, generation: int, reason: Optional[str] = None) -> Tuple[bool, Optional[Tuple[str, int, int]]]:
        """
        Update a genome's species_id.
        
        Args:
            genome_id: Unique genome identifier (str or int, converted to str internally)
            new_species_id: New species ID
            generation: Generation number
            reason: Optional reason for update (for logging)
            
        Returns:
            Tuple of (success: bool, reassignment_info: Optional[Tuple[genome_id, old_sid, new_sid]])
            - success: True if updated successfully, False otherwise
            - reassignment_info: If genome was reassigned from archive (-1) to active species (>0), returns (genome_id, old_sid, new_sid), else None.
              In the current pipeline, archive genomes are not moved back to active species; reassignment_info is set when such an update is applied (e.g. by an external caller or future feature).
        """
        # Convert to string to ensure type consistency (accepts both str and int)
        genome_id = str(genome_id)
        if genome_id not in self.genomes:
            self.logger.warning(f"Genome {genome_id} not found in tracker, registering as new")
            success = self.register(genome_id, new_species_id, generation)
            # New registration from archive to active is also a reassignment
            reassignment_info = None
            if success and new_species_id > 0:
                # New genome registered as active (not from archive, so no reassignment)
                pass
            return success, reassignment_info
        
        try:
            old_species_id = self.genomes[genome_id]["species_id"]
            self.genomes[genome_id]["species_id"] = new_species_id
            self.genomes[genome_id]["last_updated_generation"] = generation
            self.genomes[genome_id]["last_updated_timestamp"] = datetime.now().isoformat()
            self._dirty = True
            
            if reason:
                self.logger.debug(f"Genome {genome_id}: {old_species_id} → {new_species_id} ({reason})")
            
            # Check if this is a reassignment from archive (-1) to active species (>0)
            reassignment_info = None
            if old_species_id == -1 and new_species_id > 0:
                reassignment_info = (genome_id, old_species_id, new_species_id)
                self.logger.debug(f"Genome {genome_id} reassigned from archive to species {new_species_id}")
            
            return True, reassignment_info
        except Exception as e:
            self.logger.error(f"Failed to update genome {genome_id}: {e}")
            return False, None
    
    def batch_update(self, updates: Dict[Union[str, int], int], generation: int, reason: Optional[str] = None) -> Dict[str, Any]:
        """
        Update multiple genomes atomically with partial failure handling.
        
        Args:
            updates: Dictionary mapping genome_id (str or int) to new species_id
            generation: Generation number
            reason: Optional reason for batch update (for logging)
            
        Returns:
            Dictionary with results: {"total": N, "succeeded": M, "failed": K, "partial": bool, 
                                     "failed_genome_ids": [...], "errors": [...],
                                     "reassigned_from_archive": [(genome_id, old_sid, new_sid), ...]}.
            In the current design, archive genomes are not moved back; reassigned_from_archive supports possible future use or external updates.
        """
        # Convert all keys to strings for consistency (accepts both str and int keys)
        updates = {str(k): v for k, v in updates.items()}
        total = len(updates)
        succeeded = []
        failed = []
        errors = []
        reassigned_from_archive = []  # Track genomes reassigned from archived state
        
        # Attempt batch update
        for genome_id, new_species_id in updates.items():
            try:
                success, reassignment_info = self.update_species_id(genome_id, new_species_id, generation, reason)
                if success:
                    succeeded.append(genome_id)
                    # Track reassignment from archive (-1) to active species (>0)
                    if reassignment_info:
                        reassigned_from_archive.append(reassignment_info)
                else:
                    failed.append(genome_id)
                    errors.append(f"Failed to update {genome_id}")
            except Exception as e:
                failed.append(genome_id)
                errors.append(f"Error updating {genome_id}: {str(e)}")
        
        # If partial failure, retry failed items individually with exponential backoff
        if failed:
            self.logger.warning(f"Batch update partially failed: {len(failed)}/{total} failed, retrying individually with exponential backoff")
            import time
            max_retries = 3
            retry_failed = failed.copy()
            
            for retry_num in range(max_retries):
                if not retry_failed:
                    break
                
                # Exponential backoff: 0.1s, 0.2s, 0.4s
                if retry_num > 0:
                    time.sleep(0.1 * (2 ** (retry_num - 1)))
                
                failed_this_round = []
                for genome_id in retry_failed:
                    try:
                        new_species_id = updates[genome_id]
                        success, reassignment_info = self.update_species_id(
                            genome_id, new_species_id, generation, 
                            f"{reason}_retry_{retry_num + 1}" if reason else f"retry_{retry_num + 1}"
                        )
                        if success:
                            succeeded.append(genome_id)
                            # Track reassignment from archive (-1) to active species (>0) in retry
                            if reassignment_info:
                                reassigned_from_archive.append(reassignment_info)
                        else:
                            failed_this_round.append(genome_id)
                            errors.append(f"Retry {retry_num + 1} failed for {genome_id}")
                    except Exception as e:
                        failed_this_round.append(genome_id)
                        errors.append(f"Retry {retry_num + 1} error for {genome_id}: {str(e)}")
                
                retry_failed = failed_this_round
            
            # Update failed list with any remaining failures after all retries
            failed = retry_failed
            if failed:
                self.logger.error(f"Batch update failed for {len(failed)} genomes after {max_retries} retries: {failed[:5]}{'...' if len(failed) > 5 else ''}")
        
        result = {
            "total": total,
            "succeeded": len(succeeded),
            "failed": len(failed),
            "partial": len(failed) > 0 and len(succeeded) > 0,
            "failed_genome_ids": failed,
            "errors": errors,
            "reassigned_from_archive": reassigned_from_archive
        }
        
        if reason:
            self.logger.info(f"Batch update ({reason}): {result['succeeded']}/{result['total']} succeeded")
        
        if failed:
            self.logger.warning(f"Batch update failed for {len(failed)} genomes: {failed[:5]}{'...' if len(failed) > 5 else ''}")
        
        if reassigned_from_archive:
            self.logger.info(f"Batch update: {len(reassigned_from_archive)} genomes reassigned from archive to active species")
        
        return result
    
    def get_species_id(self, genome_id: Union[str, int]) -> Optional[int]:
        """
        Get current species_id for a genome.
        
        Args:
            genome_id: Unique genome identifier (str or int, converted to str internally)
            
        Returns:
            species_id if found, None otherwise
        """
        # Convert to string to ensure type consistency (accepts both str and int)
        genome_id = str(genome_id)
        if genome_id in self.genomes:
            return self.genomes[genome_id]["species_id"]
        return None
    
    def get_all_genomes_by_species(self, species_id: Union[int, str]) -> List[str]:
        """
        Get all genome IDs for a species.
        
        Args:
            species_id: Species ID to query (int or str; normalized to int for comparison)
            
        Returns:
            List of genome IDs belonging to this species
        """
        species_id = int(species_id)
        return [
            genome_id for genome_id, data in self.genomes.items()
            if data["species_id"] == species_id
        ]
    
    def exists(self, genome_id: Union[str, int]) -> bool:
        """
        Check if a genome is tracked.
        
        Args:
            genome_id: Unique genome identifier (str or int, converted to str internally)
            
        Returns:
            True if genome is tracked, False otherwise
        """
        # Convert to string to ensure type consistency (accepts both str and int)
        genome_id = str(genome_id)
        return genome_id in self.genomes
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """
        Get distribution statistics by species_id.
        
        Returns:
            Dictionary with counts by species_id and total genomes
        """
        by_species_id: Dict[int, int] = {}
        total = len(self.genomes)
        
        for data in self.genomes.values():
            species_id = data["species_id"]
            by_species_id[species_id] = by_species_id.get(species_id, 0) + 1
        
        return {
            "total_genomes": total,
            "by_species_id": {str(k): v for k, v in by_species_id.items()},
            "last_updated": datetime.now().isoformat()
        }
    
    def validate_internal_state(self) -> Tuple[bool, List[str]]:
        """
        Validate internal tracker state consistency.
        
        Checks:
        - No duplicate genome IDs
        - All species_id values are valid (-1, 0, or >0)
        - Generation numbers are reasonable (non-negative)
        - Required fields exist for each genome
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        seen_ids = set()
        
        for genome_id, data in self.genomes.items():
            # Check for duplicate genome IDs (should not happen due to dict key uniqueness, but verify)
            if genome_id in seen_ids:
                errors.append(f"Duplicate genome ID found in tracker: {genome_id}")
            seen_ids.add(genome_id)
            
            # Check required fields exist
            if "species_id" not in data:
                errors.append(f"Genome {genome_id} missing required field 'species_id'")
                continue
            
            # Check species_id is valid (-1, 0, or >0)
            species_id = data["species_id"]
            if not isinstance(species_id, int):
                errors.append(f"Genome {genome_id} has invalid species_id type: {type(species_id)}, expected int")
            elif species_id < -1:
                errors.append(f"Genome {genome_id} has invalid species_id value: {species_id} (must be -1, 0, or >0)")
            
            # Check generation numbers are reasonable
            if "created_generation" in data:
                created_gen = data["created_generation"]
                if not isinstance(created_gen, int) or created_gen < 0:
                    errors.append(f"Genome {genome_id} has invalid created_generation: {created_gen}")
            
            if "last_updated_generation" in data:
                updated_gen = data["last_updated_generation"]
                if not isinstance(updated_gen, int) or updated_gen < 0:
                    errors.append(f"Genome {genome_id} has invalid last_updated_generation: {updated_gen}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_consistency(self, elites_path: Path, reserves_path: Path, archive_path: Path, 
                            load_archive: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate consistency between tracker and files.
        
        Args:
            elites_path: Path to elites.json
            reserves_path: Path to reserves.json
            archive_path: Path to archive.json
            load_archive: Whether to load archive.json (lazy loading)
            
        Returns:
            Tuple of (is_consistent, list_of_errors)
        """
        errors = []
        
        # First validate internal state
        is_internal_valid, internal_errors = self.validate_internal_state()
        if not is_internal_valid:
            errors.extend(internal_errors)
        
        # Check if we need to load archive
        stats = self.get_distribution_stats()
        has_archived = int(stats["by_species_id"].get("-1", 0)) > 0
        
        # Load files
        elites_genomes = []
        reserves_genomes = []
        archive_genomes = []
        
        if elites_path.exists():
            try:
                with open(elites_path, 'r', encoding='utf-8') as f:
                    elites_genomes = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load elites.json: {e}")
        
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load reserves.json: {e}")
        
        if load_archive and has_archived and archive_path.exists():
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_genomes = json.load(f)
            except Exception as e:
                errors.append(f"Failed to load archive.json: {e}")
        
        # Validate tracker vs files
        all_file_genomes = {}
        for g in elites_genomes:
            gid = g.get("id")
            if gid:
                all_file_genomes[gid] = ("elites", g.get("species_id"))
        
        for g in reserves_genomes:
            gid = g.get("id")
            if gid:
                all_file_genomes[gid] = ("reserves", g.get("species_id", 0))
        
        for g in archive_genomes:
            gid = g.get("id")
            if gid:
                all_file_genomes[gid] = ("archive", g.get("species_id", -1))
        
        # Check each genome in tracker
        for genome_id, data in self.genomes.items():
            tracker_species_id = data["species_id"]
            
            if genome_id in all_file_genomes:
                file_location, file_species_id = all_file_genomes[genome_id]
                
                # Check if species_id matches
                if file_species_id != tracker_species_id:
                    errors.append(
                        f"Genome {genome_id}: tracker says species_id={tracker_species_id}, "
                        f"but {file_location}.json shows {file_species_id} (tracker is authoritative)"
                    )
                
                # Check if genome is in correct file
                expected_location = "elites" if tracker_species_id > 0 else ("reserves" if tracker_species_id == 0 else "archive")
                if file_location != expected_location:
                    errors.append(
                        f"Genome {genome_id}: tracker says species_id={tracker_species_id} (should be in {expected_location}.json), "
                        f"but found in {file_location}.json"
                    )
            else:
                # Genome in tracker but not in files (might be in temp.json or newly registered)
                pass
        
        # Check for orphaned genomes (in files but not in tracker)
        tracked_ids = set(self.genomes.keys())
        for genome_id, (file_location, _) in all_file_genomes.items():
            if genome_id not in tracked_ids:
                # This is okay - might be from previous generations or edge cases
                pass
        
        is_consistent = len(errors) == 0
        return is_consistent, errors
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load tracker from file.
        
        Args:
            path: Optional path to load from. If None, uses default outputs_path / "genome_tracker.json"
            
        Returns:
            True if loaded successfully, False otherwise
        """
        from utils import get_system_utils
        _, _, _, get_outputs_path, _, _ = get_system_utils()
        
        if path is None:
            outputs_path = get_outputs_path()
            path = str(outputs_path / "genome_tracker.json")
        
        path_obj = Path(path)
        
        if not path_obj.exists():
            self.logger.info(f"Genome tracker file not found at {path}, starting with empty tracker")
            self.genomes = {}
            self._dirty = False
            return True
        
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle version 2.0 format
            if "genomes" in data:
                self.genomes = data["genomes"]
                self._dirty = False
                self.logger.info(f"Loaded genome tracker with {len(self.genomes)} genomes from {path}")
                return True
            else:
                # Old format or empty file
                self.logger.warning(f"Genome tracker file has unexpected format, starting with empty tracker")
                self.genomes = {}
                self._dirty = False
                return True
        except Exception as e:
            self.logger.error(f"Failed to load genome tracker from {path}: {e}", exc_info=True)
            return False
    
    def save(self, path: Optional[str] = None, backup: bool = True) -> bool:
        """
        Save tracker to file.
        
        Args:
            path: Optional path to save to. If None, uses default outputs_path / "genome_tracker.json"
            backup: Whether to create backup before saving
            
        Returns:
            True if saved successfully, False otherwise
        """
        from utils import get_system_utils
        _, _, _, get_outputs_path, _, _ = get_system_utils()
        
        if path is None:
            outputs_path = get_outputs_path()
            path = str(outputs_path / "genome_tracker.json")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if requested and file exists
        if backup and path_obj.exists():
            backup_path = path_obj.with_suffix('.json.backup')
            try:
                shutil.copy2(path_obj, backup_path)
                self.logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                self.logger.warning(f"Failed to create backup: {e}")
        
        try:
            # Prepare data structure
            stats = self.get_distribution_stats()
            data = {
                "version": "2.0",
                "genomes": self.genomes,
                "summary": stats,
                "metadata": {
                    "last_updated": datetime.now().isoformat(),
                    "total_genomes": len(self.genomes)
                }
            }
            
            # Write to temporary file first, then rename (atomic write)
            temp_path = path_obj.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(path_obj)
            
            self._dirty = False
            self.logger.info(f"Saved genome tracker with {len(self.genomes)} genomes to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save genome tracker to {path}: {e}", exc_info=True)
            return False
