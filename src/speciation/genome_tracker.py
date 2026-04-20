

import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


class GenomeTracker:
    """Master registry of all genomes and their current species_id. Single source of truth for genome distribution: - species_id > 0: In elites.json (belongs to a species) - species_id == 0: In reserves.json (cluster 0 / reserves) - species_id == -1: In archive.json (archived) The tracker is authoritative - if files show different species_id, the tracker's value is correct and files should be updated."""
    
    def __init__(self, logger=None):
        
        self.genomes: Dict[str, Dict[str, Any]] = {}
        self.logger = logger or get_logger("GenomeTracker")
        self._dirty = False
    
    def register(self, genome_id: Union[str, int], species_id: int, generation: int) -> bool:
        
        genome_id = str(genome_id)
        try:
            if genome_id in self.genomes:
                self.genomes[genome_id]["species_id"] = species_id
                self.genomes[genome_id]["last_updated_generation"] = generation
                self.genomes[genome_id]["last_updated_timestamp"] = datetime.now().isoformat()
            else:
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
        
        genome_id = str(genome_id)
        if genome_id not in self.genomes:
            self.logger.debug("Genome %s not found in tracker, registering as new (expected for new variants)", genome_id)
            success = self.register(genome_id, new_species_id, generation)
            reassignment_info = None
            if success and new_species_id > 0:
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
            
            reassignment_info = None
            if old_species_id == -1 and new_species_id > 0:
                reassignment_info = (genome_id, old_species_id, new_species_id)
                self.logger.debug(f"Genome {genome_id} reassigned from archive to species {new_species_id}")
            
            return True, reassignment_info
        except Exception as e:
            self.logger.error(f"Failed to update genome {genome_id}: {e}")
            return False, None
    
    def batch_update(self, updates: Dict[Union[str, int], int], generation: int, reason: Optional[str] = None) -> Dict[str, Any]:
        
        updates = {str(k): v for k, v in updates.items()}
        total = len(updates)
        succeeded = []
        failed = []
        errors = []
        reassigned_from_archive = []
        
        for genome_id, new_species_id in updates.items():
            try:
                success, reassignment_info = self.update_species_id(genome_id, new_species_id, generation, reason)
                if success:
                    succeeded.append(genome_id)
                    if reassignment_info:
                        reassigned_from_archive.append(reassignment_info)
                else:
                    failed.append(genome_id)
                    errors.append(f"Failed to update {genome_id}")
            except Exception as e:
                failed.append(genome_id)
                errors.append(f"Error updating {genome_id}: {str(e)}")
        
        if failed:
            self.logger.warning(f"Batch update partially failed: {len(failed)}/{total} failed, retrying individually with exponential backoff")
            import time
            max_retries = 3
            retry_failed = failed.copy()
            
            for retry_num in range(max_retries):
                if not retry_failed:
                    break
                
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
                            if reassignment_info:
                                reassigned_from_archive.append(reassignment_info)
                        else:
                            failed_this_round.append(genome_id)
                            errors.append(f"Retry {retry_num + 1} failed for {genome_id}")
                    except Exception as e:
                        failed_this_round.append(genome_id)
                        errors.append(f"Retry {retry_num + 1} error for {genome_id}: {str(e)}")
                
                retry_failed = failed_this_round
            
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
        
        genome_id = str(genome_id)
        if genome_id in self.genomes:
            return self.genomes[genome_id]["species_id"]
        return None
    
    def get_all_genomes_by_species(self, species_id: Union[int, str]) -> List[str]:
        
        species_id = int(species_id)
        return [
            genome_id for genome_id, data in self.genomes.items()
            if data["species_id"] == species_id
        ]
    
    def exists(self, genome_id: Union[str, int]) -> bool:
        
        genome_id = str(genome_id)
        return genome_id in self.genomes
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        
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
        
        errors = []
        seen_ids = set()
        
        for genome_id, data in self.genomes.items():
            if genome_id in seen_ids:
                errors.append(f"Duplicate genome ID found in tracker: {genome_id}")
            seen_ids.add(genome_id)
            
            if "species_id" not in data:
                errors.append(f"Genome {genome_id} missing required field 'species_id'")
                continue
            
            species_id = data["species_id"]
            if not isinstance(species_id, int):
                errors.append(f"Genome {genome_id} has invalid species_id type: {type(species_id)}, expected int")
            elif species_id < -1:
                errors.append(f"Genome {genome_id} has invalid species_id value: {species_id} (must be -1, 0, or >0)")
            
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
        
        errors = []
        
        is_internal_valid, internal_errors = self.validate_internal_state()
        if not is_internal_valid:
            errors.extend(internal_errors)
        
        stats = self.get_distribution_stats()
        has_archived = int(stats["by_species_id"].get("-1", 0)) > 0
        
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
        
        all_file_genomes = {}
        for g in elites_genomes:
            gid = g.get("id")
            if gid is not None and gid != "":
                all_file_genomes[str(gid)] = ("elites", g.get("species_id"))
        
        for g in reserves_genomes:
            gid = g.get("id")
            if gid is not None and gid != "":
                all_file_genomes[str(gid)] = ("reserves", g.get("species_id", 0))
        
        for g in archive_genomes:
            gid = g.get("id")
            if gid is not None and gid != "":
                all_file_genomes[str(gid)] = ("archive", g.get("species_id", -1))
        
        for genome_id, data in self.genomes.items():
            tracker_species_id = data["species_id"]
            
            if genome_id in all_file_genomes:
                file_location, file_species_id = all_file_genomes[genome_id]
                
                if file_species_id != tracker_species_id:
                    errors.append(
                        f"Genome {genome_id}: tracker says species_id={tracker_species_id}, "
                        f"but {file_location}.json shows {file_species_id} (tracker is authoritative)"
                    )
                
                expected_location = "elites" if tracker_species_id > 0 else ("reserves" if tracker_species_id == 0 else "archive")
                if file_location != expected_location:
                    errors.append(
                        f"Genome {genome_id}: tracker says species_id={tracker_species_id} (should be in {expected_location}.json), "
                        f"but found in {file_location}.json"
                    )
            else:
                pass
        
        tracked_ids = set(self.genomes.keys())
        for genome_id, (file_location, _) in all_file_genomes.items():
            if genome_id not in tracked_ids:
                pass
        
        is_consistent = len(errors) == 0
        return is_consistent, errors
    
    def load(self, path: Optional[str] = None) -> bool:
        
        from utils import get_system_utils
        _, _, _, get_outputs_path, _, _, _ = get_system_utils()
        
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
            
            if "genomes" in data:
                self.genomes = data["genomes"]
                self._dirty = False
                self.logger.info(f"Loaded genome tracker with {len(self.genomes)} genomes from {path}")
                return True
            else:
                self.logger.warning(f"Genome tracker file has unexpected format, starting with empty tracker")
                self.genomes = {}
                self._dirty = False
                return True
        except Exception as e:
            self.logger.error(f"Failed to load genome tracker from {path}: {e}", exc_info=True)
            return False
    
    def save(self, path: Optional[str] = None, backup: bool = True) -> bool:
        
        from utils import get_system_utils
        _, _, _, get_outputs_path, _, _, _ = get_system_utils()
        
        if path is None:
            outputs_path = get_outputs_path()
            path = str(outputs_path / "genome_tracker.json")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if backup and path_obj.exists():
            backup_path = path_obj.with_suffix('.json.backup')
            try:
                shutil.copy2(path_obj, backup_path)
                self.logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                self.logger.warning(f"Failed to create backup: {e}")
        
        try:
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
            
            temp_path = path_obj.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            temp_path.replace(path_obj)
            
            self._dirty = False
            self.logger.info(f"Saved genome tracker with {len(self.genomes)} genomes to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save genome tracker to {path}: {e}", exc_info=True)
            return False
