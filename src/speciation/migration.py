

import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from .genome_tracker import GenomeTracker
from utils import get_custom_logging
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


def migrate_genome_tracker_from_files(
    elites_path: Optional[Path] = None,
    reserves_path: Optional[Path] = None,
    archive_path: Optional[Path] = None,
    load_archive: bool = True,
    logger=None
) -> Dict[str, Any]:
    
    if logger is None:
        logger = get_logger("Migration")
    
    outputs_path = get_outputs_path()
    
    if elites_path is None or reserves_path is None or archive_path is None:
        if elites_path is None:
            elites_path = outputs_path / "elites.json"
        if reserves_path is None:
            reserves_path = outputs_path / "reserves.json"
        if archive_path is None:
            archive_path = outputs_path / "archive.json"
    
    genome_tracker = GenomeTracker(logger=logger)
    
    tracker_path = outputs_path / "genome_tracker.json"
    if tracker_path.exists():
        logger.info("Genome tracker already exists, loading existing data...")
        genome_tracker.load()
        existing_count = len(genome_tracker.genomes)
        logger.info(f"Loaded existing tracker with {existing_count} genomes")
    else:
        logger.info("Genome tracker does not exist, creating new one...")
        existing_count = 0
    
    elites_genomes = []
    reserves_genomes = []
    archive_genomes = []
    
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
            logger.info(f"Loaded {len(elites_genomes)} genomes from elites.json")
        except Exception as e:
            logger.warning(f"Failed to load elites.json: {e}")
    
    if reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
            logger.info(f"Loaded {len(reserves_genomes)} genomes from reserves.json")
        except Exception as e:
            logger.warning(f"Failed to load reserves.json: {e}")
    
    if load_archive and archive_path.exists():
        try:
            with open(archive_path, 'r', encoding='utf-8') as f:
                archive_genomes = json.load(f)
            logger.info(f"Loaded {len(archive_genomes)} genomes from archive.json")
        except Exception as e:
            logger.warning(f"Failed to load archive.json: {e}")
    
    migrated_count = 0
    updated_count = 0
    skipped_count = 0
    
    for genome in elites_genomes:
        genome_id = genome.get("id")
        species_id = genome.get("species_id")
        
        if not genome_id:
            skipped_count += 1
            continue
        
        if species_id is None or species_id <= 0:
            species_id = 0
        
        genome_id_str = str(genome_id)
        generation = genome.get("generation", 0)
        
        if genome_tracker.exists(genome_id_str):
            old_species_id = genome_tracker.get_species_id(genome_id_str)
            if old_species_id != species_id:
                success, _ = genome_tracker.update_species_id(genome_id_str, species_id, generation, "migration_from_elites")
                if success:
                    updated_count += 1
        else:
            genome_tracker.register(genome_id_str, species_id, generation)
            migrated_count += 1
    
    for genome in reserves_genomes:
        genome_id = genome.get("id")
        species_id = genome.get("species_id", 0)
        
        if not genome_id:
            skipped_count += 1
            continue
        
        if species_id is None or species_id != 0:
            species_id = 0
        
        genome_id_str = str(genome_id)
        generation = genome.get("generation", 0)
        
        if genome_tracker.exists(genome_id_str):
            old_species_id = genome_tracker.get_species_id(genome_id_str)
            if old_species_id != species_id:
                success, _ = genome_tracker.update_species_id(genome_id_str, species_id, generation, "migration_from_reserves")
                if success:
                    updated_count += 1
        else:
            genome_tracker.register(genome_id_str, species_id, generation)
            migrated_count += 1
    
    for genome in archive_genomes:
        genome_id = genome.get("id")
        species_id = -1
        
        if not genome_id:
            skipped_count += 1
            continue
        
        genome_id_str = str(genome_id)
        generation = genome.get("generation", 0)
        
        if genome_tracker.exists(genome_id_str):
            old_species_id = genome_tracker.get_species_id(genome_id_str)
            if old_species_id != species_id:
                success, _ = genome_tracker.update_species_id(genome_id_str, species_id, generation, "migration_from_archive")
                if success:
                    updated_count += 1
        else:
            genome_tracker.register(genome_id_str, species_id, generation)
            migrated_count += 1
    
    if migrated_count > 0 or updated_count > 0:
        genome_tracker.save()
        logger.info(f"Migration complete: {migrated_count} new genomes migrated, {updated_count} existing genomes updated")
    else:
        logger.info("Migration complete: no changes needed (all genomes already in tracker)")
    
    is_consistent, errors = genome_tracker.validate_consistency(
        elites_path, reserves_path, archive_path, load_archive=load_archive
    )
    
    stats = {
        "migrated": migrated_count,
        "updated": updated_count,
        "skipped": skipped_count,
        "total_in_tracker": len(genome_tracker.genomes),
        "existing_before": existing_count,
        "validation_consistent": is_consistent,
        "validation_errors": len(errors) if not is_consistent else 0
    }
    
    if not is_consistent:
        logger.warning(f"Migration validation found {len(errors)} inconsistencies:")
        for error in errors[:10]:
            logger.warning(f"  - {error}")
    else:
        logger.info("Migration validation passed - all genomes consistent")
    
    return stats


def auto_migrate_if_needed(logger=None) -> bool:
    
    if logger is None:
        logger = get_logger("Migration")
    
    outputs_path = get_outputs_path()
    tracker_path = outputs_path / "genome_tracker.json"
    
    if tracker_path.exists():
        try:
            genome_tracker = GenomeTracker(logger=logger)
            genome_tracker.load()
            if len(genome_tracker.genomes) > 0:
                logger.debug("Genome tracker already exists with data, skipping migration")
                return False
        except Exception:
            pass
    
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    has_source_files = (elites_path.exists() or reserves_path.exists() or archive_path.exists())
    
    if not has_source_files:
        logger.debug("No source files found for migration, starting with empty tracker")
        return False
    
    logger.info("Auto-migrating genome tracker from existing files...")
    stats = migrate_genome_tracker_from_files(
        elites_path=elites_path,
        reserves_path=reserves_path,
        archive_path=archive_path,
        load_archive=True,
        logger=logger
    )
    
    logger.info(f"Auto-migration complete: {stats['migrated']} migrated, {stats['updated']} updated")
    return True
