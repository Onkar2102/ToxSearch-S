"""
Constants for the ETG framework.
Centralizes magic numbers and configuration values.
"""

class EvolutionConstants:
    """Constants related to evolutionary algorithms"""
    DEFAULT_ELITE_PERCENTAGE = 0.25
    DEFAULT_NORTH_STAR_METRIC = "toxicity"

class SystemConstants:
    """Constants related to system configuration"""
    MAX_RUNTIME_SECONDS = 61200
    HEARTBEAT_INTERVAL = 60
    MAX_MEMORY_GB = 20

class LoggingConstants:
    """Constants related to logging"""
    LOG_MAX_BYTES = 100_000_000
    LOG_BACKUP_COUNT = 10

class ModelConstants:
    """Constants related to model configuration"""
    pass

class FileConstants:
    """Constants related to file operations
    
    Note: Active population = elites.json + reserves.json
    - elites.json: Genomes with species_id > 0 (assigned to species)
    - reserves.json: Cluster 0 outliers (genomes that don't fit existing species)
    - archive.json: Archived/removed genomes (excluded from active population)
    """
    DEFAULT_ELITES_FILE = "data/outputs/elites.json"
    DEFAULT_RESERVES_FILE = "data/outputs/reserves.json"
    DEFAULT_ARCHIVE_FILE = "data/outputs/archive.json"
    DEFAULT_EVOLUTION_TRACKER_FILE = "data/outputs/EvolutionTracker.json"
