
def get_custom_logging():
    
    from .custom_logging import get_logger, get_log_filename, log_system_info, PerformanceLogger
    return get_logger, get_log_filename, log_system_info, PerformanceLogger

def get_population_io():
    
    from .population_io import (
        load_and_initialize_population, 
        get_population_files_info, 
        load_population, 
        save_population, 
        sort_population_json, 
        load_genome_by_id,
        consolidate_generations_to_single_file,
        migrate_from_split_to_single,
        sort_population_by_elite_criteria,
        load_elites,
        save_elites,
        get_population_stats_steady_state
    )
    return (
        load_and_initialize_population, 
        get_population_files_info, 
        load_population, 
        save_population, 
        sort_population_json, 
        load_genome_by_id,
        consolidate_generations_to_single_file,
        migrate_from_split_to_single,
        sort_population_by_elite_criteria,
        load_elites,
        save_elites,
        get_population_stats_steady_state
    )

def get_system_utils():
    
    from .population_io import (
        get_project_root,
        get_config_path,
        get_data_path,
        get_outputs_path,
        set_outputs_path,
        _extract_north_star_score,
        initialize_system
    )
    return (
        get_project_root,
        get_config_path,
        get_data_path,
        get_outputs_path,
        _extract_north_star_score,
        initialize_system,
        set_outputs_path,
    )

def get_cluster_quality():
    
    from .cluster_quality import (
        calculate_silhouette_score,
        calculate_davies_bouldin_index,
        calculate_calinski_harabasz_index,
        calculate_cluster_quality_metrics,
        save_cluster_quality_to_tracker
    )
    return (
        calculate_silhouette_score,
        calculate_davies_bouldin_index,
        calculate_calinski_harabasz_index,
        calculate_cluster_quality_metrics,
        save_cluster_quality_to_tracker
    )


__all__ = [
    "get_custom_logging",
    "get_population_io",
    "get_system_utils",
    "get_cluster_quality",
]
