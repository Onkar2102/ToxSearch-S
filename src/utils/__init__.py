
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
        sort_population_by_elite_criteria,
        load_elites,
        save_elites,
    )
    return (
        load_and_initialize_population,
        get_population_files_info,
        load_population,
        save_population,
        sort_population_json,
        sort_population_by_elite_criteria,
        load_elites,
        save_elites,
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


__all__ = [
    "get_custom_logging",
    "get_population_io",
    "get_system_utils",
]
