"""
operator_effectiveness.py

Calculate operator effectiveness metrics (RQ1) for operator-level analysis.
Metrics: NE, EHR, IR, cEHR, Δμ, Δσ

Based on ToxSearch paper: "Which prompt perturbation strategies are most effective?"
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from utils import get_custom_logging, get_system_utils
from utils.population_io import _extract_north_star_score

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()


# get_expected_variant_count() was removed; expected_variant_count is not used for metric calculations.
# Metrics use calculated_total from operator statistics instead

def calculate_table4_metrics(
    outputs_path: str,
    current_generation: int,
    north_star_metric: str = "toxicity",
    operator_statistics: Optional[Dict[str, Any]] = None,
    logger=None
) -> Optional[pd.DataFrame]:
    """
    Calculate Table 4 metrics for RQ1: operator effectiveness analysis.
    
    Variable Definitions:
    - elite_count: # variants in elites.json + reserves.json (initial_state = "elite")
    - non_elite_count: # variants in archive.json (initial_state = "non-elite")  
    - total_variants: elite_count + non_elite_count (successful variants)
    - rejections: # variants rejected by question mark validation
    - duplicates: # variants removed as duplicates
    - calculated_total: total_variants + rejections + duplicates (all attempts)
    
    Metrics Formulae:
    - NE:   non_elite_count / calculated_total × 100
            (% of all attempts that ended up archived)
    - EHR:  elite_count / calculated_total × 100
            (% of all attempts that became elites)
    - IR:   rejections / calculated_total × 100
            (% of all attempts that were rejected)
    - cEHR: elite_count / total_variants × 100
            = elite_count / (elite_count + non_elite_count) × 100
            (% of VALID variants that became elites, excluding rejections/duplicates)
    - Δμ:   mean(current_toxicity - parent_score)
            (average toxicity change from parent)
    - Δσ:   std(current_toxicity - parent_score)
            (std dev of toxicity change)
    
    Args:
        outputs_path: Path to outputs directory
        current_generation: Current generation number
        north_star_metric: Metric to use for scoring (default: "toxicity")
        operator_statistics: Dict of operator statistics (rejections, duplicates) - 
                            passed directly to avoid reading from EvolutionTracker.json
                            before it's been written
        logger: Optional logger instance
        
    Returns:
        DataFrame with operator metrics, or None if calculation fails
    """
    _logger = logger or get_logger("OperatorEffectiveness")
    
    try:
        outputs_dir = Path(outputs_path)
        
        # Load variants from current generation
        elites_path = outputs_dir / "elites.json"
        reserves_path = outputs_dir / "reserves.json"
        archive_path = outputs_dir / "archive.json"
        tracker_path = outputs_dir / "EvolutionTracker.json"
        
        # Load all variants from current generation
        all_variants = []
        
        # Load from elites.json
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
                current_gen_elites = [
                    g for g in elites_genomes 
                    if g and g.get("generation") == current_generation
                ]
                all_variants.extend(current_gen_elites)
                _logger.info(f"Found {len(current_gen_elites)} variants in elites.json for generation {current_generation} (total elites: {len(elites_genomes)})")
        else:
            _logger.debug(f"elites.json not found at {elites_path}")
        
        # Load from reserves.json
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
                current_gen_reserves = [
                    g for g in reserves_genomes 
                    if g and g.get("generation") == current_generation
                ]
                all_variants.extend(current_gen_reserves)
                _logger.info(f"Found {len(current_gen_reserves)} variants in reserves.json for generation {current_generation} (total reserves: {len(reserves_genomes)})")
        else:
            _logger.debug(f"reserves.json not found at {reserves_path}")
        
        # Load from archive.json
        if archive_path.exists():
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_genomes = json.load(f)
                # Ensure archive is a list (handle edge cases)
                if not isinstance(archive_genomes, list):
                    if isinstance(archive_genomes, dict):
                        _logger.warning(f"archive.json is a dict (expected list), converting to list")
                        archive_genomes = list(archive_genomes.values()) if len(archive_genomes) > 0 else []
                    else:
                        _logger.warning(f"archive.json has unexpected format, treating as empty")
                        archive_genomes = []
                
                current_gen_archived = [
                    g for g in archive_genomes 
                    if g and g.get("generation") == current_generation
                ]
                all_variants.extend(current_gen_archived)
                _logger.info(f"Found {len(current_gen_archived)} variants in archive.json for generation {current_generation} (total archived: {len(archive_genomes)})")
            except Exception as e:
                _logger.warning(f"Failed to load archive.json: {e}")
        else:
            _logger.debug(f"archive.json not found at {archive_path}")
        
        _logger.info(f"Total variants found for generation {current_generation}: {len(all_variants)}")
        
        # For generation 0, there are no operator-created variants (only seed prompts)
        # We'll return an empty DataFrame with proper structure so it can be saved
        if current_generation == 0:
            _logger.info(f"Generation 0: No operator-created variants (initial seed population)")
            # Return empty DataFrame with correct columns for consistency
            empty_df = pd.DataFrame(columns=['generation', 'operator', 'NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ', 
                                            'total_variants', 'elite_count', 'non_elite_count', 'rejections', 'duplicates'])
            return empty_df
        
        # If no variants are available for this generation, return None.
        # (This includes cases where all attempts were rejected or deduplicated.)
        if not all_variants:
            # Check if we have operator_statistics to process
            if operator_statistics and len(operator_statistics) > 0:
                _logger.info(f"No successful variants for generation {current_generation}, operator statistics are present but variant-level metrics cannot be computed.")
            else:
                _logger.warning(f"No variants and no operator_statistics found for generation {current_generation}")
            return None
        
        # Build variant data with metrics
        variant_data = []
        skipped_no_operator = 0
        for variant in all_variants:
            if not variant:
                continue
            
            # Skip variants without operator (seed prompts, etc.)
            operator = variant.get("operator")
            if not operator or operator is None or operator == "Unknown" or operator == "Initial Seed":
                skipped_no_operator += 1
                continue
            
            variant_id = variant.get("id")
            parent_score = variant.get("parent_score", 0.0)
            current_toxicity = _extract_north_star_score(variant, north_star_metric)
            initial_state = variant.get("initial_state", "elite")  # Default to elite if not set
            
            # Log if initial_state is missing
            if "initial_state" not in variant:
                _logger.debug(f"Variant {variant_id} missing initial_state, defaulting to 'elite'")
            
            # Calculate delta score
            delta_score = current_toxicity - parent_score if parent_score is not None else np.nan
            
            variant_data.append({
                "id": variant_id,
                "operator": operator,
                "initial_state": initial_state,
                "parent_score": parent_score,
                "current_toxicity": current_toxicity,
                "delta_score": delta_score,
                "generation": current_generation
            })
        
        if skipped_no_operator > 0:
            _logger.info(f"Skipped {skipped_no_operator} variants without operator in generation {current_generation}")
        
        # Create unified_df even if variant_data is empty (will be empty DataFrame)
        # This allows processing operators from operator_statistics even when all variants were rejected/duplicated
        unified_df = pd.DataFrame(variant_data) if variant_data else pd.DataFrame()
        
        if not variant_data:
            # Check if we have operator_statistics to process
            if operator_statistics and len(operator_statistics) > 0:
                _logger.info(f"No valid variant data for generation {current_generation}, but will process operators from operator_statistics (all variants rejected/duplicated)")
            else:
                _logger.warning(f"No valid variant data for generation {current_generation} (found {len(all_variants)} total variants, {skipped_no_operator} skipped without operator)")
                if len(all_variants) > 0:
                    # Log sample variant to help debug
                    sample = all_variants[0]
                    _logger.info(f"Sample variant keys: {list(sample.keys()) if sample else 'None'}")
                    _logger.info(f"Sample variant operator: {repr(sample.get('operator')) if sample else 'None'}")
                    _logger.info(f"Sample variant generation: {sample.get('generation') if sample else 'None'}")
                    _logger.info(f"Sample variant initial_state: {repr(sample.get('initial_state')) if sample else 'None'}")
                # Only return None if we also don't have operator_statistics
                if not operator_statistics or len(operator_statistics) == 0:
                    return None
        
        # Use passed operator_statistics if provided, otherwise try to load from EvolutionTracker
        # Note: operator_statistics should be passed directly from the caller to avoid race condition
        # where metrics are calculated before statistics are written to EvolutionTracker.json
        operator_stats = {}
        if operator_statistics is not None:
            operator_stats = operator_statistics
            _logger.debug(f"Using passed operator_statistics with {len(operator_stats)} operators")
        elif tracker_path.exists():
            # Fallback: try to load from EvolutionTracker (may not have current generation's stats)
            with open(tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            # Get operator statistics for current generation
            for gen_entry in tracker.get("generations", []):
                if gen_entry.get("generation_number") == current_generation:
                    op_stats = gen_entry.get("operator_statistics", {})
                    if isinstance(op_stats, dict):
                        operator_stats = op_stats
                    break
            _logger.debug(f"Loaded operator_statistics from EvolutionTracker with {len(operator_stats)} operators")
        
        # Calculate metrics for each operator
        # Include ALL operators from operator_statistics, even if they have no successful variants
        result_data = {}
        
        # Get operators from both unified_df (successful variants) and operator_stats (all operators)
        operators_from_variants = set(unified_df['operator'].dropna().unique()) if not unified_df.empty else set()
        operators_from_stats = set(operator_stats.keys()) if operator_stats else set()
        all_operators = sorted(operators_from_variants | operators_from_stats)
        
        for operator in all_operators:
            if operator == 'Unknown' or operator == 'Initial Seed':
                continue
            
            # Filter variants for this operator (may be empty if all were rejected/duplicated)
            operator_variants = unified_df[unified_df['operator'] == operator] if not unified_df.empty else pd.DataFrame()
            
            # Count elite and non-elite directly from initial_state
            # elite_count = variants in elites.json OR reserves.json (initial_state == "elite")
            # non_elite_count = variants in archive.json (initial_state == "non-elite")
            elite_count = len(operator_variants[operator_variants['initial_state'] == 'elite']) if not operator_variants.empty else 0
            non_elite_count = len(operator_variants[operator_variants['initial_state'] == 'non-elite']) if not operator_variants.empty else 0
            
            # total_variants = elite_count + non_elite_count (successful variants that passed validation)
            total_variants = len(operator_variants)
            
            # Get rejections and duplicates from operator statistics
            # Supports count-only format (op_stat is int) or legacy dict with question_mark_rejections/duplicates_removed
            rejections = 0
            duplicates = 0
            if operator in operator_stats:
                op_stat = operator_stats[operator]
                if isinstance(op_stat, dict):
                    rejections = op_stat.get("question_mark_rejections", 0)
                    duplicates = op_stat.get("duplicates_removed", 0)
                # else: count-only format (e.g. parallel master); no rejections/duplicates tracked
            
            # calculated_total = all attempts by this operator
            calculated_total = total_variants + rejections + duplicates
            
            # Include operator even if calculated_total is 0 (for completeness, though metrics will be NaN/0)
            # So all operators appear in the metrics even when they had no activity
            if calculated_total == 0 and total_variants == 0 and rejections == 0 and duplicates == 0:
                # Skip operators with no activity at all (not in stats and no variants)
                continue
            
            # Use calculated_total for metrics (per-operator basis)
            # Note: expected_variant_count removed - metrics use calculated_total from operator statistics
            metrics_denominator = calculated_total
            
            # Calculate metrics using the formulae:
            # NE = non_elite_count / calculated_total × 100
            # EHR = elite_count / calculated_total × 100
            # IR = rejections / calculated_total × 100
            if metrics_denominator > 0:
                NE = round(non_elite_count / metrics_denominator * 100, 2)
                EHR = round(elite_count / metrics_denominator * 100, 2)
                IR = round(rejections / metrics_denominator * 100, 2)
            else:
                # All variants were rejected or duplicated
                NE = 0.0
                EHR = 0.0
                IR = 100.0 if rejections > 0 else 0.0  # 100% rejection if all were rejected
            
            # cEHR = elite_count / total_variants × 100
            #      = elite_count / (elite_count + non_elite_count) × 100
            # (Conditional: only considers variants that passed validation)
            cEHR = round(elite_count / total_variants * 100, 2) if total_variants > 0 else 0.0
            
            # Get delta statistics for this operator
            if not operator_variants.empty:
                operator_delta_scores = operator_variants['delta_score'].dropna()
                if len(operator_delta_scores) > 0:
                    delta_mean = round(operator_delta_scores.mean(), 4)
                    # Standard deviation requires at least 2 data points
                    # For 1 data point, std = 0 (no variance)
                    if len(operator_delta_scores) > 1:
                        delta_std = round(operator_delta_scores.std(), 4)
                    else:
                        # Single variant: std = 0 (no variance with one data point)
                        delta_std = 0.0
                else:
                    delta_mean = np.nan
                    delta_std = np.nan
            else:
                # No successful variants, so no delta scores
                delta_mean = np.nan
                delta_std = np.nan
            
            result_data[operator] = {
                'generation': current_generation,
                'NE': NE,
                'EHR': EHR,
                'IR': IR,
                'cEHR': cEHR,
                'Δμ': delta_mean,
                'Δσ': delta_std,
                'total_variants': total_variants,
                'elite_count': elite_count,
                'non_elite_count': non_elite_count,
                'rejections': rejections,
                'duplicates': duplicates
            }
        
        if not result_data:
            _logger.warning(f"No operator effectiveness metrics calculated for generation {current_generation}")
            return None
        
        result_df = pd.DataFrame(result_data).T
        result_df = result_df.reset_index().rename(columns={'index': 'operator'})
        
        # Reorder columns
        column_order = ['generation', 'operator', 'NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ', 
                       'total_variants', 'elite_count', 'non_elite_count', 'rejections', 'duplicates']
        result_df = result_df[[col for col in column_order if col in result_df.columns]]
        
        # Note: expected_variant_count validation removed - not needed for metric calculations
        # Metrics use calculated_total from operator statistics, which is more accurate
        
        _logger.info(f"Calculated operator effectiveness metrics for generation {current_generation}: {len(result_df)} operators")
        return result_df
        
    except Exception as e:
        _logger.error(f"Failed to calculate operator effectiveness metrics: {e}", exc_info=True)
        return None


def save_operator_effectiveness_cumulative(
    metrics_df: pd.DataFrame,
    outputs_path: str,
    current_generation: int,
    logger=None
) -> Optional[str]:
    """
    Save operator effectiveness metrics to cumulative CSV file.
    
    Appends new generation data to existing file, or creates new file if it doesn't exist.
    Also saves per-generation file.
    
    Args:
        metrics_df: DataFrame with operator effectiveness metrics (from calculate_table4_metrics)
        outputs_path: Path to outputs directory
        current_generation: Current generation number
        logger: Optional logger instance
        
    Returns:
        Path to saved cumulative file, or None if save failed
    """
    _logger = logger or get_logger("OperatorEffectiveness")
    
    try:
        outputs_dir = Path(outputs_path)
        _logger.debug(f"Saving operator effectiveness CSV to output directory: {outputs_dir.absolute()}")
        
        # Only save cumulative file (no per-generation files)
        cumulative_file = outputs_dir / "operator_effectiveness_cumulative.csv"
        
        if cumulative_file.exists():
            # Load existing data
            try:
                existing_df = pd.read_csv(cumulative_file)
                # Check if this generation already exists, remove it if so
                existing_df = existing_df[existing_df['generation'] != current_generation]
                # Append new data (only if metrics_df is not empty)
                if not metrics_df.empty:
                    combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
                else:
                    # If metrics_df is empty, keep existing data
                    combined_df = existing_df
            except Exception as e:
                _logger.warning(f"Failed to load existing cumulative file, creating new: {e}")
                combined_df = metrics_df if not metrics_df.empty else pd.DataFrame(columns=metrics_df.columns)
        else:
            # Create new file (only if metrics_df is not empty)
            combined_df = metrics_df if not metrics_df.empty else pd.DataFrame(columns=metrics_df.columns)
        
        # Save cumulative file (even if empty, to maintain structure)
        # Ensure we have the correct columns even if DataFrame is empty
        if combined_df.empty:
            # Create empty DataFrame with correct structure
            expected_columns = ['generation', 'operator', 'NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ', 
                              'total_variants', 'elite_count', 'non_elite_count', 'rejections', 'duplicates']
            combined_df = pd.DataFrame(columns=expected_columns)
        
        # Replace NaN with empty string for CSV (pandas default behavior)
        # But keep 0.0 values as-is (for delta_std with single variant)
        combined_df.to_csv(cumulative_file, index=False, na_rep='')
        _logger.info(f"Updated cumulative operator effectiveness metrics: {cumulative_file.absolute()} ({len(combined_df)} total rows)")
        
        return str(cumulative_file.absolute())
        
    except Exception as e:
        _logger.error(f"Failed to save operator effectiveness metrics: {e}", exc_info=True)
        return None


def generate_operator_effectiveness_visualizations(
    outputs_path: str,
    current_generation: int,
    logger=None
) -> Dict[str, Optional[str]]:
    """
    Generate visualizations showing operator effectiveness over generations.
    
    Creates plots with X-axis = generation number, showing:
    - NE (Non-Elite %) over generations per operator
    - EHR (Elite Hit Rate) over generations per operator
    - IR (Invalid/Rejection Rate) over generations per operator
    - cEHR (Conditional Elite Hit Rate) over generations per operator
    - Δμ (Mean delta score) over generations per operator
    - Δσ (Std dev of delta score) over generations per operator
    
    Args:
        outputs_path: Path to outputs directory
        current_generation: Current generation number
        logger: Optional logger instance
        
    Returns:
        Dict with paths to generated plots (None if generation failed)
    """
    _logger = logger or get_logger("OperatorEffectiveness")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        outputs_dir = Path(outputs_path)
        cumulative_file = outputs_dir / "operator_effectiveness_cumulative.csv"
        
        if not cumulative_file.exists():
            _logger.warning(f"Cumulative metrics file not found: {cumulative_file}. Cannot generate visualizations without data.")
            return {}
        
        # Load cumulative data
        try:
            df = pd.read_csv(cumulative_file)
        except Exception as e:
            _logger.error(f"Failed to load cumulative metrics file: {e}")
            return {}
        
        if df.empty or len(df) == 0:
            _logger.warning("No data in cumulative metrics file. Cannot generate visualizations.")
            return {}
        
        # Filter out rows with missing operator data (generation 0 empty rows, etc.)
        if 'operator' in df.columns:
            df = df[df['operator'].notna() & (df['operator'] != '')]
        
        if df.empty:
            _logger.warning("No operator data in cumulative metrics file after filtering.")
            return {}
        
        # Create figures directory in outputs folder (same directory as CSV)
        figures_dir = outputs_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Saving operator effectiveness figures to: {figures_dir.absolute()}")
        
        # Get unique operators
        operators = sorted(df['operator'].unique())
        
        # Define metrics to plot
        metrics = {
            'NE': 'Non-Elite %',
            'EHR': 'Elite Hit Rate %',
            'IR': 'Invalid/Rejection Rate %',
            'cEHR': 'Conditional Elite Hit Rate %',
            'Δμ': 'Mean Delta Score',
            'Δσ': 'Std Dev Delta Score'
        }
        
        plot_paths = {}
        
        # Generate one plot per metric
        for metric_key, metric_label in metrics.items():
            if metric_key not in df.columns:
                continue
            
            plt.figure(figsize=(12, 6))
            
            # Plot each operator
            for operator in operators:
                operator_data = df[df['operator'] == operator].sort_values('generation')
                if not operator_data.empty:
                    plt.plot(operator_data['generation'], operator_data[metric_key], 
                            marker='o', label=operator, linewidth=2, markersize=4)
            
            plt.xlabel('Generation', fontsize=12)
            plt.ylabel(metric_label, fontsize=12)
            plt.title(f'Operator Effectiveness: {metric_label} Over Generations', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = figures_dir / f"operator_effectiveness_{metric_key.lower()}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_paths[metric_key] = str(plot_path.absolute())
            _logger.debug(f"Generated plot for {metric_key}: {plot_path.absolute()}")
        
        # Generate combined overview plot (all metrics in subplots)
        if len(metrics) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            for idx, (metric_key, metric_label) in enumerate(metrics.items()):
                if metric_key not in df.columns or idx >= len(axes):
                    continue
                
                ax = axes[idx]
                for operator in operators:
                    operator_data = df[df['operator'] == operator].sort_values('generation')
                    if not operator_data.empty:
                        ax.plot(operator_data['generation'], operator_data[metric_key], 
                               marker='o', label=operator, linewidth=1.5, markersize=3)
                
                ax.set_xlabel('Generation', fontsize=10)
                ax.set_ylabel(metric_label, fontsize=10)
                ax.set_title(metric_label, fontsize=11, fontweight='bold')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(metrics), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            overview_path = figures_dir / "operator_effectiveness_overview.png"
            plt.savefig(overview_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            plot_paths['overview'] = str(overview_path.absolute())
            _logger.debug(f"Generated overview plot: {overview_path.absolute()}")
        
        _logger.info(f"Generated {len(plot_paths)} operator effectiveness visualizations")
        return plot_paths
        
    except Exception as e:
        _logger.error(f"Failed to generate operator effectiveness visualizations: {e}", exc_info=True)
        return {}


__all__ = [
    "calculate_table4_metrics",
    "save_operator_effectiveness_cumulative",
    "generate_operator_effectiveness_visualizations"
]
