"""Fitness-evolution series: this-gen max vs cumulative best-so-far."""

from utils.live_analysis import _fitness_plot_series


def test_per_gen_max_is_this_generation_not_durable_store():
    gens = [
        {"max_score_variants": 0.20, "population_max_fitness": 0.20, "best_fitness": 0.20},
        {"max_score_variants": 0.05, "population_max_fitness": 0.20, "best_fitness": 0.20},
        {"max_score_variants": 0.35, "population_max_fitness": 0.35, "best_fitness": 0.35},
        {"max_score_variants": 0.10, "population_max_fitness": 0.35, "best_fitness": 0.35},
    ]
    per_gen, _mins, _avgs, cumulative = _fitness_plot_series(gens)
    assert per_gen == [0.20, 0.05, 0.35, 0.10]
    assert cumulative == [0.20, 0.20, 0.35, 0.35]


def test_missing_population_max_uses_running_max_of_this_gen():
    gens = [
        {"max_score_variants": 0.08, "best_fitness": 0.08},
        {"max_score_variants": 0.11, "best_fitness": 0.11},
        {"max_score_variants": 0.07, "best_fitness": 0.07},
    ]
    per_gen, _mins, _avgs, cumulative = _fitness_plot_series(gens)
    assert per_gen == [0.08, 0.11, 0.07]
    assert cumulative == [0.08, 0.11, 0.11]


def test_missing_this_gen_max_is_zero_not_durable_max():
    gens = [
        {"population_max_fitness": 0.40, "best_fitness": 0.40},
        {"max_score_variants": 0.12, "population_max_fitness": 0.40},
    ]
    per_gen, _mins, _avgs, cumulative = _fitness_plot_series(gens)
    assert per_gen == [0.0, 0.12]
    assert cumulative == [0.40, 0.40]


def test_legacy_population_max_toxicity_used_when_fitness_missing():
    gens = [
        {"max_score_variants": 0.10, "population_max_toxicity": 0.10},
        {"max_score_variants": 0.04, "population_max_toxicity": 0.10},
    ]
    per_gen, _mins, _avgs, cumulative = _fitness_plot_series(gens)
    assert per_gen == [0.10, 0.04]
    assert cumulative == [0.10, 0.10]


def test_falsy_zero_this_gen_max_is_kept():
    gens = [
        {"max_score_variants": 0.0, "population_max_fitness": 0.25},
        {"max_score_variants": 0.0, "population_max_fitness": 0.25},
    ]
    per_gen, _mins, _avgs, cumulative = _fitness_plot_series(gens)
    assert per_gen == [0.0, 0.0]
    assert cumulative == [0.25, 0.25]
