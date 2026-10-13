"""Post-hoc analysis for the Stage A-D protocol (paper Section 4).

Modules:
    runs         index run directories into the factorial (evaluator x clustering x distance x seed)
    metrics      per-run outcomes computed from final population files + live snapshots
    stage_a      offline distance agreement / probes / threshold calibration on P0
    rq           RQ1-RQ4 aggregation (factorial tests, dynamics, cross-oracle)
    run_all      Stage C/D table CLI
    run_stage_a  Stage A build P0 + intrinsic tables CLI
    run_stage_d  Stage D top-k cross-oracle rescoring CLI

Run from project root with ``PYTHONPATH=src:.``.
"""
