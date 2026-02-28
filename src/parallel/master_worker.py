import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from mpi4py import MPI

PARENTS_REQUEST   = 10
PARENTS           = 11
EVALUATED_VARIANT = 12
GEN0_BATCH        = 13


def send_payload(comm, dest, tag, payload, logger=None):
    """Send a dict (or None) to dest with the given tag."""
    if logger:
        logger.debug("send -> rank %d  tag=%d  payload_type=%s", dest, tag, type(payload).__name__)
    comm.send(payload, dest=dest, tag=tag)


def recv_payload(comm, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, logger=None):
    """Recv from source/tag. Returns (data, tag_id, source_rank)."""
    status = MPI.Status()
    data = comm.recv(source=source, tag=tag, status=status)
    tag_id = status.Get_tag()
    source_rank = status.Get_source()
    if logger:
        logger.debug("recv <- rank %d  tag=%d  payload_type=%s", source_rank, tag_id, type(data).__name__)
    return data, tag_id, source_rank


# ---------------------------------------------------------------------------
# Helpers: dedup, merge, parent selection, tracker
# ---------------------------------------------------------------------------

def _load_existing_prompts(outputs_path, logger):
    """Load all prompts from elites, reserves, and archive for dedup."""
    existing = set()
    for fname in ("elites.json", "reserves.json", "archive.json"):
        fpath = outputs_path / fname
        if not fpath.exists():
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else list(data.values()) if isinstance(data, dict) else []
            for g in items:
                if isinstance(g, dict) and g.get("prompt"):
                    existing.add(g["prompt"])
        except Exception as e:
            logger.warning("Failed to load %s for dedup: %s", fname, e)
    return existing


def _merge_and_speciate(buffers, K, outputs_path, generation_id, next_genome_id,
                        north_star_metric, speciation_config, log_file, logger,
                        run_speciation_fn=None):
    """Drain up to K genomes from buffers (round-robin), dedup, write temp, run speciation.

    Returns (accepted_count, discarded_count, new_next_genome_id, speciation_result, accepted_genomes).
    """
    merge_start = time.time()
    buffer_snapshot = {r: len(b) for r, b in buffers.items() if b}
    logger.debug("Merge start: draining up to K=%d from buffers (round-robin). "
                 "Buffer snapshot per worker: %s", K, buffer_snapshot)

    existing_prompts = _load_existing_prompts(outputs_path, logger)
    logger.debug("Loaded %d existing prompts for dedup (elites+reserves+archive)",
                 len(existing_prompts))
    temp_prompts = set()

    sorted_ranks = sorted(buffers.keys())
    accepted = []
    discarded = 0
    error_count = 0
    dedup_count = 0
    idx = 0

    while len(accepted) < K and any(buffers[r] for r in sorted_ranks):
        rank = sorted_ranks[idx % len(sorted_ranks)]
        idx += 1
        if not buffers[rank]:
            if not any(buffers[r] for r in sorted_ranks):
                break
            continue
        genome = buffers[rank].pop(0)

        if genome.get("status") == "error":
            discarded += 1
            error_count += 1
            logger.debug("Skipping error genome from worker %d: %s",
                         rank, genome.get("error", "unknown"))
            continue

        prompt = genome.get("prompt", "")

        if prompt in existing_prompts or prompt in temp_prompts:
            discarded += 1
            dedup_count += 1
            logger.debug("Dedup discard: prompt already exists (worker %d)", rank)
            continue

        genome["id"] = next_genome_id
        genome["generation"] = generation_id
        next_genome_id += 1
        temp_prompts.add(prompt)
        accepted.append(genome)

    merge_elapsed = time.time() - merge_start
    logger.info("Merge done (%.2fs): %d accepted, %d discarded (%d dedup, %d errors), "
                "generation=%d, next_genome_id=%d",
                merge_elapsed, len(accepted), discarded, dedup_count, error_count,
                generation_id, next_genome_id)

    temp_path = outputs_path / "temp.json"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=2, ensure_ascii=False)
    logger.debug("Wrote %d genomes to temp.json", len(accepted))

    speciation_start = time.time()
    speciation_result = {}
    if run_speciation_fn is not None:
        logger.info("Speciation starting for generation %d (%d genomes in temp)...",
                     generation_id, len(accepted))
        try:
            speciation_result = run_speciation_fn(
                temp_path=str(temp_path),
                current_generation=generation_id,
                config=speciation_config,
                log_file=log_file,
                north_star_metric=north_star_metric,
            )
            speciation_elapsed = time.time() - speciation_start
            if speciation_result.get("success"):
                logger.info("Speciation gen %d done (%.2fs): %d species, %d reserves, "
                            "%d elites moved",
                            generation_id, speciation_elapsed,
                            speciation_result.get("species_count", 0),
                            speciation_result.get("reserves_size", 0),
                            speciation_result.get("elites_moved", 0))
            else:
                logger.warning("Speciation gen %d completed with warnings (%.2fs): %s",
                               generation_id, speciation_elapsed,
                               speciation_result.get("error", "unknown"))
        except Exception as e:
            speciation_elapsed = time.time() - speciation_start
            logger.error("Speciation gen %d failed (%.2fs): %s",
                         generation_id, speciation_elapsed, e, exc_info=True)
    else:
        _stub_speciation(outputs_path, temp_path, logger)
        speciation_result = {"success": True, "stub": True}
        logger.debug("Stub speciation completed for generation %d", generation_id)

    return len(accepted), discarded, next_genome_id, speciation_result, accepted


def _stub_speciation(outputs_path, temp_path, logger):
    """Fallback when real speciation is unavailable: move all temp genomes to reserves."""
    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            genomes = json.load(f)
    except Exception:
        genomes = []

    for g in genomes:
        g.setdefault("species_id", 0)

    reserves_path = outputs_path / "reserves.json"
    existing = []
    if reserves_path.exists():
        try:
            with open(reserves_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.extend(genomes)
    with open(reserves_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    for fname in ("elites.json", "archive.json"):
        fpath = outputs_path / fname
        if not fpath.exists():
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump([], f)

    logger.info("Stub speciation: moved %d genomes to reserves, temp cleared", len(genomes))


def _select_parents(outputs_path, north_star_metric, generation_id, logger):
    """Run parent selection and return (parents_list, top_10_list)."""
    from ea.parent_selector import ParentSelector

    op = str(outputs_path)

    tracker_path = outputs_path / "EvolutionTracker.json"
    evolution_tracker = None
    if tracker_path.exists():
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                evolution_tracker = json.load(f)
        except Exception as e:
            logger.warning("Could not load EvolutionTracker for parent selection: %s", e)

    selector = ParentSelector(north_star_metric)
    selector.adaptive_tournament_selection(
        evolution_tracker=evolution_tracker,
        outputs_path=op,
        current_generation=generation_id,
    )

    parents = []
    parents_path = outputs_path / "parents.json"
    if parents_path.exists():
        with open(parents_path, "r", encoding="utf-8") as f:
            parents = json.load(f)

    top_10 = []
    top10_path = outputs_path / "top_10.json"
    if top10_path.exists():
        with open(top10_path, "r", encoding="utf-8") as f:
            top_10 = json.load(f)

    return parents, top_10


def _collect_operator_stats(accepted_genomes):
    """Aggregate operator statistics from accepted genomes."""
    stats = {}
    for g in accepted_genomes:
        op_name = g.get("operator", "unknown")
        vtype = g.get("variant_type", "mutation")
        if op_name not in stats:
            stats[op_name] = {"count": 0, "mutation": 0, "crossover": 0}
        stats[op_name]["count"] += 1
        if vtype == "crossover":
            stats[op_name]["crossover"] += 1
        else:
            stats[op_name]["mutation"] += 1
    return stats


def _update_tracker(outputs_path, generation_id, total_evaluated, total_integrated,
                    total_discarded, speciation_result, logger,
                    north_star_metric="toxicity", log_file=None,
                    accepted_genomes=None):
    """Update EvolutionTracker with full generation statistics after speciation."""
    logger.debug("Updating EvolutionTracker (full): gen=%d  evaluated=%d  "
                 "integrated=%d  discarded=%d  accepted_genomes=%d",
                 generation_id, total_evaluated, total_integrated, total_discarded,
                 len(accepted_genomes) if accepted_genomes else 0)
    tracker_path = outputs_path / "EvolutionTracker.json"
    if not tracker_path.exists():
        tracker = {
            "status": "not_complete",
            "total_generations": 0,
            "generations_since_improvement": 0,
            "avg_fitness_history": [],
            "slope_of_avg_fitness": 0.0,
            "selection_mode": "default",
            "generations": [],
        }
        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)

    try:
        from utils.population_io import (
            calculate_generation_statistics,
            update_evolution_tracker_with_statistics,
        )

        gen_stats = calculate_generation_statistics(
            outputs_path=str(outputs_path),
            north_star_metric=north_star_metric,
            current_generation=generation_id,
            logger=logger,
            log_file=log_file,
        )

        gen_stats["total_evaluated"] = total_evaluated
        gen_stats["total_integrated"] = total_integrated
        gen_stats["total_discarded"] = total_discarded

        for key in ("species_count", "active_species_count", "frozen_species_count",
                     "reserves_size", "speciation_events", "merge_events",
                     "extinction_events", "archived_count", "elites_moved",
                     "reserves_moved", "genomes_updated", "inter_species_diversity",
                     "intra_species_diversity", "cluster_quality"):
            if key in speciation_result:
                gen_stats[key] = speciation_result[key]

        operator_statistics = None
        if accepted_genomes:
            operator_statistics = _collect_operator_stats(accepted_genomes)
            mutation_count = sum(1 for g in accepted_genomes if g.get("variant_type") != "crossover")
            crossover_count = sum(1 for g in accepted_genomes if g.get("variant_type") == "crossover")
            gen_stats["variants_created"] = len(accepted_genomes)
            gen_stats["mutation_variants"] = mutation_count
            gen_stats["crossover_variants"] = crossover_count

        update_evolution_tracker_with_statistics(
            evolution_tracker_path=str(tracker_path),
            current_generation=generation_id,
            statistics=gen_stats,
            operator_statistics=operator_statistics,
            logger=logger,
            log_file=log_file,
        )

        logger.info("Tracker updated: gen=%d  evaluated=%d  integrated=%d  discarded=%d  "
                     "elites=%d  reserves=%d  avg_fitness=%.4f",
                     generation_id, total_evaluated, total_integrated, total_discarded,
                     gen_stats.get("elites_count", 0), gen_stats.get("reserves_count", 0),
                     gen_stats.get("avg_fitness_generation", 0.0001))

    except Exception as e:
        logger.error("Full tracker update failed, falling back to minimal: %s", e, exc_info=True)
        with open(tracker_path, "r", encoding="utf-8") as f:
            tracker = json.load(f)
        gen_entry = {
            "generation_number": generation_id,
            "total_evaluated": total_evaluated,
            "total_integrated": total_integrated,
            "total_discarded": total_discarded,
            "species_count": speciation_result.get("species_count", 0),
            "reserves_size": speciation_result.get("reserves_size", 0),
            "elites_moved": speciation_result.get("elites_moved", 0),
            "reserves_moved": speciation_result.get("reserves_moved", 0),
        }
        tracker["total_generations"] = generation_id + 1
        tracker.setdefault("generations", []).append(gen_entry)
        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        logger.info("Tracker updated (minimal): gen=%d  evaluated=%d  integrated=%d  discarded=%d",
                     generation_id, total_evaluated, total_integrated, total_discarded)


def _run_live_analysis(outputs_path, logger):
    """Generate live visualizations after speciation (best-effort)."""
    try:
        from utils.live_analysis import run_live_analysis
        results = run_live_analysis(outputs_path=str(outputs_path), logger=logger)
        if results:
            ok = sum(1 for v in results.values() if v is not None)
            logger.info("Live analysis: generated %d/%d visualizations", ok, len(results))
    except Exception as e:
        logger.warning("Live analysis failed (non-fatal): %s", e)


def _run_final_statistics(outputs_path, north_star_metric, start_time, generations_completed,
                          log_file, logger):
    """Generate final statistics and plots at the end of a parallel run."""
    try:
        execution_time = time.time() - start_time
        logger.info("Generating final statistics: execution_time=%.1fs  "
                     "generations_completed=%d", execution_time, generations_completed)

        tracker_path = outputs_path / "EvolutionTracker.json"
        if not tracker_path.exists():
            logger.warning("No EvolutionTracker.json found; skipping final statistics.")
            return

        with open(tracker_path, "r", encoding="utf-8") as f:
            tracker = json.load(f)

        tracker["status"] = "complete"
        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        logger.debug("Marked EvolutionTracker status='complete'")

        from ea.run_evolution import create_final_statistics_with_tracker
        final_stats = create_final_statistics_with_tracker(
            evolution_tracker=tracker,
            north_star_metric=north_star_metric,
            execution_time=execution_time,
            generations_completed=generations_completed,
            logger=logger,
            log_file=log_file,
        )
        logger.info("Final statistics: %s", {k: v for k, v in final_stats.items()
                     if k in ("total_generations", "execution_time", "best_fitness")})
    except Exception as e:
        logger.warning("Final statistics generation failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

def master_main(comm, size, K, outputs_path, north_star_metric,
                speciation_config, log_file, logger,
                max_generations=None, run_speciation_fn=None,
                config_dict=None):
    """Master process (rank 0). Dispatch loop with per-worker buffers."""
    from utils.population_io import get_max_genome_id_from_all_files

    start_time = time.time()
    n_workers = size - 1
    logger.info("Master started. Workers: %d  K: %d  outputs: %s", n_workers, K, outputs_path)

    comm.bcast(config_dict, root=0)
    logger.info("Broadcast config to workers: %s", list((config_dict or {}).keys()))

    outputs_path = Path(outputs_path)
    outputs_path.mkdir(parents=True, exist_ok=True)
    for fname in ("temp.json", "elites.json", "reserves.json", "archive.json"):
        fpath = outputs_path / fname
        if not fpath.exists():
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump([], f)

    buffers = defaultdict(list)
    generation_id = 0
    next_genome_id = get_max_genome_id_from_all_files(str(outputs_path)) + 1
    total_evaluated = 0
    total_integrated = 0
    total_discarded = 0
    gen0_complete = False
    shutdown = False
    finished_workers = 0

    gen0_assignments = {}
    gen0_expected = 0
    gen0_returned = 0
    seed_file = (config_dict or {}).get("seed_file")
    if seed_file:
        try:
            df = pd.read_csv(seed_file, engine="python", on_bad_lines="skip",
                             sep=",", quotechar='"', skipinitialspace=True)
            if "questions" in df.columns:
                n_prompts = len(df["questions"].dropna())
                chunk = max(n_prompts // n_workers, 1)
                remainder = n_prompts % n_workers
                start = 0
                for w in range(1, size):
                    end = start + chunk + (1 if (w - 1) < remainder else 0)
                    end = min(end, n_prompts)
                    gen0_assignments[w] = (start, end)
                    start = end
                gen0_expected = sum(e - s for s, e in gen0_assignments.values())
                logger.info("Gen0: %d prompts (%d expected) distributed among %d workers: %s",
                            n_prompts, gen0_expected, n_workers, gen0_assignments)
        except Exception as e:
            logger.warning("Failed to read seed file for gen0 distribution: %s", e)

    def _total_buffered():
        return sum(len(b) for b in buffers.values())

    def _buffer_state_str():
        per_worker = {r: len(b) for r, b in buffers.items() if b}
        return f"total={_total_buffered()} per_worker={per_worker}"

    recv_count = 0
    logger.info("="*60)
    logger.info("Dispatch loop started. Waiting for messages from %d worker(s). "
                "K=%d  max_generations=%s", n_workers, K, max_generations)
    logger.info("="*60)

    while finished_workers < n_workers:
        data, tag_id, source = recv_payload(comm, logger=logger)
        recv_count += 1

        tag_name = {PARENTS_REQUEST: "PARENTS_REQUEST", EVALUATED_VARIANT: "EVALUATED_VARIANT",
                    PARENTS: "PARENTS", GEN0_BATCH: "GEN0_BATCH"}.get(tag_id, f"UNKNOWN({tag_id})")
        logger.debug("recv #%d: source=worker%d  tag=%s  gen0_complete=%s  shutdown=%s",
                      recv_count, source, tag_name, gen0_complete, shutdown)

        # ---- PARENTS_REQUEST ----
        if tag_id == PARENTS_REQUEST:
            req_id = data.get("request_id") if data else None
            logger.info("PARENTS_REQUEST from worker %d  request_id=%s", source, req_id)

            if not gen0_complete:
                if source in gen0_assignments:
                    s, e = gen0_assignments.pop(source)
                    payload = {"request_id": req_id, "prompt_start": s, "prompt_end": e}
                else:
                    payload = {"request_id": req_id, "prompt_start": 0, "prompt_end": 0}
                num_keys = len((config_dict or {}).get("perspective_api_keys", []))
                if num_keys:
                    payload["perspective_key_index"] = (source - 1) % num_keys
                send_payload(comm, source, GEN0_BATCH, payload, logger=logger)
                logger.info("Sent GEN0_BATCH to worker %d  [%d:%d]  "
                            "remaining_gen0_assignments=%d",
                            source, payload["prompt_start"], payload["prompt_end"],
                            len(gen0_assignments))

            elif shutdown:
                send_payload(comm, source, PARENTS, None, logger=logger)
                finished_workers += 1
                logger.info("Shutdown sent to worker %d  (%d/%d workers finished)",
                            source, finished_workers, n_workers)

            else:
                try:
                    logger.debug("Running parent selection for worker %d (generation=%d)...",
                                 source, generation_id)
                    parent_sel_start = time.time()
                    parents, top_10 = _select_parents(
                        outputs_path, north_star_metric, generation_id, logger)
                    parent_sel_elapsed = time.time() - parent_sel_start
                    num_keys = len((config_dict or {}).get("perspective_api_keys", []))
                    payload = {
                        "request_id": req_id,
                        "parents": parents,
                        "top_10": top_10,
                    }
                    if num_keys:
                        payload["perspective_key_index"] = (source - 1) % num_keys
                    send_payload(comm, source, PARENTS, payload, logger=logger)
                    logger.info("Sent PARENTS to worker %d  (%d parents, %d top_10)  "
                                "selection_time=%.2fs",
                                source, len(parents), len(top_10), parent_sel_elapsed)
                except Exception as e:
                    logger.error("Parent selection failed for worker %d: %s", source, e, exc_info=True)
                    send_payload(comm, source, PARENTS, None, logger=logger)
                    finished_workers += 1

        # ---- EVALUATED_VARIANT ----
        elif tag_id == EVALUATED_VARIANT:
            req_id = data.get("request_id")
            local_id = data.get("local_variant_id")
            status = data.get("status", "unknown")
            logger.info("EVALUATED_VARIANT from worker %d  request_id=%s  "
                        "local_variant_id=%s  status=%s  prompt=%.40s",
                        source, req_id, local_id, status,
                        str(data.get("prompt", ""))[:40])

            buffers[source].append(data)
            total_evaluated += 1
            if not gen0_complete:
                gen0_returned += 1

            if total_evaluated % 5 == 0 or _total_buffered() >= K:
                logger.info("Buffer state: %s  total_evaluated=%d  gen0_returned=%d/%d",
                            _buffer_state_str(), total_evaluated,
                            gen0_returned, gen0_expected)

            should_speciate = False
            if _total_buffered() >= K:
                should_speciate = True
            elif (not gen0_complete and not gen0_assignments
                  and gen0_returned >= gen0_expected
                  and _total_buffered() > 0):
                logger.info("Gen0 complete with partial batch: %d/%d returned, "
                            "%d buffered (< K=%d). Triggering speciation.",
                            gen0_returned, gen0_expected, _total_buffered(), K)
                should_speciate = True

            if should_speciate:
                cycle_start = time.time()
                logger.info("-"*50)
                logger.info("Merge+speciation triggered: %d buffered, K=%d, "
                            "generation=%d", _total_buffered(), K, generation_id)

                batch_size = min(_total_buffered(), K)
                accepted, discarded, next_genome_id, spec_result, accepted_genomes = \
                    _merge_and_speciate(
                        buffers, batch_size, outputs_path, generation_id, next_genome_id,
                        north_star_metric, speciation_config, log_file, logger,
                        run_speciation_fn=run_speciation_fn,
                    )
                total_integrated += accepted
                total_discarded += discarded

                _update_tracker(outputs_path, generation_id, total_evaluated,
                                total_integrated, total_discarded, spec_result, logger,
                                north_star_metric=north_star_metric, log_file=log_file,
                                accepted_genomes=accepted_genomes)

                _run_live_analysis(outputs_path, logger)

                cycle_elapsed = time.time() - cycle_start
                logger.info("Post-speciation summary: generation_id=%d  "
                            "total_evaluated=%d  total_integrated=%d  "
                            "total_discarded=%d  cycle_time=%.2fs",
                            generation_id, total_evaluated, total_integrated,
                            total_discarded, cycle_elapsed)
                logger.info("Remaining buffer after speciation: %s", _buffer_state_str())
                logger.info("-"*50)

                gen0_complete = True
                generation_id += 1

                if max_generations is not None and generation_id >= max_generations:
                    shutdown = True
                    logger.info("Max generations reached (%d/%d). Shutdown flag set.",
                                generation_id, max_generations)

    logger.info("All workers finished requesting. recv_count=%d", recv_count)

    if _total_buffered() > 0:
        logger.info("="*50)
        logger.info("Drain phase: %d genomes remaining in buffers. %s",
                     _total_buffered(), _buffer_state_str())
        drain_start = time.time()
        accepted, discarded, next_genome_id, spec_result, accepted_genomes = \
            _merge_and_speciate(
                buffers, _total_buffered(), outputs_path, generation_id, next_genome_id,
                north_star_metric, speciation_config, log_file, logger,
                run_speciation_fn=run_speciation_fn,
            )
        total_integrated += accepted
        total_discarded += discarded
        _update_tracker(outputs_path, generation_id, total_evaluated,
                        total_integrated, total_discarded, spec_result, logger,
                        north_star_metric=north_star_metric, log_file=log_file,
                        accepted_genomes=accepted_genomes)
        _run_live_analysis(outputs_path, logger)
        drain_elapsed = time.time() - drain_start
        logger.info("Drain complete (%.2fs): accepted=%d  discarded=%d",
                     drain_elapsed, accepted, discarded)
        logger.info("="*50)

    _run_final_statistics(outputs_path, north_star_metric, start_time, generation_id, log_file, logger)

    total_elapsed = time.time() - start_time
    logger.info("="*60)
    logger.info("Master done. generations=%d  evaluated=%d  integrated=%d  "
                "discarded=%d  total_time=%.1fs",
                generation_id, total_evaluated, total_integrated,
                total_discarded, total_elapsed)
    logger.info("="*60)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _load_seed_prompts(seed_file, start, end, logger):
    """Load prompts from the seed CSV file for the given index range."""
    try:
        df = pd.read_csv(seed_file, engine="python", on_bad_lines="skip",
                         sep=",", quotechar='"', skipinitialspace=True)
        if "questions" not in df.columns:
            logger.error("Seed file %s missing 'questions' column", seed_file)
            return []
        prompts = df["questions"].dropna().astype(str).str.strip().tolist()
        return prompts[start:end]
    except Exception as e:
        logger.error("Failed to load seed prompts from %s: %s", seed_file, e, exc_info=True)
        return []


def worker_main(comm, rank, size, logger, config_dict=None,
                response_generator=None, prompt_generator=None, evaluator=None):
    """Worker process (rank > 0).

    Receives config via bcast, initialises LLM/evaluator (or uses injected
    mocks), then loops: request work -> process -> send EVALUATED_VARIANTs.
    """
    config_dict = comm.bcast(None, root=0)
    logger.info("Worker %d received config: %s", rank, list((config_dict or {}).keys()))

    cfg = config_dict or {}
    north_star_metric = cfg.get("north_star_metric", "toxicity")
    operators_mode = cfg.get("operators_mode", "all")
    seed_file = cfg.get("seed_file", "data/prompt.csv")
    moderation_methods = cfg.get("moderation_methods")
    base_log_file = cfg.get("log_file")
    log_file = _rank_log_file(base_log_file, rank) or base_log_file
    outputs_path = cfg.get("outputs_path")

    if response_generator is None or prompt_generator is None:
        from gne import get_ResponseGenerator, get_PromptGenerator
        if response_generator is None:
            RG = get_ResponseGenerator()
            response_generator = RG(model_key="response_generator",
                                    config_path="config/RGConfig.yaml",
                                    log_file=log_file)
            logger.info("Worker %d: ResponseGenerator initialised", rank)
        if prompt_generator is None:
            PG = get_PromptGenerator()
            prompt_generator = PG(model_key="prompt_generator",
                                  config_path="config/PGConfig.yaml",
                                  log_file=log_file)
            logger.info("Worker %d: PromptGenerator initialised", rank)

    from ea.evolution_engine import set_global_generators, generate_single_variant
    set_global_generators(response_generator, prompt_generator)

    if evaluator is None:
        from gne.evaluator import HybridModerationEvaluator
        api_keys = cfg.get("perspective_api_keys")
        evaluator = HybridModerationEvaluator(
            config_path="config/RGConfig.yaml", log_file=log_file,
            api_keys=api_keys or None)
        logger.info("Worker %d: HybridModerationEvaluator initialised", rank)

    from gne.response_generator import process_single_genome
    from gne.evaluator import evaluate_single_genome
    from utils.refusal_penalty import apply_refusal_penalty_single

    seq = 0
    cycle = 0
    total_errors = 0
    worker_start_time = time.time()

    logger.info("="*50)
    logger.info("Worker %d ready. Entering request loop.", rank)
    logger.info("="*50)

    while True:
        req_id = f"{rank}_{cycle}"
        send_payload(comm, 0, PARENTS_REQUEST, {"request_id": req_id}, logger=logger)
        logger.info("Sent PARENTS_REQUEST  request_id=%s  (cycle=%d, total_sent=%d)",
                     req_id, cycle, seq)

        data, tag_id, _ = recv_payload(comm, source=0, logger=logger)

        if tag_id == PARENTS and data is None:
            logger.info("Received shutdown (None) from master. Exiting request loop.")
            break

        # ---- GEN0_BATCH (evaluation-only) ----
        if tag_id == GEN0_BATCH:
            prompt_start = data.get("prompt_start", 0)
            prompt_end = data.get("prompt_end", 0)
            n_prompts = prompt_end - prompt_start
            key_idx = data.get("perspective_key_index")
            logger.info("GEN0_BATCH received: request_id=%s  prompts[%d:%d] (%d prompts)  "
                        "key_idx=%s",
                        data.get("request_id"), prompt_start, prompt_end, n_prompts, key_idx)

            if key_idx is not None and hasattr(evaluator, "select_key"):
                evaluator.select_key(key_idx)

            prompts = _load_seed_prompts(seed_file, prompt_start, prompt_end, logger)
            logger.info("Gen0 batch: loaded %d prompts from seed file, processing...", len(prompts))
            gen0_batch_start = time.time()
            gen0_ok = 0
            gen0_err = 0

            for i, p in enumerate(prompts):
                local_variant_id = f"{rank}_{seq}"
                genome = {
                    "request_id": req_id,
                    "local_variant_id": local_variant_id,
                    "prompt": p,
                    "status": "pending_generation",
                }
                try:
                    logger.debug("Gen0 variant %d/%d: generating response for prompt=%.40s",
                                 i + 1, len(prompts), p[:40])
                    process_single_genome(response_generator, genome)
                    logger.debug("Gen0 variant %d/%d: evaluating...", i + 1, len(prompts))
                    evaluate_single_genome(evaluator, genome,
                                           moderation_methods=moderation_methods)
                    apply_refusal_penalty_single(genome, north_star_metric)
                    gen0_ok += 1
                except Exception as exc:
                    logger.error("Pipeline error (gen0) local_variant_id=%s: %s",
                                 local_variant_id, exc, exc_info=True)
                    genome["status"] = "error"
                    genome["error"] = str(exc)
                    gen0_err += 1
                    total_errors += 1
                send_payload(comm, 0, EVALUATED_VARIANT, genome, logger=logger)
                logger.debug("Sent EVALUATED_VARIANT (gen0)  local_variant_id=%s  status=%s",
                             local_variant_id, genome.get("status"))
                seq += 1

                if (i + 1) % 5 == 0:
                    logger.info("Gen0 progress: %d/%d prompts processed (%d ok, %d errors)",
                                 i + 1, len(prompts), gen0_ok, gen0_err)

            gen0_batch_elapsed = time.time() - gen0_batch_start
            logger.info("Gen0 batch complete: sent %d variants (%d ok, %d errors) "
                        "for request_id=%s in %.2fs",
                        len(prompts), gen0_ok, gen0_err, req_id, gen0_batch_elapsed)

        # ---- PARENTS (evolve + respond + evaluate) ----
        elif tag_id == PARENTS:
            parents = data.get("parents", [])
            top_10 = data.get("top_10", [])
            key_idx = data.get("perspective_key_index")
            logger.info("PARENTS received: request_id=%s  parents=%d  top_10=%d  "
                        "key_idx=%s  (cycle=%d)",
                        data.get("request_id"), len(parents), len(top_10), key_idx, cycle)

            if key_idx is not None and hasattr(evaluator, "select_key"):
                evaluator.select_key(key_idx)

            logger.info("Evolution cycle %d: generating variants from %d parents...",
                         cycle, len(parents))
            evolve_start = time.time()
            variants = generate_single_variant(
                parents, prompt_generator,
                north_star_metric=north_star_metric,
                operators_mode=operators_mode,
                top_10=top_10,
                log_file=log_file,
                outputs_path=outputs_path,
            )
            evolve_elapsed = time.time() - evolve_start
            logger.info("Generated %d variant(s) in %.2fs. Processing pipeline...",
                         len(variants), evolve_elapsed)

            cycle_ok = 0
            cycle_err = 0
            for i, variant in enumerate(variants):
                local_variant_id = f"{rank}_{seq}"
                variant["request_id"] = req_id
                variant["local_variant_id"] = local_variant_id
                try:
                    logger.debug("Variant %d/%d: generating response for prompt=%.40s",
                                 i + 1, len(variants),
                                 str(variant.get("prompt", ""))[:40])
                    process_single_genome(response_generator, variant)
                    logger.debug("Variant %d/%d: evaluating...", i + 1, len(variants))
                    evaluate_single_genome(evaluator, variant,
                                           moderation_methods=moderation_methods)
                    apply_refusal_penalty_single(variant, north_star_metric)
                    cycle_ok += 1
                except Exception as exc:
                    logger.error("Pipeline error local_variant_id=%s: %s",
                                 local_variant_id, exc, exc_info=True)
                    variant["status"] = "error"
                    variant["error"] = str(exc)
                    cycle_err += 1
                    total_errors += 1
                send_payload(comm, 0, EVALUATED_VARIANT, variant, logger=logger)
                logger.debug("Sent EVALUATED_VARIANT  local_variant_id=%s  status=%s",
                             local_variant_id, variant.get("status"))
                seq += 1

            cycle_elapsed = time.time() - evolve_start
            logger.info("Evolution cycle %d complete: sent %d variants (%d ok, %d errors) "
                        "in %.2fs  (total_sent=%d)",
                        cycle, len(variants), cycle_ok, cycle_err, cycle_elapsed, seq)

        cycle += 1

    worker_elapsed = time.time() - worker_start_time
    logger.info("="*50)
    logger.info("Worker %d done. total_variants_sent=%d  cycles=%d  "
                "total_errors=%d  uptime=%.1fs",
                rank, seq, cycle, total_errors, worker_elapsed)
    logger.info("="*50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_perspective_api_keys():
    """Load Perspective API keys from environment (same priority as evaluator)."""
    import os
    multi = os.getenv("PERSPECTIVE_API_KEYS", "").strip()
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    idx = 0
    indexed = []
    while True:
        val = os.getenv(f"PERSPECTIVE_API_KEY_{idx}", "").strip()
        if not val:
            break
        indexed.append(val)
        idx += 1
    if indexed:
        return indexed
    single = os.getenv("PERSPECTIVE_API_KEY", "").strip()
    if single:
        if "," in single:
            return [k.strip() for k in single.split(",") if k.strip()]
        return [single]
    return []


def _rank_log_file(base_log_file, rank):
    """Derive a per-rank log filename from the base log file path.

    Example: logs/20260227_run1.log -> logs/20260227_run1_master.log (rank 0)
                                    -> logs/20260227_run1_worker1.log (rank 1)
    """
    if base_log_file is None:
        return None
    stem, ext = os.path.splitext(base_log_file)
    suffix = "master" if rank == 0 else f"worker{rank}"
    return f"{stem}_{suffix}{ext}"


def run(logger, K=4, outputs_path=None, north_star_metric="toxicity",
        speciation_config=None, log_file=None, max_generations=2,
        run_speciation_fn=None,
        operators_mode="all", seed_file="data/prompt.csv",
        moderation_methods=None,
        response_generator=None, prompt_generator=None, evaluator=None,
        perspective_api_keys=None):
    """MPI entry point. Branch on rank."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        raise RuntimeError("Need at least 2 MPI ranks (1 master + 1 worker)")

    # Assign device by rank: rank 0 = CPU, rank 1 = cuda:0, rank 2 = cuda:1, ...
    from utils.device_utils import device_manager
    device_manager.set_mpi_device(rank, size)

    rank_log_file = _rank_log_file(log_file, rank)
    if rank_log_file is not None:
        import logging
        rank_label = "master" if rank == 0 else f"worker{rank}"
        rank_logger = logging.getLogger(f"mw_{rank_label}")
        if not rank_logger.hasHandlers():
            rank_logger.setLevel(logging.DEBUG)
            os.makedirs(os.path.dirname(rank_log_file) or ".", exist_ok=True)
            fh = logging.FileHandler(rank_log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "[%(asctime)s] [%(levelname)-8s] [%(name)-20s] [%(filename)s:%(lineno)d]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"))
            rank_logger.addHandler(fh)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                f"[%(asctime)s] [%(levelname)s] [{rank_label}] [%(name)s]: %(message)s",
                datefmt="%H:%M:%S"))
            rank_logger.addHandler(ch)
        logger = rank_logger

    if outputs_path is None:
        from utils.population_io import get_outputs_path
        outputs_path = str(get_outputs_path())

    if perspective_api_keys is None:
        perspective_api_keys = _load_perspective_api_keys()

    config_dict = {
        "north_star_metric": north_star_metric,
        "operators_mode": operators_mode,
        "seed_file": seed_file,
        "moderation_methods": moderation_methods,
        "log_file": log_file,
        "outputs_path": outputs_path,
        "K": K,
        "perspective_api_keys": perspective_api_keys,
    }

    if rank == 0:
        master_main(comm, size, K, outputs_path, north_star_metric,
                    speciation_config, rank_log_file or log_file, logger,
                    max_generations=max_generations,
                    run_speciation_fn=run_speciation_fn,
                    config_dict=config_dict)
    else:
        worker_main(comm, rank, size, logger,
                    response_generator=response_generator,
                    prompt_generator=prompt_generator,
                    evaluator=evaluator)

    comm.Barrier()
