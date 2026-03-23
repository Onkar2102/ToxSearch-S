# ToxSearch-S — Architecture Diagram (Master-Worker)

Single diagram for ToxSearch-S with the master-worker (MPI) parallel extension. See [ARCHITECTURE.md](ARCHITECTURE.md) for full specification.

---

Optional renderer config (for Mermaid Live / Kroki; GitHub may ignore):

```yaml
---
config:
  layout: dagre
  look: neo
---
```

---

```mermaid
flowchart LR
  subgraph Master["Master (MPI Rank 0, CPU)"]
    direction TB
    Select["Parent Selection"]
    Dispatch["Dispatch / Receive Loop"]
    Merge["Merge + Dedup"]
    Speciate["Speciation"]
    State[("Population State:\nelites + reserves\narchive + temp")]
    Tracker[("Evolution Tracker")]
    Select --> Dispatch
    Dispatch --> Merge
    Merge --> Speciate
    Speciate --> State
    Speciate --> Tracker
    State --> Select
    State --> Outputs
    Tracker --> Outputs
  end

  subgraph W1["Worker i"]
    direction TB
    Var["Variant Creation"]
    Gen["Response Generation"]
    Eval["Evaluation"]
    LocalModel["Local LLM\n(llama-cpp on GPU)"]
    Var --> Gen
    Gen --> Eval
    LocalModel --> Gen
  end

  subgraph WN["Worker j"]
    direction TB
    Var2["Variant Creation"]
    Gen2["Response Generation"]
    Eval2["Evaluation"]
    LocalModel2["Local LLM\n(llama-cpp on GPU)"]
    Var2 --> Gen2
    Gen2 --> Eval2
    LocalModel2 --> Gen2
  end

  subgraph Workers["Worker Pool (MPI Ranks 1..N)"]
    direction LR
    W1
    WN
  end

  subgraph ToxSearchS["ToxSearch-S with Master-Worker Parallelism"]
    direction TB
    Master
    Workers
  end

  Seeds["Seed Prompts CSV"] --> Master
  Config["RG/PG Config YAMLs"] --> Master
  Config --> Workers

  Dispatch -->|GEN0_BATCH / PARENTS| W1
  Dispatch -->|GEN0_BATCH / PARENTS| WN
  W1 -->|EVALUATED_VARIANT| Dispatch
  WN -->|EVALUATED_VARIANT| Dispatch

  Eval -->|toxicity| Perspective["Perspective API"]
  Eval2 -->|toxicity| Perspective

  Outputs[("Outputs:\nelites.json\nreserves.json\narchive.json\nEvolutionTracker.json")]
```

---

## MPI communication

### All message tags (direction and meaning)

```mermaid
flowchart LR
  W["Worker\n(rank i)"]
  M["Master\n(rank 0)"]

  W -->|"PARENTS_REQUEST (10)  EVALUATED_VARIANT (12)  WORKER_READY (20)  WORKER_INIT_FAILED (21)"| M
  M -->|"PARENTS (11)  GEN0_BATCH (13)  STOP (14)"| W
```

- **Worker → Master:** `PARENTS_REQUEST` (request work), `EVALUATED_VARIANT` (one genome), `WORKER_READY` (init OK), `WORKER_INIT_FAILED` (init error).
- **Master → Worker:** `PARENTS` (parents + top_10, or None for shutdown), `GEN0_BATCH` (seed prompt range), `STOP` (stop signal).

| Tag | Name | Direction | When / payload |
|-----|------|-----------|----------------|
| 10 | PARENTS_REQUEST | Worker → Master | Request work (`request_id`) |
| 11 | PARENTS | Master → Worker | Parents + top_10 + key_index; or `None` (shutdown) |
| 12 | EVALUATED_VARIANT | Worker → Master | One evaluated genome (`request_id`, genome) |
| 13 | GEN0_BATCH | Master → Worker | Seed prompt range (`prompt_start`, `prompt_end`) |
| 14 | STOP | Master → Worker | Stop signal |
| 20 | WORKER_READY | Worker → Master | Init success (models loaded) |
| 21 | WORKER_INIT_FAILED | Worker → Master | Init failure (`rank`, `error`) |

### Startup, evolution cycle, and shutdown

**Note:** The **message tags** (10, 11, 12, 13, 14, 20, 21) are used only for **point-to-point** messages (`send`/`recv`) between master and workers. **`bcast` (broadcast)** is an **MPI collective**: the root (master) sends the same data to all processes in one call; it does not use a tag. So config is distributed via `comm.bcast(config_dict, root=0)`, not via a tagged message.

```mermaid
sequenceDiagram
  participant M as Master (rank 0)
  participant W as Worker (rank i)

  Note over M,W: Startup (collective: no tag)
  M->>W: bcast(config_dict)
  Note over W: Load .env, init RG, PG, Evaluator
  alt Init OK
    W->>M: WORKER_READY (20)
  else Init failed
    W->>M: WORKER_INIT_FAILED (21) + error
    M->>M: Abort
  end
  Note over M: Wait all WORKER_READY (timeout 900s)

  loop Evolution
    W->>M: PARENTS_REQUEST (10)
    alt Generation 0
      M->>W: GEN0_BATCH (13)
      Note over W: generate → evaluate per prompt
      W->>M: EVALUATED_VARIANT (12) × N
    else Generation ≥ 1
      M->>W: PARENTS (11)
      Note over W: variant → generate → evaluate
      W->>M: EVALUATED_VARIANT (12)
    end
  end

  Note over M,W: Shutdown (max genomes / generations)
  M->>W: PARENTS (11) = None or STOP (14)
  Note over W: Exit request loop
```

---

## References

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full method and system specification
- [README.md](README.md) — Setup, running, worker log interpretation
- [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt) — Output file field definitions
