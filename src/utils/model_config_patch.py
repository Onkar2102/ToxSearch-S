"""Patch GGUF model paths into RG/PG YAML configs from CLI --rg / --pg."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from utils import get_custom_logging

get_logger, _, _, PerformanceLogger = get_custom_logging()


def _is_gguf_path(value: str) -> bool:
    p = Path(value)
    return str(value).lower().endswith(".gguf") and (
        p.is_absolute() or str(value).startswith("./") or str(value).startswith("models/")
    )


def _patch_yaml_section_model_name(config_path: Path, section: str, new_name: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    section_line = f"{section}:"
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_line or stripped.startswith(section_line + " ") or re.match(
            rf"^{re.escape(section)}\s*:", stripped
        ):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Section '{section}' not found in {config_path}")
    section_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    name_re = re.compile(r"^(\s*)name:\s*.+$")
    for j in range(start_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cur_indent = len(line) - len(line.lstrip())
        if cur_indent <= section_indent:
            break
        if not line.lstrip().startswith("name:"):
            continue
        m = name_re.match(line.rstrip("\r\n"))
        if not m:
            continue
        indent = m.group(1)
        lines[j] = f"{indent}name: {json.dumps(new_name)}\n"
        config_path.write_text("".join(lines))
        return
    raise ValueError(f"No 'name:' key found under '{section}' in {config_path}")


def update_model_configs(
    rg_model: str,
    pg_model: str,
    logger,
    *,
    get_project_root,
    get_config_path,
) -> None:
    """Resolve GGUF paths and write ``name`` fields under RG/PG YAML sections."""
    try:
        logger.info("Updating config files with models: RG=%s, PG=%s", rg_model, pg_model)

        pref_order = [
            "f32", "Q8_0", "Q8_K", "Q8_K_M", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q4_K", "Q3_K_M", "Q3_K_L", "Q2_K"
        ]

        def resolve_model_entry(value: str) -> Optional[str]:
            if not value:
                return None
            requested_path = None
            if _is_gguf_path(value):
                p = Path(value)
                if not p.is_absolute():
                    p = get_project_root() / value
                if p.exists():
                    return value
                parent = p.parent
                if parent.exists():
                    requested_path = value
                    alias = str(Path(value).parent).replace("\\", "/")
                    value = alias
                else:
                    logger.warning("Model file not found and parent dir missing: %s", value)
                    return None

            alias = value
            if str(alias).startswith("models/") or Path(alias).is_absolute():
                base_dir = get_project_root() / alias if not Path(alias).is_absolute() else Path(alias)
            else:
                base_dir = get_project_root() / "models" / alias
            if not base_dir.exists():
                logger.warning("Model alias directory not found: %s", base_dir)
                return None
            ggufs = sorted([p for p in base_dir.glob("*.gguf")], key=lambda p: p.name)
            if not ggufs:
                logger.warning("No GGUF files found under: %s", base_dir)
                return None
            order = pref_order
            if requested_path:
                preferred = [q for q in pref_order if q in requested_path]
                if preferred:
                    order = preferred + [p for p in pref_order if p not in preferred]
                    logger.info("Preferring quantization from requested path: %s", preferred[0])
            for pref in order:
                for f in ggufs:
                    if pref in f.name:
                        rel = (Path(alias) / f.name) if str(alias).startswith("models/") else (Path("./models") / alias / f.name)
                        logger.info("Resolved %s -> %s", alias, rel)
                        return str(rel)
            return None

        rg_file = resolve_model_entry(rg_model)
        pg_file = resolve_model_entry(pg_model)

        if not rg_file and not pg_file:
            logger.error("No models could be resolved for RG=%s, PG=%s", rg_model, pg_model)
            raise ValueError(f"No models could be resolved for RG={rg_model}, PG={pg_model}")

        rg_config_path = get_config_path() / "RGConfig.yaml"
        if rg_config_path.exists() and rg_file:
            _patch_yaml_section_model_name(rg_config_path, "response_generator", rg_file)
            logger.info("Config updated from script (--rg): RGConfig.yaml response_generator.name = %s", rg_file)
        elif rg_config_path.exists() and not rg_file:
            logger.warning("Skipped RGConfig.yaml update; no file resolved for alias '%s'", rg_model)

        pg_config_path = get_config_path() / "PGConfig.yaml"
        if pg_config_path.exists() and pg_file:
            _patch_yaml_section_model_name(pg_config_path, "prompt_generator", pg_file)
            logger.info("Config updated from script (--pg): PGConfig.yaml prompt_generator.name = %s", pg_file)
        elif pg_config_path.exists() and not pg_file:
            logger.warning("Skipped PGConfig.yaml update; no file resolved for alias '%s'", pg_model)

        logger.info(
            "Project configs updated from script parameters: RG=%s, PG=%s",
            rg_file or "(unchanged)",
            pg_file or "(unchanged)",
        )

    except Exception as e:
        logger.error("Failed to update model configurations: %s", e)
        raise
