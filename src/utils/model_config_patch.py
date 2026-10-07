"""Patch GGUF model paths into RG/PG YAML configs from CLI --rg / --pg."""

from __future__ import annotations

import json
import re
from pathlib import Path

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


def _resolve_exact_gguf(value: str, *, get_project_root) -> str:
    """Require an existing ``.gguf`` file path — no quantization / directory fallbacks."""
    if not value or not str(value).strip():
        raise ValueError("Model path is empty; pass an existing .gguf via --rg / --pg")
    value = str(value).strip()
    if not _is_gguf_path(value):
        raise ValueError(
            f"Model path must be an existing .gguf file (got {value!r}). "
            "Example: models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-Q4_K_S.gguf"
        )
    p = Path(value)
    if not p.is_absolute():
        p = get_project_root() / value
    if not p.is_file():
        raise FileNotFoundError(f"GGUF not found: {value} (resolved: {p})")
    return value


def update_model_configs(
    rg_model: str,
    pg_model: str,
    logger,
    *,
    get_project_root,
    get_config_path,
) -> None:
    """Write exact ``--rg`` / ``--pg`` GGUF paths into RG/PG YAML (fail if missing)."""
    try:
        logger.info("Updating config files with models: RG=%s, PG=%s", rg_model, pg_model)

        rg_file = _resolve_exact_gguf(rg_model, get_project_root=get_project_root)
        pg_file = _resolve_exact_gguf(pg_model, get_project_root=get_project_root)

        rg_config_path = get_config_path() / "RGConfig.yaml"
        if not rg_config_path.exists():
            raise FileNotFoundError(f"Missing {rg_config_path}")
        _patch_yaml_section_model_name(rg_config_path, "response_generator", rg_file)
        logger.info("Config updated from script (--rg): RGConfig.yaml response_generator.name = %s", rg_file)

        pg_config_path = get_config_path() / "PGConfig.yaml"
        if not pg_config_path.exists():
            raise FileNotFoundError(f"Missing {pg_config_path}")
        _patch_yaml_section_model_name(pg_config_path, "prompt_generator", pg_file)
        logger.info("Config updated from script (--pg): PGConfig.yaml prompt_generator.name = %s", pg_file)

        logger.info("Project configs updated from script parameters: RG=%s, PG=%s", rg_file, pg_file)

    except Exception as e:
        logger.error("Failed to update model configurations: %s", e)
        raise
