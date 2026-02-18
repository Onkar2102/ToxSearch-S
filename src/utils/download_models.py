"""
download_models.py


Idempotent multi-model downloader & local loader for Hugging Face models.
- Downloads into a project-local folder (default: ./models)
- Supports aliases, --all / --only selection
- Always loads from disk after download

Usage examples:
    python download_models.py --all
    python download_models.py --only llama3.2-3b-instruct --target-dir models
    python download_models.py --only llama3.2-3b-instruct --revision main
    python download_models.py --all --test --config config/RGConfig.yaml --prompt "Hello there!"
    python download_models.py --only qwen2.5-7b-instruct-gguf --gguf

Env token: set HUGGINGFACE_HUB_TOKEN (preferred) or HF_TOKEN / HF_API_TOKEN to access gated/private repos.
Note: The official Meta LLaMA 3.2 text-only sizes are 1B and 3B; 8B is available in LLaMA 3/3.1 series.
Also available (transformers): Llama-3.3-70B-Instruct; plus recent families like Mistral-7B-Instruct-v0.3, Qwen2.5-7B(-1M)-Instruct, Gemma-2-9B-IT, Phi-3.5-mini-instruct, and DeepSeek-Coder-V2-Instruct.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

import yaml
from huggingface_hub import HfApi, login, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class ModelManager:
    def __init__(
        self,
        model_registry: Optional[Dict[str, str]] = None,
        default_target_dir: Optional[Path] = None,
        registry_file: Optional[Path] = None,
    ):
        self.MODEL_REGISTRY = model_registry or {
            "llama3.2-1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3.2-3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",

            "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",

            "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5-7b-instruct-1m": "Qwen/Qwen2.5-7B-Instruct-1M",

            "gemma-2-9b-it": "google/gemma-2-9b-it",

            "phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",

        }
        
        self.GGUF_MODEL_REGISTRY = {


            "qwen2.5-7b-instruct-gguf": "bartowski/Qwen2.5-7B-Instruct-GGUF",


            "phi-3.5-mini-instruct-gguf": "bartowski/Phi-3.5-mini-instruct-GGUF",

            "mixtral-8x7b-instruct-gguf": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",

            "llama3.2-3b-instruct-gguf": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "llama3.2-1b-instruct-gguf": "bartowski/Llama-3.2-1B-Instruct-GGUF",

            "stable-lm-2-1.6b-gguf": "bartowski/stable-lm-2-1.6b-GGUF",

            "falcon-7b-instruct-gguf": "bartowski/Falcon-7B-Instruct-GGUF",

        }
        self.DEFAULT_TARGET_DIR = default_target_dir or Path("models")
        self.REGISTRY_FILE = registry_file or Path("models/models_registry.json")

    def get_env_token(self) -> Optional[str]:
        """
        Returns token from environment in priority order:
        1) HUGGINGFACE_HUB_TOKEN (preferred by huggingface_hub)
        2) HF_TOKEN
        3) HF_API_TOKEN
        """
        return (
            os.getenv("HUGGINGFACE_HUB_TOKEN")
            or os.getenv("HF_TOKEN")
            or os.getenv("HF_API_TOKEN")
        )

    def ensure_dir(self, path: Path) -> None:
        """Create directory and parents if they do not exist."""
        path.mkdir(parents=True, exist_ok=True)

    def local_model_dir(self, target_dir: Path, alias: str) -> Path:
        """Return the path where a model with the given alias is stored (target_dir / alias)."""
        return target_dir / alias

    def model_already_present(self, path: Path) -> bool:
        """Return True if path exists and contains required transformers files (config, tokenizer)."""
        required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        return path.exists() and any((path / r).exists() for r in required)

    def gguf_model_already_present(self, path: Path) -> bool:
        """Return True if path exists and contains at least one .gguf file."""
        gguf_files = list(path.glob("*.gguf"))
        return path.exists() and len(gguf_files) > 0

    def repo_exists(self, repo_id: str, revision: Optional[str] = None) -> bool:
        """Return True if the Hugging Face repo exists and is accessible; False on not found or error."""
        api = HfApi(token=self.get_env_token())
        try:
            api.model_info(repo_id, revision=revision)
            return True
        except RepositoryNotFoundError:
            return False
        except Exception:
            print(f"[warn] Could not validate repo: {repo_id} (revision={revision})")
            return False

    def download_one(
        self,
        alias: str,
        repo_id: str,
        target_dir: Path,
        revision: Optional[str] = None,
        gguf: bool = False,
    ) -> Path:
        out_dir = self.local_model_dir(target_dir, alias)
        self.ensure_dir(out_dir)

        if gguf and self.gguf_model_already_present(out_dir):
            print(f"[skip] GGUF {alias} already present at {out_dir}")
            return out_dir
        elif not gguf and self.model_already_present(out_dir):
            print(f"[skip] {alias} already present at {out_dir}")
            return out_dir

        if not self.repo_exists(repo_id, revision):
            print(f"[skip] Repository not found or inaccessible: {repo_id}. "
                  f"Tip: verify the repo id or accept the license for gated models.")
            return out_dir

        print(f"[download] {alias} ← {repo_id} -> {out_dir}")
        
        if gguf:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(out_dir),
                revision=revision,
                token=self.get_env_token(),
                allow_patterns=["*.gguf", "*.json", "tokenizer*"]
            )
        else:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(out_dir),
                revision=revision,
                token=self.get_env_token(),
            )
        
        print(f"[ok] Saved {alias} at {out_dir}")
        return out_dir

    def write_registry(self, registry_path: Path, mapping: Dict[str, str]) -> None:
        """Merge mapping into the JSON registry file at registry_path (atomic write)."""
        existing = {}
        if registry_path.exists():
            try:
                existing = json.loads(registry_path.read_text())
            except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                pass
        existing.update(mapping)
        tmp = registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.replace(registry_path)
        print(f"[registry] Updated {registry_path}")

    def maybe_login(self, do_login: bool) -> None:
        """Log in to Hugging Face Hub using env token if set, else interactive login if do_login is True."""
        token = self.get_env_token()
        if token:
            try:
                login(token=token, add_to_git_credential=True)
                print("[auth] Using Hugging Face token from environment.")
                return
            except Exception as e:
                print(f"[warn] environment token login failed: {e}")
        if do_login:
            try:
                print("[auth] Logging in to Hugging Face Hub…")
                login()
            except Exception as e:
                print(f"[warn] interactive login failed or skipped: {e}")
        else:
            if not token:
                print("[auth] No HF token in environment; public models will work, but gated/private models may fail.")

    def download_models(
        self,
        all_models: bool = False,
        only: Optional[list] = None,
        target_dir: Optional[Path] = None,
        revision: Optional[str] = None,
        login_flag: bool = False,
        gguf: bool = False,
    ) -> None:
        self.maybe_login(login_flag)

        target_dir = (target_dir or self.DEFAULT_TARGET_DIR).resolve()
        self.ensure_dir(target_dir)

        registry = self.GGUF_MODEL_REGISTRY if gguf else self.MODEL_REGISTRY
        
        if all_models:
            to_get = list(registry.keys())
        else:
            to_get = only or []

        missing = [a for a in to_get if a not in registry]
        if missing:
            print(f"[error] Unknown model alias(es): {missing}")
            print("Available aliases:", ", ".join(registry.keys()))
            sys.exit(1)

        resolved = {}
        for alias in to_get:
            repo_id = registry[alias]
            path = self.download_one(alias, repo_id, target_dir, revision=revision, gguf=gguf)
            resolved[alias] = str(path)

        self.write_registry(self.REGISTRY_FILE, resolved)

        print("\nDone. Local models:")
        for k, v in resolved.items():
            print(f"  - {k}: {v}")

    def _read_registry(self) -> Dict[str, str]:
        """Load and return the local models registry JSON; returns {} if file missing or empty."""
        if self.REGISTRY_FILE.exists():
            return json.loads(self.REGISTRY_FILE.read_text())
        return {}

    def load_local_model(self, alias: str, device_map: str = "auto") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load a model previously downloaded by this script.
        Always uses local disk (no network).
        """
        reg = self._read_registry()
        if alias not in reg:
            raise ValueError(
                f"Alias '{alias}' not found in {self.REGISTRY_FILE}. "
                f"Run: python download_models.py --only {alias}"
            )
        local_path = reg[alias]
        tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            local_files_only=True,
            device_map=device_map,
            torch_dtype="auto",
        )
        return tok, model

    def load_config(self, config_path="config/RGConfig.yaml") -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def test_model(self, config: dict, prompt: Optional[str] = None) -> None:
        model_name = config["llama"]["name"]
        gen_args = config["llama"]["generation_args"]

        test_prompt = prompt or "Hello, how are you today?"

        print(f"[INFO] Loading model from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)

        pipe = pipeline(
            task=config["llama"]["task_type"],
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )

        prompt_template = config["llama"]["prompt_template"]["format"].format(
            user_prefix=config["llama"]["prompt_template"]["user_prefix"],
            prompt=test_prompt,
            assistant_prefix=config["llama"]["prompt_template"]["assistant_prefix"],
        )

        print(f"[INFO] Running generation test with prompt:\n{prompt_template}\n")
        output = pipe(prompt_template, **gen_args)
        print("[OUTPUT]")
        print(output[0]["generated_text"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download HF models locally and load from disk.")
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="Download all models in registry.")
    sel.add_argument("--only", nargs="+", help="Download only these aliases (space separated).")
    p.add_argument("--target-dir", default=None, help="Where to store models (default: ./models)")
    p.add_argument("--revision", default=None, help="Optional HF revision (branch/tag/commit) for all downloads.")
    p.add_argument("--login", action="store_true", help="Run huggingface login first (useful for gated models).")
    p.add_argument("--gguf", action="store_true", help="Download GGUF models instead of standard models.")
    p.add_argument("--test", action="store_true", help="After download, run a quick generation test using the config.")
    p.add_argument("--config", default="config/RGConfig.yaml", help="Path to YAML config for --test.")
    p.add_argument("--prompt", default="Hello, how are you today?", help="Test prompt for --test mode.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mgr = ModelManager()

    mgr.download_models(
        all_models=args.all,
        only=args.only,
        target_dir=Path(args.target_dir) if args.target_dir else None,
        revision=args.revision,
        login_flag=args.login,
        gguf=args.gguf,
    )

    if args.test:
        cfg = mgr.load_config(args.config)
        mgr.test_model(cfg, prompt=args.prompt)