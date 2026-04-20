

import os
import json
import yaml
import time
import psutil
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from utils import get_custom_logging
from .model_interface import LlamaCppChatInterface

get_logger, _, _, _ = get_custom_logging()

class PromptGenerator:
    """Prompt generator using v1/chat/completions interface for prompt generation and modification."""
    
    def __init__(self, model_key="prompt_generator", config_path="config/PGConfig.yaml", log_file: Optional[str] = None, seed: Optional[int] = None):
        self.log_file = log_file
        self.logger = get_logger("PromptGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        from pathlib import Path
        if config_path and not Path(config_path).is_absolute():
            _project_root = Path(__file__).resolve().parents[2]
            config_path = str(_project_root / config_path)

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError(f"Configuration file is empty: {config_path}")
            if model_key not in config:
                raise ValueError(f"Model '{model_key}' not found in configuration. Available keys: {list(config.keys())}")
            self.model_cfg = dict(config[model_key])
            if config.get("device_config"):
                self.model_cfg["device_config"] = config["device_config"]
            if seed is not None:
                self.model_cfg.setdefault("generation_args", {})["seed"] = seed
            if not self.model_cfg.get("name"):
                raise ValueError(f"Model configuration missing 'name' field for {model_key}")
                
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(
                "Failed to parse YAML configuration (%s): %s. "
                "Restore a valid file from the repository (e.g. `git checkout -- config/PGConfig.yaml`) "
                "if an older build rewrote this file with `yaml.dump`.",
                config_path,
                e,
            )
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            raise

        try:
            self.model_interface = LlamaCppChatInterface(
                self.model_cfg, log_file, cache_key_suffix="prompt_generator"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize model interface: {e}")
            raise

        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0


    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        
        return self.model_interface.chat_completion(messages, **kwargs)



    
    def _extract_content_from_xml_tags(self, response: str, tag_name: str) -> str:
        
        try:
            import re
            
            response = response.strip()
            
            pattern = f'<{tag_name}>(.*?)</{tag_name}>'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, tag_name):
                    self.logger.debug(f"Successfully extracted {tag_name} content: {content[:50]}...")
                    return content
            
            pattern = f'<{tag_name.lower()}>(.*?)</{tag_name.lower()}>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, tag_name):
                    self.logger.debug(f"Successfully extracted {tag_name} content (case-insensitive): {content[:50]}...")
                    return content
            
            pattern = f'<{tag_name}\\s*>(.*?)</{tag_name}\\s*>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and self._validate_extracted_content(content, tag_name):
                    self.logger.debug(f"Successfully extracted {tag_name} content (whitespace-tolerant): {content[:50]}...")
                    return content
            
                if len(tag_name) > 3:
                    pattern = f'<{tag_name[:3]}.*?>(.*?)</{tag_name[:3]}.*?>'
                    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        if content and self._validate_extracted_content(content, tag_name):
                            self.logger.debug(f"Successfully extracted {tag_name} content (partial match): {content[:50]}...")
                            return content
                
                if tag_name in ['synonyms', 'antonyms']:
                    pattern = f'([a-zA-Z]+)</{tag_name[:3]}.*?>'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        if content and self._validate_extracted_content(content, tag_name):
                            self.logger.debug(f"Successfully extracted {tag_name} content (malformed XML): {content[:50]}...")
                            return content
            
            self.logger.debug("Failed to extract valid %s content from response: %.200s...", tag_name, response)
            return ""
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {tag_name} tags: {e}")
            return ""
    
    def _validate_extracted_content(self, content: str, tag_name: str) -> bool:
        
        if not content or len(content.strip()) < 2:
            return False
            
        if tag_name in ['variant', 'paraphrase', 'modified', 'trans', 'triple_bracket']:
            question_marks = ['?', '？', '؟', ';']
            if not any(content.endswith(qm) for qm in question_marks):
                self.logger.warning(f"Extracted content does not end with question mark: {content}")
                return False
                
                
                
        return True
