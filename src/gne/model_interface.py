"""
Model Interface Abstraction Layer

Provides OpenAI-compatible v1/chat/completions interface for model-agnostic architecture.
Currently implements llama_cpp provider with chat completions support.
"""

import os
import time
import psutil
import re
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


class ModelInterface(ABC):
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat completion response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass


class LlamaCppChatInterface(ModelInterface):
    """LlamaCpp implementation of chat completions interface."""
    
    _MODEL_CACHE = {}
    _MODEL_CACHE_ACCESS_COUNT = {}
    _MODEL_CACHE_LOCK = None
    
    def __init__(self, model_cfg: Dict[str, Any], log_file: Optional[str] = None, cache_key_suffix: Optional[str] = None):
        """
        Initialize the LlamaCpp chat interface.

        Args:
            model_cfg: Model configuration dictionary
            log_file: Optional log file path
            cache_key_suffix: If set (e.g. "response_generator", "prompt_generator"), the cache key
                is path + suffix so RG and PG get separate instances even when using the same model
                path. Ensures correct behavior for different seeds and any RG/PG model combination.
        """
        self.logger = get_logger("LlamaCppChatInterface", log_file)
        self.model_cfg = model_cfg
        
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        model_path = model_cfg["name"]
        
        if not os.path.isabs(model_path):
            from pathlib import Path
            project_root = Path(__file__).resolve().parents[2]
            absolute_model_path = str(project_root / model_path)
        else:
            absolute_model_path = model_path
        
        import threading
        if self._MODEL_CACHE_LOCK is None:
            self._MODEL_CACHE_LOCK = threading.Lock()
        
        # Key by (path, role) so RG and PG never share an instance — supports different models and seeds.
        self._model_cache_key = f"{absolute_model_path}|{cache_key_suffix}" if cache_key_suffix else absolute_model_path

        self.logger.info(f"Loading llama.cpp model: {absolute_model_path}" + (f" (role={cache_key_suffix})" if cache_key_suffix else ""))
        self._load_model(absolute_model_path, cache_key=self._model_cache_key)
        self.model = self._MODEL_CACHE[self._model_cache_key]
        self.generation_args = model_cfg.get("generation_args", {})
        
        self.last_memory_check = time.time()
        self.memory_check_interval = 60
    
    @classmethod
    def clear_model_cache(cls):
        """Clear the entire model cache to force fresh model loading."""
        logger = get_logger("LlamaCppChatInterface")
        with cls._MODEL_CACHE_LOCK:
            cls._MODEL_CACHE.clear()
            cls._MODEL_CACHE_ACCESS_COUNT.clear()
            logger.info("Model cache cleared - all models will be reloaded")

    @classmethod
    def close_and_clear_model_cache(cls):
        """
        Close all cached Llama models and clear the cache (e.g. on worker exit).
        Swallows AttributeError from llama-cpp-python's LlamaModel.close() when
        the internal 'sampler' attribute is missing (known library bug on teardown).
        """
        logger = get_logger("LlamaCppChatInterface")
        with cls._MODEL_CACHE_LOCK:
            for cache_key, model in list(cls._MODEL_CACHE.items()):
                try:
                    if hasattr(model, "close"):
                        model.close()
                except AttributeError as e:
                    # Known llama-cpp-python bug: LlamaModel.close() accesses self.sampler
                    # which may not exist if the object was partially initialized or torn down.
                    logger.debug(
                        "Ignoring AttributeError when closing model %s (known llama-cpp-python teardown): %s",
                        cache_key[:80], e,
                    )
                except Exception as e:
                    logger.warning("Error closing model %s: %s", cache_key[:80], e)
            cls._MODEL_CACHE.clear()
            cls._MODEL_CACHE_ACCESS_COUNT.clear()
            logger.info("Model cache closed and cleared")
    
    def _cleanup_model_cache_if_needed(self):
        """Clean up unused models from cache, but preserve main RG/PG models."""
        max_cache_size = 5
        
        if len(self._MODEL_CACHE) > max_cache_size:
            main_models = set()
            for model_path in self._MODEL_CACHE.keys():
                if any(keyword in model_path.lower() for keyword in ['q4_k_m', 'q3_k_s']):
                    main_models.add(model_path)
            
            models_to_remove = []
            for model_path, access_count in self._MODEL_CACHE_ACCESS_COUNT.items():
                if model_path not in main_models:
                    models_to_remove.append((model_path, access_count))
            
            models_to_remove.sort(key=lambda x: x[1])
            excess_count = len(self._MODEL_CACHE) - max_cache_size
            
            for i in range(min(excess_count, len(models_to_remove))):
                model_path, access_count = models_to_remove[i]
                del self._MODEL_CACHE[model_path]
                del self._MODEL_CACHE_ACCESS_COUNT[model_path]
                self.logger.info(f"Removed model {model_path} from cache (access count: {access_count})")
    
    def _load_model(self, model_path: str, cache_key: Optional[str] = None):
        """Load model using llama.cpp with device-specific optimizations.
        cache_key: Key for model cache (default: model_path). Use path|role so RG and PG get separate instances.
        """
        if cache_key is None:
            cache_key = model_path
        try:
            if not os.path.isabs(model_path):
                from pathlib import Path
                project_root = Path(__file__).resolve().parents[2]
                model_path = str(project_root / model_path)

            if cache_key in self._MODEL_CACHE:
                # Reuse only when same path and same role (e.g. same operator loading same model again).
                self._MODEL_CACHE_ACCESS_COUNT[cache_key] = self._MODEL_CACHE_ACCESS_COUNT.get(cache_key, 0) + 1
                self.logger.debug("Reusing cached model: %s (accesses=%d)", cache_key, self._MODEL_CACHE_ACCESS_COUNT[cache_key])
                self._cleanup_model_cache_if_needed()
                return
            
            if not os.path.exists(model_path):
                gguf_path = f"{model_path}.gguf"
                if os.path.exists(gguf_path):
                    model_path = gguf_path
                    # Keep cache_key unchanged (it already includes role suffix if any)
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            device_config = self._get_device_specific_config()
            
            llama_params = {
                "model_path": model_path,
                "n_ctx": device_config.get("context_length", 4096),
                "n_batch": device_config.get("n_batch", 1024),
                "n_threads": device_config.get("num_threads", None),
                "n_gpu_layers": device_config.get("gpu_layers", 0),
                "verbose": False,
                "use_mmap": device_config.get("use_mmap", True),
                "use_mlock": device_config.get("use_mlock", False),
                "low_vram": device_config.get("low_vram", False),
                "f16_kv": device_config.get("f16_kv", True),
                "logits_all": False,
                "vocab_only": False,
                "use_mmap": device_config.get("use_mmap", True),
                "use_mlock": device_config.get("use_mlock", False),
            }
            
            device_name = device_config.get("device", "cpu")
            if device_name == "mps":
                llama_params.update({
                    "n_gpu_layers": device_config.get("gpu_layers", 20),
                    "main_gpu": 0,
                    "tensor_split": None,
                })
            elif device_name == "cuda" or (isinstance(device_name, str) and device_name.startswith("cuda:")):
                llama_params.update({
                    "n_gpu_layers": device_config.get("gpu_layers", -1),
                    "main_gpu": device_config.get("main_gpu", 0),
                    "tensor_split": device_config.get("tensor_split", None),
                })
            else:
                llama_params.update({
                    "n_gpu_layers": 0,
                    "n_threads": device_config.get("num_threads", None),
                })
            
            self.logger.info(f"Loading model with llama.cpp on {device_config.get('device', 'cpu')}...")
            self.logger.debug(f"Llama.cpp parameters: {llama_params}")
            model = Llama(**llama_params)
            
            self._MODEL_CACHE[cache_key] = model
            self._MODEL_CACHE_ACCESS_COUNT[cache_key] = 1
            self.logger.info(f"Model loaded successfully: {model_path}")
            self._cleanup_model_cache_if_needed()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_device_specific_config(self) -> Dict[str, Any]:
        """Get device-specific configuration for llama.cpp."""
        from utils.device_utils import device_manager
        
        device = device_manager.get_optimal_device()
        config = self.model_cfg.get("device_config", {})
        
        device_config = {
            "device": device,
            "context_length": 4096,
            "n_batch": 1024,
            "num_threads": None,
            "use_mmap": True,
            "use_mlock": False,
            "low_vram": False,
            "f16_kv": True,
        }
        
        if device == "mps":
            device_config.update({
                "gpu_layers": config.get("mps", {}).get("gpu_layers", -1),
                "use_mmap": True,
                "use_mlock": True,
                "low_vram": False,
                "f16_kv": True,
            })
        elif device == "cuda" or (isinstance(device, str) and device.startswith("cuda:")):
            cuda_config = config.get("cuda", {})
            main_gpu = 0
            if isinstance(device, str) and device.startswith("cuda:"):
                try:
                    main_gpu = int(device.split(":", 1)[1])
                except (IndexError, ValueError):
                    main_gpu = 0
            device_config.update({
                "gpu_layers": cuda_config.get("gpu_layers", -1),
                "use_mmap": True,
                "use_mlock": False,
                "low_vram": cuda_config.get("low_vram", False),
                "f16_kv": True,
                "tensor_split": cuda_config.get("tensor_split", None),
                "main_gpu": main_gpu,
            })
        else:
            cpu_config = config.get("cpu", {})
            device_config.update({
                "gpu_layers": 0,
                "num_threads": cpu_config.get("num_threads", None),
                "use_mmap": True,
                "use_mlock": False,
                "low_vram": False,
                "f16_kv": False,
            })
        if config.get("n_batch") is not None:
            device_config["n_batch"] = config["n_batch"]
        return device_config
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get model cache statistics."""
        return {
            "cached_models": len(cls._MODEL_CACHE),
            "model_paths": list(cls._MODEL_CACHE.keys()),
            "access_counts": dict(cls._MODEL_CACHE_ACCESS_COUNT),
            "total_accesses": sum(cls._MODEL_CACHE_ACCESS_COUNT.values())
        }
    
    @classmethod
    def clear_cache(cls, preserve_main_models: bool = True):
        """Clear model cache, optionally preserving main RG/PG models."""
        if preserve_main_models:
            main_models = {}
            main_access_counts = {}
            for model_path in cls._MODEL_CACHE.keys():
                if any(keyword in model_path.lower() for keyword in ['q4_k_m', 'q3_k_s']):
                    main_models[model_path] = cls._MODEL_CACHE[model_path]
                    main_access_counts[model_path] = cls._MODEL_CACHE_ACCESS_COUNT.get(model_path, 0)
            
            cls._MODEL_CACHE.clear()
            cls._MODEL_CACHE_ACCESS_COUNT.clear()
            
            cls._MODEL_CACHE.update(main_models)
            cls._MODEL_CACHE_ACCESS_COUNT.update(main_access_counts)
        else:
            cls._MODEL_CACHE.clear()
            cls._MODEL_CACHE_ACCESS_COUNT.clear()
    
    def _check_memory_usage(self):
        """Check memory usage and perform cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_memory_check > self.memory_check_interval:
            try:
                memory_percent = psutil.virtual_memory().percent
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                
                self.logger.debug(f"Memory usage: {memory_percent:.1f}%, Available: {available_memory_gb:.1f}GB")
                
                if memory_percent > 85:
                    self.logger.warning(f"High memory usage detected: {memory_percent:.1f}%")
                    self._perform_memory_cleanup()
                
                self.last_memory_check = current_time
            except ImportError:
                self.logger.debug("psutil not available for memory monitoring")
            except Exception as e:
                self.logger.warning(f"Memory check failed: {e}")
    
    def _perform_memory_cleanup(self):
        """Perform memory cleanup."""
        try:
            import gc
            gc.collect()
            self.logger.info("Memory cleanup performed")
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to appropriate prompt format for GGUF models.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "").strip()
            
            if not content:
                continue
                
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                if prompt_parts and prompt_parts[-1].startswith("System:"):
                    prompt_parts[-1] += f"\n\nUser: {content}"
                else:
                    prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        formatted_prompt = "\n".join(prompt_parts)
        
        if not formatted_prompt.endswith("Assistant:"):
            formatted_prompt += "\nAssistant:"
        
        return formatted_prompt
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text using a simple heuristic.
        This is a rough approximation - actual tokenization may vary.
        
        Args:
            text: Input text to count tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        char_count = len(text)
        
        word_count = len(text.split())
        
        char_based_estimate = char_count // 4
        word_based_estimate = word_count * 1.3
        
        estimated_tokens = max(char_based_estimate, int(word_based_estimate))
        
        overhead = 50
        
        return estimated_tokens + overhead
    
    def _validate_context_window(self, formatted_prompt: str, max_new_tokens: int, context_length: int = 4096) -> tuple[str, int]:
        """
        Validate and adjust prompt/tokens to fit within context window.
        
        Args:
            formatted_prompt: The formatted prompt text
            max_new_tokens: Requested maximum new tokens to generate
            context_length: Total context window length
            
        Returns:
            Tuple of (adjusted_prompt, adjusted_max_tokens)
        """
        prompt_tokens = self._estimate_token_count(formatted_prompt)
        total_requested = prompt_tokens + max_new_tokens
        
        self.logger.debug(f"Token validation: prompt={prompt_tokens}, max_new={max_new_tokens}, total={total_requested}, context={context_length}")
        
        if total_requested <= context_length:
            return formatted_prompt, max_new_tokens
        
        available_for_prompt = context_length - max_new_tokens
        
        if available_for_prompt <= 0:
            self.logger.warning(f"max_new_tokens ({max_new_tokens}) exceeds context window ({context_length}). Reducing to {context_length - 100}")
            return formatted_prompt, context_length - 100
        
        if prompt_tokens > available_for_prompt:
            self.logger.warning(f"Prompt too long ({prompt_tokens} tokens). Truncating to fit context window.")
            
            truncated_prompt = self._truncate_prompt(formatted_prompt, available_for_prompt)
            truncated_tokens = self._estimate_token_count(truncated_prompt)
            
            self.logger.debug(f"Truncated prompt: {truncated_tokens} tokens")
            return truncated_prompt, max_new_tokens
        
        return formatted_prompt, max_new_tokens
    
    def _truncate_prompt(self, prompt: str, max_tokens: int) -> str:
        """
        Truncate prompt to fit within token limit.
        Prioritizes keeping the end of the prompt (user input).
        
        Args:
            prompt: Original prompt text
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated prompt text
        """
        if not prompt:
            return prompt
        
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token
        
        if len(prompt) <= max_chars:
            return prompt
        
        user_sections = prompt.split("User:")
        if len(user_sections) > 1:
            system_part = user_sections[0] + "User:"
            user_part = user_sections[-1]
            
            if len(system_part) > max_chars * 0.7:
                system_part = system_part[:int(max_chars * 0.7)]
            
            truncated = system_part + user_part
            if len(truncated) > max_chars:
                truncated = truncated[:max_chars]
            
            return truncated
        else:
            return prompt[:max_chars]
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat completion response using llama.cpp.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        try:
            self._check_memory_usage()

            if hasattr(self, "_model_cache_key"):
                with self._MODEL_CACHE_LOCK:
                    self._MODEL_CACHE_ACCESS_COUNT[self._model_cache_key] = self._MODEL_CACHE_ACCESS_COUNT.get(self._model_cache_key, 0) + 1
                self._cleanup_model_cache_if_needed()
            
            formatted_prompt = self._convert_messages_to_prompt(messages)
            
            generation_kwargs = self.generation_args.copy()
            generation_kwargs.update(kwargs)
            
            device_config = self._get_device_specific_config()
            context_length = device_config.get("context_length", 4096)
            
            max_new_tokens = generation_kwargs.get("max_new_tokens", 2048)
            validated_prompt, validated_max_tokens = self._validate_context_window(
                formatted_prompt, max_new_tokens, context_length
            )
            
            self.logger.debug(f"Generating chat completion for prompt: {validated_prompt[:100]}...")
            
            response = self.model(
                validated_prompt,
                max_tokens=validated_max_tokens,
                temperature=generation_kwargs.get("temperature", 0.7),
                top_p=generation_kwargs.get("top_p", 0.9),
                top_k=generation_kwargs.get("top_k", 40),
                repeat_penalty=generation_kwargs.get("repetition_penalty", 1.1),
                stop=["</s>", "<|endoftext|>", "User:", "System:"],
                seed=generation_kwargs["seed"] if "seed" in generation_kwargs else random.randint(0, 2**31 - 1),
                echo=False,
            )
            
            if isinstance(response, dict) and 'choices' in response:
                generated_text = response['choices'][0]['text']
            elif isinstance(response, str):
                generated_text = response
            else:
                self.logger.warning(f"Unexpected response format: {type(response)}")
                generated_text = str(response)
            
            generated_text = generated_text.strip()
            
            self.generation_count += 1
            self.total_tokens_generated += len(generated_text.split())
            
            self.logger.debug(f"Generated chat completion: {generated_text[:100]}...")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}", exc_info=True)
            return ""
        finally:
            end_time = time.time()
            generation_time = end_time - start_time
            self.total_generation_time += generation_time
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the model interface."""
        return {
            "generation_count": self.generation_count,
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time": self.total_generation_time,
            "average_tokens_per_generation": (
                self.total_tokens_generated / self.generation_count 
                if self.generation_count > 0 else 0
            ),
            "average_time_per_generation": (
                self.total_generation_time / self.generation_count 
                if self.generation_count > 0 else 0
            )
        }
