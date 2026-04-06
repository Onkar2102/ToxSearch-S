"""
device_utils.py

Device detection and configuration utilities for CUDA, MPS, and CPU support.
Provides centralized device detection and configuration for all components.
"""

import torch
import yaml
from typing import Optional, Dict, Any
from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


class DeviceManager:
    """
    Centralized device management for CUDA, MPS, and CPU support.
    
    Provides device detection, configuration, and fallback mechanisms
    for all components in the project.
    When running under MPI, use set_mpi_device(rank, size) so rank 0 uses CPU
    and workers use one GPU each (rank 1 -> cuda:0, rank 2 -> cuda:1, ...).
    """
    
    _instance = None
    _device_cache = None
    _config_cache = None
    _mpi_rank = None
    _mpi_size = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = get_logger("DeviceManager")
            self.initialized = True
            self._load_config()
    
    def set_mpi_device(self, rank: int, size: int) -> None:
        """
        Set device by MPI rank: rank 0 = CPU (master), workers = CUDA, or MPS on Mac if no CUDA, else CPU.
        Call this at MPI entry (e.g. in master_worker.run()) before any model load.
        """
        self._mpi_rank = rank
        self._mpi_size = size
        self._device_cache = None
        if rank == 0:
            self._device_cache = "cpu"
            self.logger.info("MPI rank 0 (master): using CPU")
        else:
            worker_id = rank - 1
            try:
                if torch.cuda.is_available() and worker_id < torch.cuda.device_count():
                    self._device_cache = f"cuda:{worker_id}"
                    self.logger.info("MPI rank %d (worker): using cuda:%d", rank, worker_id)
                else:
                    # No CUDA (e.g. Mac): try MPS for workers so Apple Silicon GPU is used
                    import platform
                    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available()
                    if platform.system() == "Darwin" and has_mps:
                        self._device_cache = "mps"
                        self.logger.info("MPI rank %d (worker): using MPS (Apple Silicon)", rank)
                    else:
                        self._device_cache = "cpu"
                        self.logger.warning(
                            "MPI rank %d: no CUDA/MPS; using CPU",
                            rank)
            except Exception as e:
                self.logger.warning("MPI rank %d: GPU check failed (%s), using CPU", rank, e)
                self._device_cache = "cpu"
    
    def _load_config(self):
        """Load device configuration from RGConfig.yaml (from project root, not cwd)."""
        try:
            from pathlib import Path
            config_path = Path(__file__).resolve().parents[2] / "config" / "RGConfig.yaml"
            text = config_path.read_text(encoding="utf-8")
            try:
                config = yaml.safe_load(text)
            except yaml.YAMLError:
                # Full file can be invalid if a prior run used yaml.dump on the whole file.
                # ``device_config`` lives at the top; parse only through ``response_generator:``.
                head_lines = []
                for line in text.splitlines(keepends=True):
                    stripped = line.strip()
                    if stripped.startswith("response_generator:") or stripped.startswith("response_generator "):
                        break
                    head_lines.append(line)
                head = "".join(head_lines)
                config = yaml.safe_load(head) if head.strip() else {}
                if config:
                    self.logger.warning(
                        "RGConfig.yaml: full parse failed; loaded device_config from header only. "
                        "Restore config/RGConfig.yaml from the repo or fix YAML below response_generator."
                    )
            self._config_cache = (config or {}).get("device_config", {})
            self.logger.debug("Device configuration loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load device config, using defaults: {e}")
            self._config_cache = {}
    
    def get_optimal_device(self) -> str:
        """
        Get the best available device with comprehensive error handling.
        If set_mpi_device() was called, returns device for this MPI rank
        (rank 0 = cpu, rank 1 = cuda:0, rank 2 = cuda:1, ...).
        Otherwise priority order: MPS -> CUDA -> CPU.
        
        Returns:
            str: Device name ('mps', 'cuda', 'cuda:0', 'cuda:1', or 'cpu')
        """
        if self._device_cache is not None:
            return self._device_cache
        if self._mpi_rank is not None:
            if self._mpi_rank == 0:
                return "cpu"
            worker_id = self._mpi_rank - 1
            try:
                if torch.cuda.is_available() and worker_id < torch.cuda.device_count():
                    return f"cuda:{worker_id}"
                import platform
                if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        
        preferred_device = self._config_cache.get("preferred_device")
        if preferred_device and self._is_device_available(preferred_device):
            self._device_cache = preferred_device
            self.logger.info(f"Using preferred device from config: {preferred_device}")
            return self._device_cache
        
        if self._config_cache.get("auto_detect", True):
            try:
                import platform
                is_macos = platform.system() == "Darwin"
                has_mps_backend = hasattr(torch.backends, 'mps')
                is_mps_built = torch.backends.mps.is_built() if has_mps_backend else False
                is_mps_available = torch.backends.mps.is_available() if has_mps_backend else False
                
                self.logger.debug(f"MPS Detection - macOS: {is_macos}, has_backend: {has_mps_backend}, built: {is_mps_built}, available: {is_mps_available}")
                
                if is_macos and has_mps_backend and is_mps_built and is_mps_available:
                    self._device_cache = "mps"
                    self.logger.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
                    self._apply_device_optimizations("mps")
                    return self._device_cache
                elif is_macos:
                    self.logger.warning(f"MPS not fully available - macOS: {is_macos}, backend: {has_mps_backend}, built: {is_mps_built}, available: {is_mps_available}")
                elif not is_macos and (has_mps_backend or is_mps_built or is_mps_available):
                    self.logger.debug(f"MPS backend exists on Linux but is not available for use. OS: {platform.system()}")
            except Exception as e:
                self.logger.warning(f"MPS check failed: {e}")
                import traceback
                self.logger.debug(f"MPS check traceback: {traceback.format_exc()}")
            
            try:
                if torch.cuda.is_available():
                    self._device_cache = "cuda"
                    gpu_name = torch.cuda.get_device_name()
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    self.logger.info(f"Using CUDA (GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB)")
                    self._apply_device_optimizations("cuda")
                    return self._device_cache
            except Exception as e:
                self.logger.warning(f"CUDA check failed: {e}")
        
        self._device_cache = "cpu"
        self.logger.info("Using CPU (fallback)")
        self._apply_device_optimizations("cpu")
        return self._device_cache
    
    def _is_device_available(self, device: str) -> bool:
        """Check if a specific device is available"""
        try:
            if device == "mps":
                import platform
                is_macos = platform.system() == "Darwin"
                has_mps_backend = hasattr(torch.backends, 'mps')
                is_mps_built = torch.backends.mps.is_built() if has_mps_backend else False
                is_mps_available = torch.backends.mps.is_available() if has_mps_backend else False
                return is_macos and has_mps_backend and is_mps_built and is_mps_available
            elif device == "cuda":
                return torch.cuda.is_available()
            elif device == "cpu":
                return True
            return False
        except Exception as e:
            self.logger.debug(f"Device availability check failed for {device}: {e}")
            return False
    
    def _apply_device_optimizations(self, device: str):
        """Apply device-specific optimizations"""
        try:
            if device == "cuda":
                cuda_config = self._config_cache.get("cuda", {})
                if cuda_config.get("enable_tf32", True):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                if cuda_config.get("enable_cudnn_benchmark", True):
                    torch.backends.cudnn.benchmark = True
                memory_fraction = cuda_config.get("memory_fraction")
                if memory_fraction:
                    torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            elif device == "cpu":
                cpu_config = self._config_cache.get("cpu", {})
                num_threads = cpu_config.get("num_threads")
                if num_threads:
                    torch.set_num_threads(num_threads)
                elif num_threads is None:
                    import os
                    cpu_count = os.cpu_count()
                    torch.set_num_threads(min(cpu_count, 8))
            
            self.logger.debug(f"Applied optimizations for {device}")
        except Exception as e:
            self.logger.warning(f"Failed to apply {device} optimizations: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get comprehensive device information.
        
        Returns:
            Dict containing device information and capabilities
        """
        device = self.get_optimal_device()
        info = {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_available": False,
            "mps_available": False,
            "cpu_count": torch.get_num_threads()
        }
        
        try:
            info["cuda_available"] = torch.cuda.is_available()
            if info["cuda_available"]:
                info["cuda_version"] = torch.version.cuda
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name()
                info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        except Exception as e:
            self.logger.warning(f"CUDA info collection failed: {e}")
        
        try:
            import platform
            has_mps_backend = hasattr(torch.backends, 'mps')
            is_mps_built = torch.backends.mps.is_built() if has_mps_backend else False
            is_mps_available = torch.backends.mps.is_available() if has_mps_backend else False
            
            info["mps_available"] = has_mps_backend and is_mps_built and is_mps_available
            info["mps_backend_exists"] = has_mps_backend
            info["mps_is_built"] = is_mps_built
            info["platform"] = platform.system()
            info["platform_version"] = platform.release()
            info["machine"] = platform.machine()
        except Exception as e:
            self.logger.warning(f"MPS info collection failed: {e}")
        
        return info
    
    def move_to_device(self, tensor_or_model, device: Optional[str] = None) -> Any:
        """
        Move tensor or model to specified device with fallback handling.
        
        Args:
            tensor_or_model: PyTorch tensor or model to move
            device: Target device (if None, uses optimal device)
            
        Returns:
            Moved tensor or model
        """
        if device is None:
            device = self.get_optimal_device()
        
        try:
            if hasattr(tensor_or_model, 'to'):
                result = tensor_or_model.to(device)
                self.logger.debug(f"Successfully moved to {device}")
                return result
            else:
                self.logger.warning(f"Object {type(tensor_or_model)} has no 'to' method")
                return tensor_or_model
        except Exception as e:
            self.logger.warning(f"Failed to move to {device}, falling back to CPU: {e}")
            try:
                result = tensor_or_model.to("cpu")
                self.logger.info("Successfully moved to CPU")
                return result
            except Exception as cpu_e:
                self.logger.error(f"Failed to move to CPU: {cpu_e}")
                raise RuntimeError(f"Unable to move object to any device: {cpu_e}")
    
    def get_generation_kwargs(self, device: Optional[str] = None) -> Dict[str, Any]:
        """
        Get generation kwargs optimized for the specified device.
        
        Args:
            device: Target device (if None, uses optimal device)
            
        Returns:
            Dict containing device-optimized generation parameters
        """
        if device is None:
            device = self.get_optimal_device()
        
        kwargs = {
            "pad_token_id": None,
            "eos_token_id": None,
            "use_cache": True,
        }
        
        if device == "cuda":
            kwargs.update({
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.15,
            })
        elif device == "mps":
            kwargs.update({
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.15,
            })
        else:
            kwargs.update({
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
            })
        
        return kwargs
    
    def clear_cache(self):
        """Clear device cache to force re-detection. Does not clear MPI rank."""
        self._device_cache = None
        self.logger.info("Device cache cleared")


device_manager = DeviceManager()


def get_optimal_device() -> str:
    """Convenience function to get optimal device."""
    return device_manager.get_optimal_device()


def get_device_info() -> Dict[str, Any]:
    """Convenience function to get device information."""
    return device_manager.get_device_info()


def move_to_device(tensor_or_model, device: Optional[str] = None) -> Any:
    """Convenience function to move tensor/model to device."""
    return device_manager.move_to_device(tensor_or_model, device)


def get_generation_kwargs(device: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get device-optimized generation kwargs."""
    return device_manager.get_generation_kwargs(device)
