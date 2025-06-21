# AI/core/system_detector.py - Universal system detection and auto-configuration

import platform
import os
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Universal system configuration"""
    os_type: str  # 'windows', 'linux', 'macos'
    architecture: str  # 'x86_64', 'arm64', 'aarch64'
    device: str  # 'cuda', 'mps', 'cpu'
    device_name: Optional[str]
    cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_memory_gb: Optional[float]
    cache_dir: str
    optimal_workers: int
    optimal_batch_size: int
    optimal_max_candidates: int
    threading_config: Dict[str, Any]
    torch_config: Dict[str, Any]


class UniversalSystemDetector:
    """Auto-detects system capabilities and configures optimal settings"""

    def __init__(self):
        self.config = self._detect_system()
        self._apply_optimizations()

    def _detect_system(self) -> SystemConfig:
        """Detect and analyze system capabilities"""
        logger.info("🔍 Detecting system configuration...")

        # Basic system info
        os_type = self._detect_os()
        architecture = self._detect_architecture()
        cpu_cores = os.cpu_count() or 4

        # Memory info
        total_memory, available_memory = self._detect_memory(os_type)

        # Device detection (GPU/CPU)
        device, device_name, gpu_memory = self._detect_compute_device(
            total_memory)

        # Cache directory
        cache_dir = self._get_cache_directory(os_type)

        # Optimal configurations based on detected hardware
        optimal_config = self._calculate_optimal_settings(
            os_type, device, cpu_cores, total_memory, gpu_memory
        )

        config = SystemConfig(
            os_type=os_type,
            architecture=architecture,
            device=device,
            device_name=device_name,
            cpu_cores=cpu_cores,
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            gpu_memory_gb=gpu_memory,
            cache_dir=cache_dir,
            **optimal_config
        )

        logger.info(
            f"✅ System detected: {config.os_type} {config.architecture}")
        logger.info(
            f"🎯 Compute device: {config.device} ({config.device_name})")
        logger.info(
            f"💾 Memory: {config.total_memory_gb:.1f}GB total, {config.available_memory_gb:.1f}GB available")
        if config.gpu_memory_gb:
            logger.info(f"🚀 GPU Memory: {config.gpu_memory_gb:.1f}GB")

        return config

    def _detect_os(self) -> str:
        """Detect operating system"""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        elif system == "linux":
            return "linux"
        elif system == "windows":
            return "windows"
        else:
            logger.warning(f"Unknown OS: {system}, defaulting to linux")
            return "linux"

    def _detect_architecture(self) -> str:
        """Detect CPU architecture"""
        arch = platform.machine().lower()
        if arch in ["x86_64", "amd64"]:
            return "x86_64"
        elif arch in ["arm64", "aarch64"]:
            return "arm64"
        else:
            logger.warning(
                f"Unknown architecture: {arch}, defaulting to x86_64")
            return "x86_64"

    def _detect_memory(self, os_type: str) -> Tuple[float, float]:
        """Detect system memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            return total_gb, available_gb
        except ImportError:
            logger.warning("psutil not available, estimating memory")
            # Fallback estimation
            if os_type == "linux":
                try:
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                total_kb = int(line.split()[1])
                                total_gb = total_kb / (1024**2)
                                return total_gb, total_gb * 0.7  # Estimate 70% available
                except:
                    pass
            return 8.0, 6.0  # Safe default

    def _detect_compute_device(self, total_memory_gb: float) -> Tuple[str, Optional[str], Optional[float]]:
        """Detect best compute device (CUDA, MPS, or CPU)"""
        device_name = None
        gpu_memory = None

        try:
            import torch

            # Check CUDA (NVIDIA GPUs)
            if torch.cuda.is_available():
                device = "cuda"
                device_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(
                    0).total_memory / (1024**3)
                logger.info(
                    f"🚀 CUDA detected: {device_name} with {gpu_memory:.1f}GB VRAM")
                return device, device_name, gpu_memory

            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                device_name = "Apple Metal Performance Shaders"
                # MPS shares system memory, estimate based on system RAM
                gpu_memory = min(total_memory_gb * 0.6,
                                 32.0)  # Conservative estimate
                logger.info(f"🍎 MPS detected: {device_name}")
                return device, device_name, gpu_memory

            # Fallback to CPU
            device = "cpu"
            device_name = f"{platform.processor()} ({os.cpu_count()} cores)"
            logger.info(f"💻 Using CPU: {device_name}")
            return device, device_name, None

        except ImportError:
            logger.warning("PyTorch not available, defaulting to CPU")
            return "cpu", "CPU (PyTorch not detected)", None

    def _get_cache_directory(self, os_type: str) -> str:
        """Get appropriate cache directory for each OS"""
        if os_type == "windows":
            cache_dir = os.path.expandvars(r"%LOCALAPPDATA%\transformers")
        elif os_type == "macos":
            cache_dir = os.path.expanduser("~/Library/Caches/transformers")
        else:  # linux and others
            cache_dir = os.path.expanduser("~/.cache/transformers")

        # Create directory if it doesn't exist
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _calculate_optimal_settings(
        self,
        os_type: str,
        device: str,
        cpu_cores: int,
        total_memory: float,
        gpu_memory: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate optimal settings based on detected hardware"""

        # Base configurations
        if device == "cuda":
            # NVIDIA GPU configuration
            if gpu_memory and gpu_memory >= 24:  # RTX 4090, RTX 5090, etc.
                optimal_workers = min(cpu_cores // 2, 8)
                optimal_batch_size = 16
                optimal_max_candidates = 20
            elif gpu_memory and gpu_memory >= 12:  # RTX 4070 Ti, RTX 3080, etc.
                optimal_workers = min(cpu_cores // 2, 6)
                optimal_batch_size = 8
                optimal_max_candidates = 15
            elif gpu_memory and gpu_memory >= 8:  # RTX 4060 Ti, RTX 3070, etc.
                optimal_workers = min(cpu_cores // 2, 4)
                optimal_batch_size = 4
                optimal_max_candidates = 10
            else:  # Lower-end GPUs
                optimal_workers = min(cpu_cores // 2, 2)
                optimal_batch_size = 2
                optimal_max_candidates = 8

            threading_config = {
                "use_threading": True,
                "max_concurrent_requests": optimal_workers * 2,
                "enable_gpu_optimization": True
            }

            torch_config = {
                "torch_dtype": "float16",  # Use half precision on GPU
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "enable_autocast": True
            }

        elif device == "mps":
            # Apple Silicon configuration
            optimal_workers = min(cpu_cores // 2, 4)
            optimal_batch_size = 6
            optimal_max_candidates = 12

            threading_config = {
                "use_threading": True,
                "max_concurrent_requests": optimal_workers,
                "enable_gpu_optimization": True
            }

            torch_config = {
                "torch_dtype": "float32",  # MPS works better with float32
                "device_map": None,
                "low_cpu_mem_usage": True,
                "enable_autocast": False  # Not always stable on MPS
            }

        else:
            # CPU configuration
            if total_memory >= 32:  # High-end server
                optimal_workers = min(cpu_cores, 8)
                optimal_batch_size = 8
                optimal_max_candidates = 15
            elif total_memory >= 16:  # Standard system
                optimal_workers = min(cpu_cores // 2, 4)
                optimal_batch_size = 4
                optimal_max_candidates = 10
            else:  # Low memory system
                optimal_workers = min(cpu_cores // 2, 2)
                optimal_batch_size = 2
                optimal_max_candidates = 6

            threading_config = {
                "use_threading": True,
                "max_concurrent_requests": optimal_workers,
                "enable_gpu_optimization": False
            }

            torch_config = {
                "torch_dtype": "float32",
                "device_map": None,
                "low_cpu_mem_usage": True,
                "enable_autocast": False
            }

        # OS-specific adjustments
        if os_type == "windows":
            # Windows might need more conservative threading
            threading_config["max_concurrent_requests"] = max(
                1, threading_config["max_concurrent_requests"] // 2)

        return {
            "optimal_workers": optimal_workers,
            "optimal_batch_size": optimal_batch_size,
            "optimal_max_candidates": optimal_max_candidates,
            "threading_config": threading_config,
            "torch_config": torch_config
        }

    def _apply_optimizations(self):
        """Apply system-specific optimizations"""
        config = self.config

        # Set environment variables for optimal performance
        os.environ["OMP_NUM_THREADS"] = str(config.cpu_cores)
        os.environ["MKL_NUM_THREADS"] = str(config.cpu_cores)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Device-specific optimizations
        if config.device == "cuda":
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async CUDA operations
            os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN v8
        elif config.device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        # OS-specific optimizations
        if config.os_type == "linux":
            os.environ["MALLOC_ARENA_MAX"] = "4"  # Reduce memory fragmentation

        logger.info("✅ System optimizations applied")

    def get_model_config(self) -> Dict[str, Any]:
        """Get optimal model configuration for detected system"""
        config = self.config

        model_config = {
            "device": config.device,
            "cache_dir": config.cache_dir,
            "torch_dtype": config.torch_config["torch_dtype"],
            "low_cpu_mem_usage": config.torch_config["low_cpu_mem_usage"],
        }

        if config.device == "cuda":
            model_config.update({
                "device_map": "auto",
                "max_memory": {0: f"{int(config.gpu_memory_gb * 0.9)}GB"} if config.gpu_memory_gb else None
            })

        return model_config

    def get_generation_config(self) -> Dict[str, Any]:
        """Get optimal generation parameters for detected system"""
        config = self.config

        if config.device == "cuda" and config.gpu_memory_gb and config.gpu_memory_gb >= 12:
            # High-end GPU settings
            return {
                "num_return_sequences": config.optimal_max_candidates,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.9,
                "temperature": 0.8,
                "num_beams": min(config.optimal_max_candidates, 8),
                "max_length": 512,
                "use_threading": True,
                "batch_size": config.optimal_batch_size
            }
        elif config.device in ["cuda", "mps"]:
            # Mid-range GPU/MPS settings
            return {
                "num_return_sequences": config.optimal_max_candidates,
                "do_sample": True,
                "top_k": 45,
                "top_p": 0.85,
                "temperature": 0.75,
                "num_beams": min(config.optimal_max_candidates, 6),
                "max_length": 384,
                "use_threading": True,
                "batch_size": config.optimal_batch_size
            }
        else:
            # CPU settings
            return {
                "num_return_sequences": config.optimal_max_candidates,
                "do_sample": True,
                "top_k": 40,
                "top_p": 0.8,
                "temperature": 0.7,
                "num_beams": min(config.optimal_max_candidates, 4),
                "max_length": 256,
                "use_threading": config.cpu_cores > 4,
                "batch_size": config.optimal_batch_size
            }

    def get_api_config(self) -> Dict[str, Any]:
        """Get optimal API configuration for detected system"""
        config = self.config

        return {
            "max_candidates": config.optimal_max_candidates,
            "max_batch_size": config.optimal_batch_size,
            "workers": 1,  # Single worker for model sharing
            "timeout_keep_alive": 60 if config.device == "cpu" else 30,
            "request_timeout": 120 if config.device == "cpu" else 60,
            # ~200MB per GB
            "cache_size": min(2000, int(config.available_memory_gb * 200)),
            "rate_limit": "200/minute" if config.device == "cuda" else "100/minute"
        }

    def log_system_info(self):
        """Log comprehensive system information"""
        config = self.config

        logger.info("=" * 60)
        logger.info("🖥️  UNIVERSAL SYSTEM CONFIGURATION")
        logger.info("=" * 60)
        logger.info(
            f"Operating System: {config.os_type.title()} ({platform.platform()})")
        logger.info(f"Architecture: {config.architecture}")
        logger.info(f"CPU Cores: {config.cpu_cores}")
        logger.info(f"Total Memory: {config.total_memory_gb:.1f}GB")
        logger.info(f"Available Memory: {config.available_memory_gb:.1f}GB")
        logger.info(f"Compute Device: {config.device.upper()}")
        logger.info(f"Device Name: {config.device_name}")
        if config.gpu_memory_gb:
            logger.info(f"GPU Memory: {config.gpu_memory_gb:.1f}GB")
        logger.info(f"Cache Directory: {config.cache_dir}")
        logger.info("-" * 60)
        logger.info(f"Optimal Workers: {config.optimal_workers}")
        logger.info(f"Optimal Batch Size: {config.optimal_batch_size}")
        logger.info(f"Max Candidates: {config.optimal_max_candidates}")
        logger.info("=" * 60)


# Global system detector instance
system_detector = UniversalSystemDetector()

# Convenience functions for easy access


def get_system_config() -> SystemConfig:
    """Get the detected system configuration"""
    return system_detector.config


def get_optimal_device() -> str:
    """Get the optimal device for the current system"""
    return system_detector.config.device


def get_model_config() -> Dict[str, Any]:
    """Get optimal model configuration"""
    return system_detector.get_model_config()


def get_generation_config() -> Dict[str, Any]:
    """Get optimal generation configuration"""
    return system_detector.get_generation_config()


def get_api_config() -> Dict[str, Any]:
    """Get optimal API configuration"""
    return system_detector.get_api_config()


def log_system_info():
    """Log system information"""
    system_detector.log_system_info()
