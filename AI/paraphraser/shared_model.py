# AI/paraphraser/shared_model.py - Universal model manager

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
from typing import Optional, Dict, Any
import gc
from AI.core.system_detector import get_system_config, get_model_config, log_system_info

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global model instances
polish_tokenizer: Optional[T5Tokenizer] = None
polish_model: Optional[T5ForConditionalGeneration] = None
_model_config: Optional[Dict[str, Any]] = None


def get_universal_model_config() -> Dict[str, Any]:
    """Get model configuration optimized for current system"""
    global _model_config

    if _model_config is None:
        system_config = get_system_config()
        _model_config = get_model_config()

        # Add model-specific configurations
        _model_config.update({
            "model_name": "RinaldiDev/flan_paradetox_full",
            "revision": "main",
            "trust_remote_code": False,

        })

        # Device-specific model optimizations
        if system_config.device == "cuda":
            _model_config.update({
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_8bit": system_config.gpu_memory_gb < 12,  # Use 8-bit for lower VRAM
                "load_in_4bit": system_config.gpu_memory_gb < 8,   # Use 4-bit for very low VRAM
            })
        elif system_config.device == "mps":
            _model_config.update({
                "torch_dtype": torch.float32,  # MPS works better with float32
                "device_map": None,
            })
        else:  # CPU
            _model_config.update({
                "torch_dtype": torch.float32,
                "device_map": None,
            })

    return _model_config


def initialize_universal_model(
    model_name: Optional[str] = None,
    force_reload: bool = False
) -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
    """
    Initialize model with universal compatibility across all systems.
    """
    global polish_tokenizer, polish_model

    # Return cached model if available
    if not force_reload and polish_tokenizer is not None and polish_model is not None:
        logger.info("✅ Model already loaded, returning cached instance")
        return polish_tokenizer, polish_model

    # Get system-optimized configuration
    system_config = get_system_config()
    model_config = get_universal_model_config()

    if model_name:
        model_config["model_name"] = model_name

    logger.info("🚀 Initializing universal model...")
    log_system_info()

    try:
        # Load tokenizer
        logger.info(f"📝 Loading tokenizer: {model_config['model_name']}")
        polish_tokenizer = T5Tokenizer.from_pretrained(
            model_config["model_name"],
            cache_dir=model_config["cache_dir"],
            revision=model_config["revision"],

            trust_remote_code=model_config["trust_remote_code"],
            legacy=False
        )

        # Configure tokenizer
        if polish_tokenizer.pad_token is None:
            logger.info("🔧 Configuring tokenizer pad token")
            polish_tokenizer.add_special_tokens({'pad_token': '<pad>'})

        # Load model with system-specific optimizations
        logger.info(
            f"🧠 Loading model with {system_config.device.upper()} optimization...")

        model_kwargs = {
            "cache_dir": model_config["cache_dir"],
            "revision": model_config["revision"],

            "trust_remote_code": model_config["trust_remote_code"],
            "low_cpu_mem_usage": model_config["low_cpu_mem_usage"],
        }

        # Add device-specific parameters
        if system_config.device == "cuda":
            model_kwargs.update({
                "torch_dtype": model_config["torch_dtype"],
                "device_map": model_config.get("device_map"),
            })

            # Add quantization if needed for lower VRAM
            if model_config.get("load_in_8bit"):
                logger.info(
                    "⚡ Using 8-bit quantization for GPU memory efficiency")
                model_kwargs["load_in_8bit"] = True
            elif model_config.get("load_in_4bit"):
                logger.info(
                    "⚡ Using 4-bit quantization for very low GPU memory")
                model_kwargs["load_in_4bit"] = True

        else:
            model_kwargs["torch_dtype"] = model_config["torch_dtype"]

        polish_model = T5ForConditionalGeneration.from_pretrained(
            model_config["model_name"],
            **model_kwargs
        )

        # Resize embeddings if tokenizer was modified
        if len(polish_tokenizer) != polish_model.config.vocab_size:
            logger.info("🔧 Resizing model embeddings to match tokenizer")
            polish_model.resize_token_embeddings(len(polish_tokenizer))

        # Move model to device if not already handled by device_map
        if not model_config.get("device_map"):
            logger.info(f"📱 Moving model to {system_config.device}")
            polish_model.to(system_config.device)

        # Set model to evaluation mode
        polish_model.eval()

        # Apply device-specific optimizations
        _apply_device_optimizations(polish_model, system_config)

        # Log success
        num_parameters = sum(p.numel()
                             for p in polish_model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in polish_model.parameters())
        memory_footprint = total_parameters * \
            4 / (1024**3)  # Rough estimate in GB

        logger.info(f"✅ Model loaded successfully!")
        logger.info(
            f"📊 Parameters: {total_parameters:,} total, {num_parameters:,} trainable")
        logger.info(f"💾 Estimated memory: {memory_footprint:.1f}GB")
        logger.info(
            f"🎯 Device: {system_config.device} ({system_config.device_name})")

        return polish_tokenizer, polish_model

    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        # Cleanup on failure
        cleanup_model()
        raise RuntimeError(f"Universal model initialization failed: {e}")


def _apply_device_optimizations(model, system_config):
    """Apply device-specific optimizations to the model"""

    if system_config.device == "cuda":
        # CUDA optimizations
        logger.info("⚡ Applying CUDA optimizations...")

        # Disable gradient computation for inference
        for param in model.parameters():
            param.requires_grad = False

        # Enable autocast if supported
        if hasattr(torch.cuda.amp, 'autocast'):
            logger.info("🚀 CUDA autocast enabled")

        # Optimize memory usage
        if hasattr(torch.backends.cudnn, 'benchmark'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    elif system_config.device == "mps":
        # MPS optimizations
        logger.info("🍎 Applying MPS optimizations...")

        for param in model.parameters():
            param.requires_grad = False

        # MPS-specific settings
        torch.backends.mps.enabled = True

    else:
        # CPU optimizations
        logger.info("💻 Applying CPU optimizations...")

        for param in model.parameters():
            param.requires_grad = False

        # CPU-specific PyTorch settings
        torch.set_num_threads(system_config.cpu_cores)
        torch.set_num_interop_threads(system_config.cpu_cores)

        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')


def cleanup_model():
    """Clean up model and tokenizer to free memory"""
    global polish_tokenizer, polish_model

    logger.info("🧹 Cleaning up model...")

    if polish_model is not None:
        del polish_model
        polish_model = None

    if polish_tokenizer is not None:
        del polish_tokenizer
        polish_tokenizer = None

    # Clear device-specific caches
    system_config = get_system_config()
    if system_config.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif system_config.device == "mps" and hasattr(torch.backends, 'mps'):
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

    # Force garbage collection
    gc.collect()

    logger.info("✅ Model cleanup complete")


def get_model_memory_usage() -> Dict[str, float]:
    """Get current model memory usage"""
    system_config = get_system_config()
    memory_info = {}

    try:
        import psutil
        process = psutil.Process()
        memory_info["cpu_memory_mb"] = process.memory_info().rss / (1024**2)
        memory_info["cpu_memory_percent"] = process.memory_percent()
    except ImportError:
        memory_info["cpu_memory_mb"] = 0
        memory_info["cpu_memory_percent"] = 0

    if system_config.device == "cuda" and torch.cuda.is_available():
        memory_info["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / \
            (1024**2)
        memory_info["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / \
            (1024**2)
        memory_info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(
            0).total_memory / (1024**2)

    return memory_info


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    system_config = get_system_config()
    model_config = get_universal_model_config()

    info = {
        "model_loaded": polish_model is not None and polish_tokenizer is not None,
        "model_name": model_config.get("model_name"),
        "device": system_config.device,
        "device_name": system_config.device_name,
        "system_type": f"{system_config.os_type} {system_config.architecture}",
        "memory_usage": get_model_memory_usage(),
    }

    if polish_model is not None:
        info.update({
            "vocab_size": len(polish_tokenizer) if polish_tokenizer else 0,
            "model_dtype": str(next(polish_model.parameters()).dtype),
            "model_device": str(next(polish_model.parameters()).device),
            "total_parameters": sum(p.numel() for p in polish_model.parameters()),
            "trainable_parameters": sum(p.numel() for p in polish_model.parameters() if p.requires_grad),
        })

    return info


# Auto-initialize model when module is imported
try:
    logger.info("🚀 Auto-initializing universal model...")
    polish_tokenizer, polish_model = initialize_universal_model()
    logger.info("✅ Universal model auto-initialization complete")
except Exception as e:
    logger.error(f"❌ Auto-initialization failed: {e}")
    logger.info("ℹ️  Model will be initialized on first request")
    polish_tokenizer = None
    polish_model = None
