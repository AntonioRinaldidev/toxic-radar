# AI/paraphraser/generator.py - Universal auto-adaptive generator

from collections import OrderedDict
from typing import List, Optional, Dict, Any
import logging
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch
from transformers import pipeline
from AI.paraphraser.shared_model import polish_model, polish_tokenizer, cleanup_model
from AI.paraphraser.scorer import score_toxicity, taunt_equivalence_score
from AI.core.system_detector import get_system_config, get_generation_config

# Setup logging
logger = logging.getLogger(__name__)

# Global pipeline with thread safety
_pipeline_lock = threading.Lock()
_global_pipeline = None
_system_config = None
_generation_config = None


def get_universal_pipeline():
    """Get universally optimized pipeline for any system"""
    global _global_pipeline, _system_config, _generation_config

    with _pipeline_lock:
        if _global_pipeline is None:
            _system_config = get_system_config()
            _generation_config = get_generation_config()

            logger.info(
                f"🔧 Creating universal pipeline for {_system_config.device.upper()}")

            # Pipeline configuration based on system
            pipeline_kwargs = {
                "task": "text2text-generation",
                "model": polish_model,
                "tokenizer": polish_tokenizer,
                "framework": "pt"
            }

            # Device-specific optimizations
            if _system_config.device == "cuda":
                pipeline_kwargs.update({
                    "device": 0,  # Use first CUDA device
                    "torch_dtype": torch.float16 if _system_config.gpu_memory_gb > 8 else torch.float32,
                    "batch_size": _generation_config["batch_size"]
                })
            elif _system_config.device == "mps":
                pipeline_kwargs.update({
                    "device": "mps",
                    "torch_dtype": torch.float32,
                    "batch_size": _generation_config["batch_size"]
                })
            else:  # CPU
                pipeline_kwargs.update({
                    "device": "cpu",
                    "torch_dtype": torch.float32,
                    "batch_size": _generation_config["batch_size"]
                })

            _global_pipeline = pipeline(**pipeline_kwargs)

            logger.info(
                f"✅ Universal pipeline created for {_system_config.device.upper()}")
            logger.info(f"🎯 Optimized for: {_system_config.device_name}")

    return _global_pipeline

# Universal caching with system-adaptive sizes


def get_cache_size() -> int:
    """Get optimal cache size based on available memory"""
    system_config = get_system_config()
    # Allocate ~10% of available memory for cache, but cap it reasonably
    cache_size = int(system_config.available_memory_gb * 100)
    return min(max(cache_size, 500), 5000)  # Between 500-5000 entries


@lru_cache(maxsize=None)  # Will be set dynamically
def cached_build_prompt(text: str) -> str:
    """Cache prompt building with universal compatibility"""
    return f"Rewrite politely:\nInput: {text.strip()}\nPolite:"


@lru_cache(maxsize=None)  # Will be set dynamically
def cached_clean_text(text: str) -> str:
    """Cache text cleaning with universal compatibility"""
    cleaned = text.strip().strip('"""\'''""''')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# Set cache sizes based on system
_cache_size = get_cache_size()
cached_build_prompt.__wrapped__.__func__ = lru_cache(
    maxsize=_cache_size)(cached_build_prompt.__wrapped__)
cached_clean_text.__wrapped__.__func__ = lru_cache(
    maxsize=_cache_size)(cached_clean_text.__wrapped__)


def generate_paraphrases_universal(
    text: str,
    num_return_sequences: Optional[int] = None,
    **override_params
) -> List[str]:
    """
    Universal paraphrase generation that adapts to any system.
    Auto-detects optimal parameters based on hardware.
    """

    if not text.strip():
        logger.warning("📝 Empty text provided")
        return []

    # Get system-optimized configuration
    system_config = get_system_config()
    generation_config = get_generation_config()

    # Use system-optimal number of sequences if not specified
    if num_return_sequences is None:
        num_return_sequences = generation_config["num_return_sequences"]

    # Merge generation config with any overrides
    params = {**generation_config, **override_params}
    params["num_return_sequences"] = min(
        num_return_sequences, params["num_return_sequences"])

    # Adaptive parameters based on text length
    text_length = len(text.split())
    if text_length > 100:
        # Long text: use conservative parameters
        params.update({
            "max_length": min(params["max_length"], 256),
            "num_beams": min(params["num_beams"], 3),
            "num_return_sequences": min(params["num_return_sequences"], 3)
        })
        logger.debug("📝 Long text detected: using conservative parameters")
    elif text_length > 50:
        # Medium text: balanced parameters
        params.update({
            "max_length": min(params["max_length"], 384),
            "num_beams": min(params["num_beams"], 4)
        })
        logger.debug("📝 Medium text detected: using balanced parameters")

    start_time = time.time()

    try:
        # Get optimized pipeline
        pipe = get_universal_pipeline()
        prompt = cached_build_prompt(text)

        logger.debug(
            f"🎯 Generating with {system_config.device.upper()} ({params['num_return_sequences']} candidates)")

        # Generation with device-specific context
        generation_kwargs = {
            "do_sample": params["do_sample"],
            "num_return_sequences": params["num_return_sequences"],
            "early_stopping": True,
            "pad_token_id": polish_tokenizer.pad_token_id,
            "eos_token_id": polish_tokenizer.eos_token_id,
        }

        # Add sampling parameters if do_sample is True
        if params["do_sample"]:
            generation_kwargs.update({
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "top_k": params["top_k"],
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 2
            })
        else:
            generation_kwargs.update({
                "num_beams": params["num_beams"],
                "length_penalty": 1.0
            })

        # Device-specific generation
        if system_config.device == "cuda":
            # CUDA generation with autocast
            with torch.cuda.amp.autocast(enabled=system_config.gpu_memory_gb > 8):
                with torch.no_grad():
                    raw_outputs = pipe(prompt, **generation_kwargs)
        elif system_config.device == "mps":
            # MPS generation
            with torch.no_grad():
                raw_outputs = pipe(prompt, **generation_kwargs)
        else:
            # CPU generation
            with torch.no_grad():
                raw_outputs = pipe(prompt, **generation_kwargs)

        # Process outputs with adaptive threading
        def process_single_output(output):
            cleaned = cached_clean_text(output["generated_text"])
            if (cleaned and
                len(cleaned.split()) >= 2 and
                    cleaned.lower() != text.lower()):
                return cleaned
            return None

        # Use threading based on system capabilities
        if params["use_threading"] and len(raw_outputs) > 2 and system_config.cpu_cores > 2:
            max_workers = min(system_config.optimal_workers, len(raw_outputs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                processed = list(executor.map(
                    process_single_output, raw_outputs))
        else:
            processed = [process_single_output(
                output) for output in raw_outputs]

        # Filter and deduplicate
        candidates = list(OrderedDict.fromkeys([p for p in processed if p]))

        if not candidates:
            logger.warning("🚫 No valid candidates after cleaning")
            return []

        # Toxicity filtering with adaptive parameters
        min_improvement = 0.1 if system_config.device == "cpu" else 0.05  # More lenient on CPU
        valid_candidates = _filter_by_toxicity(
            text, candidates, min_improvement)

        if not valid_candidates:
            logger.warning(
                "🚫 No candidates with sufficient toxicity reduction")
            return []

        # Similarity scoring and ranking
        result = _rank_by_similarity(
            text, valid_candidates, num_return_sequences)

        # Cleanup based on system
        _adaptive_cleanup(system_config)

        elapsed = time.time() - start_time
        logger.info(
            f"✅ Generated {len(result)} candidates in {elapsed:.2f}s ({system_config.device.upper()})")

        return result

    except Exception as e:
        logger.error(f"❌ Universal generation failed: {e}")
        return []


def _filter_by_toxicity(text: str, candidates: List[str], min_improvement: float) -> List[str]:
    """Filter candidates by toxicity improvement"""
    try:
        original_tox = score_toxicity([text])[0].get('toxicity', 0.0)
        candidate_tox = score_toxicity(candidates)

        valid_candidates = []
        for candidate, tox_dict in zip(candidates, candidate_tox):
            improvement = original_tox - tox_dict.get('toxicity', 1.0)
            if improvement >= min_improvement:
                valid_candidates.append(candidate)

        return valid_candidates
    except Exception as e:
        logger.warning(
            f"⚠️ Toxicity filtering failed: {e}, returning all candidates")
        return candidates


def _rank_by_similarity(text: str, candidates: List[str], max_results: int) -> List[str]:
    """Rank candidates by similarity to original"""
    try:
        similarity_scores = taunt_equivalence_score(text, candidates)
        scored_candidates = list(zip(candidates, similarity_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [cand for cand, _ in scored_candidates[:max_results]]
    except Exception as e:
        logger.warning(
            f"⚠️ Similarity ranking failed: {e}, using original order")
        return candidates[:max_results]


def _adaptive_cleanup(system_config):
    """Perform cleanup based on system type"""
    if system_config.device == "cuda":
        # CUDA cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    elif system_config.device == "mps":
        # MPS cleanup
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    else:
        # CPU cleanup
        import gc
        gc.collect()

# Batch processing with universal optimization


def generate_batch_universal(
    texts: List[str],
    num_candidates_each: Optional[int] = None,
    **kwargs
) -> List[List[str]]:
    """
    Universal batch processing optimized for any system.
    """
    system_config = get_system_config()

    if num_candidates_each is None:
        num_candidates_each = min(3, system_config.optimal_max_candidates)

    logger.info(
        f"🔄 Processing batch of {len(texts)} texts on {system_config.device.upper()}")

    # Determine optimal batch processing strategy
    max_workers = min(system_config.optimal_workers, len(texts))

    if system_config.device == "cuda" and system_config.gpu_memory_gb > 12:
        # High-end GPU: process multiple texts in parallel
        chunk_size = min(4, len(texts))
    elif system_config.device in ["cuda", "mps"]:
        # Mid-range GPU/MPS: moderate parallelization
        chunk_size = min(2, len(texts))
    else:
        # CPU: conservative parallelization
        chunk_size = 1
        max_workers = min(max_workers, 2)

    # Process in chunks
    results = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]

        if len(chunk) == 1:
            # Single text processing
            result = generate_paraphrases_universal(
                chunk[0], num_candidates_each, **kwargs)
            results.append(result)
        else:
            # Parallel processing within chunk
            with ThreadPoolExecutor(max_workers=min(max_workers, len(chunk))) as executor:
                futures = [
                    executor.submit(generate_paraphrases_universal,
                                    text, num_candidates_each, **kwargs)
                    for text in chunk
                ]

                chunk_results = []
                for j, future in enumerate(futures):
                    try:
                        result = future.result(timeout=120)
                        chunk_results.append(result)
                        logger.debug(f"✅ Completed {i+j+1}/{len(texts)}")
                    except Exception as e:
                        logger.error(f"❌ Failed processing text {i+j+1}: {e}")
                        chunk_results.append([])

                results.extend(chunk_results)

        # Adaptive cleanup between chunks
        if chunk_size > 1:
            _adaptive_cleanup(system_config)

    return results

# Memory-efficient mode for resource-constrained systems


def generate_paraphrases_efficient(
    text: str,
    num_return_sequences: int = 2,
    max_length: int = 128
) -> List[str]:
    """
    Memory-efficient generation for resource-constrained systems.
    """
    logger.info("💡 Using memory-efficient mode")

    return generate_paraphrases_universal(
        text,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        do_sample=False,  # Greedy decoding is faster
        num_beams=2,      # Minimal beam search
        use_threading=False,  # Reduce overhead
        temperature=0.7,
        top_p=0.8,
        top_k=30
    )

# High-quality mode for high-end systems


def generate_paraphrases_quality(
    text: str,
    num_return_sequences: int = 10,
    max_length: int = 512
) -> List[str]:
    """
    High-quality generation for high-end systems.
    """
    system_config = get_system_config()

    if system_config.device == "cpu" or (system_config.device == "cuda" and system_config.gpu_memory_gb < 8):
        logger.warning("⚠️ High-quality mode may be slow on this system")

    logger.info("🎯 Using high-quality mode")

    return generate_paraphrases_universal(
        text,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        do_sample=True,
        num_beams=8,
        use_threading=True,
        temperature=0.8,
        top_p=0.95,
        top_k=60,
        repetition_penalty=1.1
    )

# Main function with automatic mode selection


def generate_paraphrases(
    text: str,
    num_return_sequences: int = 3,
    mode: str = "auto",
    **kwargs
) -> List[str]:
    """
    Main generation function with automatic system adaptation.

    Args:
        text: Input text to paraphrase
        num_return_sequences: Number of paraphrases to generate
        mode: Generation mode ('auto', 'efficient', 'quality', 'universal')
        **kwargs: Additional parameters to override defaults

    Returns:
        List of paraphrased texts
    """

    system_config = get_system_config()

    if mode == "auto":
        # Auto-select mode based on system capabilities
        if system_config.device == "cuda" and system_config.gpu_memory_gb >= 12:
            logger.debug("🚀 Auto-selected: quality mode (high-end GPU)")
            return generate_paraphrases_quality(text, num_return_sequences, **kwargs)
        elif system_config.total_memory_gb < 8 or system_config.device == "cpu":
            logger.debug(
                "💡 Auto-selected: efficient mode (resource-constrained)")
            return generate_paraphrases_efficient(text, min(num_return_sequences, 3), **kwargs)
        else:
            logger.debug("⚖️ Auto-selected: universal mode (balanced)")
            return generate_paraphrases_universal(text, num_return_sequences, **kwargs)

    elif mode == "efficient":
        return generate_paraphrases_efficient(text, num_return_sequences, **kwargs)
    elif mode == "quality":
        return generate_paraphrases_quality(text, num_return_sequences, **kwargs)
    else:  # universal or any other mode
        return generate_paraphrases_universal(text, num_return_sequences, **kwargs)

# Batch processing main function


def generate_paraphrases_batch(
    texts: List[str],
    num_candidates_each: int = 3,
    mode: str = "auto",
    **kwargs
) -> List[List[str]]:
    """
    Batch processing with automatic system adaptation.
    """
    if mode == "auto":
        return generate_batch_universal(texts, num_candidates_each, **kwargs)
    else:
        # Process individually with specified mode
        return [
            generate_paraphrases(
                text, num_candidates_each, mode=mode, **kwargs)
            for text in texts
        ]
