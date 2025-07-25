# AI/paraphraser/service.py - Universal auto-adaptive service

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
import logging
from AI.paraphraser.voting import custom_utility_score
from AI.paraphraser.generator import generate_paraphrases
from AI.paraphraser.scorer import score_toxicity, taunt_equivalence_score, score_fluency
from AI.reasoning.reasoning import apply_reasoning
from AI.core.system_detector import get_system_config, get_api_config, log_system_info
from AI.paraphraser.shared_model import get_model_info, get_model_memory_usage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Get system-optimized API configuration
api_config = get_api_config()
system_config = get_system_config()

# Initialize FastAPI with system info
app = FastAPI(
    title="Universal AI Paraphraser",
    description=f"Auto-adaptive paraphrasing service optimized for {system_config.os_type.title()} {system_config.device.upper()}",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware with system-adaptive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request & Response Schemas


class ParaphraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Text to paraphrase")
    num_candidates: int = Field(
        default=3, ge=1, le=api_config["max_candidates"], description="Number of paraphrases to generate")
    mode: str = Field(
        default="auto", pattern="^(auto|efficient|quality|universal)$", description="Generation mode")
    custom_params: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom generation parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "You're such an idiot!",
                "num_candidates": 3,
                "mode": "auto",
                "custom_params": {"temperature": 0.8}
            }
        }


class ParaphraseCandidate(BaseModel):
    text: str
    toxicity: float
    similarity: float
    fluency: float
    rank: int


class ParaphraseResponse(BaseModel):
    original: str
    candidates: List[ParaphraseCandidate]
    metadata: Dict[str, Any]
    system_info: Dict[str, str]


class AnalysisResponse(BaseModel):
    text: str
    analysis: Dict[str, Any]
    metadata: Dict[str, Any]
    system_info: Dict[str, str]


class SystemInfoResponse(BaseModel):
    system_config: Dict[str, Any]
    model_info: Dict[str, Any]
    api_config: Dict[str, Any]
    performance_stats: Dict[str, Any]

# Utility Functions


def get_fallback_message(original_toxicity: float) -> str:
    """
    Return appropriate fallback message based on original toxicity level

    Args:
        original_toxicity: Toxicity score of the original text (0.0 to 1.0)

    Returns:
        Appropriate fallback message in English
    """
    if original_toxicity >= 0.7:  # High toxicity threshold
        return "You should reconsider saying something like that"
    elif original_toxicity <= 0.3:  # Low toxicity threshold
        return "You don't need to reduce toxicity of this phrase"
    else:  # Medium toxicity - still suggest reconsideration
        return "You should reconsider saying something like that"


def create_fallback_response(
    original_text: str,
    original_toxicity: float,
    processing_time: float,
    generation_mode: str
) -> ParaphraseResponse:
    """
    Create a fallback response when paraphrasing fails or isn't needed

    Args:
        original_text: The original input text
        original_toxicity: Toxicity score of the original text
        processing_time: Time taken for processing
        generation_mode: The generation mode used

    Returns:
        ParaphraseResponse with fallback message
    """
    fallback_message = get_fallback_message(original_toxicity)

    # Create a single fallback candidate
    fallback_candidate = ParaphraseCandidate(
        text=fallback_message,
        toxicity=0.1,  # Fallback messages are designed to be non-toxic
        similarity=0.0,  # Fallback messages are completely different from original
        fluency=1.0,    # Fallback messages are perfectly fluent
        rank=1
    )

    # Determine the fallback reason
    if original_toxicity <= 0.3:
        fallback_reason = "low_toxicity_no_reduction_needed"
    elif original_toxicity >= 0.9:
        fallback_reason = "high_toxicity_paraphrasing_failed"
    else:
        fallback_reason = "paraphrasing_failed"

    metadata = {
        "processing_time_seconds": round(processing_time, 3),
        "original_toxicity": round(original_toxicity, 3),
        "best_candidate_toxicity": 0.1,  # Fallback message toxicity
        "toxicity_reduction": round(max(0.0, original_toxicity - 0.1), 3),
        "generation_mode": generation_mode,
        "candidates_generated": 0,
        "candidates_returned": 1,
        "fallback_used": True,
        "fallback_reason": fallback_reason,
        "fallback_message": fallback_message
    }

    return ParaphraseResponse(
        original=original_text,
        candidates=[fallback_candidate],
        metadata=metadata,
        system_info=get_system_metadata()
    )


def extract_toxicity_scores(adjusted_results: List[Dict[str, Any]]) -> List[float]:
    """Extract toxicity scores from reasoning results"""
    return [
        result.get('adjusted_labels', {}).get('toxicity', 0.0)
        for result in adjusted_results
    ]


def format_candidates(
    candidates: List[str],
    toxicity: List[float],
    similarity: List[float],
    fluency: List[float],
    ranking: List[int]
) -> List[ParaphraseCandidate]:
    """Format candidates with scores and ranking"""
    return [
        ParaphraseCandidate(
            text=candidates[idx],
            toxicity=toxicity[idx],
            similarity=similarity[idx],
            fluency=fluency[idx],
            rank=rank + 1
        )
        for rank, idx in enumerate(ranking)
    ]


def validate_request(request: ParaphraseRequest) -> None:
    """Validate request with system-adaptive limits"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # System-adaptive text length limits
    max_length = 5000 if system_config.device == "cpu" else 10000
    if len(request.text) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text too long. Maximum length for {system_config.device.upper()}: {max_length} characters"
        )

    if request.num_candidates > api_config["max_candidates"]:
        raise HTTPException(
            status_code=400,
            detail=f"Too many candidates requested. Maximum for this system: {api_config['max_candidates']}"
        )


def get_system_metadata() -> Dict[str, str]:
    """Get system information for responses"""
    return {
        "device": system_config.device.upper(),
        "device_name": system_config.device_name or "Unknown",
        "os": f"{system_config.os_type.title()} {system_config.architecture}",
        "optimization": f"Optimized for {system_config.device.upper()}"
    }

# Main Endpoints


@app.post("/paraphrase", response_model=ParaphraseResponse)
async def paraphrase(request: ParaphraseRequest):
    """
    Generate detoxified paraphrases with universal system optimization.
    """
    start_time = time.time()
    validate_request(request)

    try:

        logger.info("Analyzing original text with SCP reasoning...")
        original_raw_toxicity = score_toxicity([request.text])[0]
        original_reasoning_result = apply_reasoning(original_raw_toxicity)
        original_adjusted_labels = original_reasoning_result.get(
            'adjusted_labels', original_raw_toxicity)
        original_adjusted_toxicity = original_adjusted_labels.get(
            'toxicity', 0.0)

        logger.info(
            f"Original toxicity: {original_raw_toxicity.get('toxicity', 0.0):.3f} → {original_adjusted_toxicity:.3f} (after reasoning)")

        if original_adjusted_toxicity <= 0.3:
            logger.info("💡 Text has low toxicity, returning fallback message")
            processing_time = time.time() - start_time
            return create_fallback_response(
                request.text,
                original_adjusted_toxicity,
                processing_time,
                request.mode
            )
        # Generate paraphrases with system-adaptive parameters
        kwargs = request.custom_params or {}
        candidates = generate_paraphrases(
            text=request.text,
            num_return_sequences=request.num_candidates,
            mode=request.mode,
            **kwargs
        )

        if not candidates:
            logger.warning(
                "⚠️ No candidates generated, returning fallback message")
            processing_time = time.time() - start_time
            return create_fallback_response(
                request.text,
                original_adjusted_toxicity,
                processing_time,
                request.mode
            )

        # Score and rank candidates
        raw_toxicity_scores = score_toxicity(candidates)
        adjusted_results = [apply_reasoning(
            raw_scores) for raw_scores in raw_toxicity_scores]
        adjusted_toxicity_scores = extract_toxicity_scores(adjusted_results)

        similarity_scores = taunt_equivalence_score(request.text, candidates)
        fluency_scores = score_fluency(candidates)

        # Rank candidates
        score_lists = {
            "toxicity": adjusted_toxicity_scores,
            "similarity": similarity_scores,
            "fluency": fluency_scores
        }
        ranking = custom_utility_score(score_lists)

        # Format response
        ranked_candidates = format_candidates(
            candidates, adjusted_toxicity_scores, similarity_scores, fluency_scores, ranking
        )
        best_toxicity = ranked_candidates[0].toxicity if ranked_candidates else 1.0
        toxicity_improvement = original_adjusted_toxicity - best_toxicity

        # If no significant improvement and original toxicity is high, return fallback
        if toxicity_improvement < 0.1 and original_adjusted_toxicity >= 0.7:
            logger.warning(
                f"⚠️ Insufficient toxicity improvement ({toxicity_improvement:.3f}), returning fallback message")
            processing_time = time.time() - start_time
            return create_fallback_response(
                request.text,
                original_adjusted_toxicity,
                processing_time,
                request.mode
            )

        # Calculate metadata with ADJUSTED scores for consistency
        processing_time = time.time() - start_time

        metadata = {
            "processing_time_seconds": round(processing_time, 3),
            "original_toxicity_raw": round(original_raw_toxicity.get('toxicity', 0.0), 3),
            "original_toxicity": round(original_adjusted_toxicity, 3),
            "best_candidate_toxicity": round(best_toxicity, 3),
            "toxicity_reduction": round(max(0.0, original_adjusted_toxicity - best_toxicity), 3),
            "reasoning_rules_applied": len([r for r in adjusted_results if r.get('explanations')]),
            "generation_mode": request.mode,
            "candidates_generated": len(candidates),
            "candidates_returned": len(ranked_candidates),
            "reasoning_applied_to_original": True,
            "fallback_used": False  # Normal paraphrasing was successful
        }

        return ParaphraseResponse(
            original=request.text,
            candidates=ranked_candidates,
            metadata=metadata,
            system_info=get_system_metadata()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Paraphrasing failed: {e}")
        # Return fallback message even on unexpected errors
        try:
            processing_time = time.time() - start_time
            # Try to get original toxicity, default to high if failed
            original_toxicity = original_adjusted_toxicity if 'original_adjusted_toxicity' in locals() else 0.8
            return create_fallback_response(
                request.text,
                original_toxicity,
                processing_time,
                request.mode
            )
        except:
            # If even fallback fails, raise the original error
            raise HTTPException(
                status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/paraphrase_advanced")
async def paraphrase_advanced(
    text: str,
    num_candidates: int = 5,
    mode: str = "auto",
    enable_reasoning: bool = True,
    custom_weights: Optional[Dict[str, float]] = None,
    return_scores: bool = False
):
    """
    Advanced paraphrasing with detailed control and scoring information.
    """
    try:
        # Generate with custom parameters
        candidates = generate_paraphrases(text, num_candidates, mode=mode)

        if not candidates:
            raise HTTPException(
                status_code=422, detail="No valid candidates generated")

        # Score candidates
        raw_toxicity_scores = score_toxicity(candidates)

        if enable_reasoning:
            adjusted_results = [apply_reasoning(
                raw_scores) for raw_scores in raw_toxicity_scores]
            toxicity_scores = extract_toxicity_scores(adjusted_results)
            reasoning_explanations = [
                r.get('explanations', []) for r in adjusted_results]
        else:
            toxicity_scores = [r.get('toxicity', 0.0)
                               for r in raw_toxicity_scores]
            reasoning_explanations = [[] for _ in candidates]

        similarity_scores = taunt_equivalence_score(text, candidates)
        fluency_scores = score_fluency(candidates)

        # Custom ranking if weights provided
        if custom_weights:
            score_lists = {
                "toxicity": toxicity_scores,
                "similarity": similarity_scores,
                "fluency": fluency_scores
            }
            ranking = custom_utility_score(score_lists, custom_weights)
        else:
            score_lists = {
                "toxicity": toxicity_scores,
                "similarity": similarity_scores,
                "fluency": fluency_scores
            }
            ranking = custom_utility_score(score_lists)

        # Build response
        response = {
            "original": text,
            "candidates": [
                {
                    "text": candidates[idx],
                    "rank": rank + 1,
                    "scores": {
                        "toxicity": round(toxicity_scores[idx], 3),
                        "similarity": round(similarity_scores[idx], 3),
                        "fluency": round(fluency_scores[idx], 3)
                    }
                }
                for rank, idx in enumerate(ranking)
            ],
            "system_info": get_system_metadata()
        }

        if return_scores:
            response["detailed_scores"] = {
                "raw_toxicity_scores": raw_toxicity_scores,
                "reasoning_explanations": reasoning_explanations,
                "similarity_scores": similarity_scores,
                "fluency_scores": fluency_scores
            }

        return response

    except Exception as e:
        logger.error(f"Advanced paraphrasing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: ParaphraseRequest):
    """
    Complete text analysis: Classification + Reasoning (without paraphrasing).
    Returns toxicity analysis with rule-based adjustments.
    """
    start_time = time.time()
    validate_request(request)

    try:
        logger.info(f"Analyzing text: '{request.text[:50]}...'")

        # Step 1: Get raw toxicity classification
        raw_toxicity_scores = score_toxicity([request.text])
        if not raw_toxicity_scores:
            raise HTTPException(
                status_code=422,
                detail="Unable to classify text toxicity"
            )

        raw_labels = raw_toxicity_scores[0]

        # Step 2: Apply reasoning rules
        reasoning_result = apply_reasoning(raw_labels)
        adjusted_labels = reasoning_result.get('adjusted_labels', raw_labels)
        explanations = reasoning_result.get('explanations', [])

        # Calculate improvements/changes
        toxicity_change = raw_labels.get(
            'toxicity', 0.0) - adjusted_labels.get('toxicity', 0.0)

        processing_time = time.time() - start_time

        analysis_result = {
            "raw_classification": {
                label: round(score, 3) for label, score in raw_labels.items()
            },
            "adjusted_classification": {
                label: round(score, 3) for label, score in adjusted_labels.items()
            },
            "reasoning_explanations": explanations,
            "summary": {
                "is_toxic": adjusted_labels.get('toxicity', 0.0) > 0.5,
                "toxicity_level": "high" if adjusted_labels.get('toxicity', 0.0) > 0.7
                else "medium" if adjusted_labels.get('toxicity', 0.0) > 0.3
                else "low",
                "main_issues": [
                    label for label, score in adjusted_labels.items()
                    if score > 0.5 and label != 'toxicity'
                ],
                "reasoning_applied": len(explanations) > 0,
                "toxicity_adjustment": round(toxicity_change, 3)
            }
        }

        metadata = {
            "processing_time_seconds": round(processing_time, 3),
            "rules_applied": len(explanations),
            "analysis_mode": "classification_with_reasoning"
        }

        logger.info(f"✅ Analysis completed in {processing_time:.2f}s")

        return AnalysisResponse(
            text=request.text,
            analysis=analysis_result,
            metadata=metadata,
            system_info=get_system_metadata()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )


# System Information Endpoints


@app.get("/health")
async def health_check():
    """Universal health check with system-specific information"""
    try:
        memory_usage = get_model_memory_usage()
        model_info = get_model_info()

        health_status = {
            "status": "healthy",
            "service": "Universal AI Paraphraser",
            "system": {
                "os": f"{system_config.os_type} {system_config.architecture}",
                "device": system_config.device.upper(),
                "device_name": system_config.device_name,
                "cpu_cores": system_config.cpu_cores,
                "total_memory_gb": round(system_config.total_memory_gb, 1),
                "available_memory_gb": round(system_config.available_memory_gb, 1)
            },
            "model": {
                "loaded": model_info["model_loaded"],
                "device": model_info.get("model_device", "unknown"),
                "memory_usage_mb": round(memory_usage.get("cpu_memory_mb", 0), 1)
            },
            "api_config": {
                "max_candidates": api_config["max_candidates"],
                "max_batch_size": api_config["max_batch_size"],
                "timeout": api_config["request_timeout"]
            }
        }

        # Add GPU-specific info if available
        if system_config.device == "cuda" and "gpu_memory_allocated_mb" in memory_usage:
            health_status["model"]["gpu_memory_allocated_mb"] = round(
                memory_usage["gpu_memory_allocated_mb"], 1)
            health_status["model"]["gpu_memory_total_mb"] = round(
                memory_usage["gpu_memory_total_mb"], 1)

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "system": get_system_metadata()
        }


@app.get("/system_info", response_model=SystemInfoResponse)
async def get_system_info():
    """Comprehensive system information and capabilities"""
    model_info = get_model_info()
    memory_usage = get_model_memory_usage()

    # Performance estimation based on system
    if system_config.device == "cuda" and system_config.gpu_memory_gb >= 12:
        performance_tier = "High Performance"
        estimated_speed = "Very Fast (0.5-2s per request)"
    elif system_config.device in ["cuda", "mps"]:
        performance_tier = "Medium Performance"
        estimated_speed = "Fast (1-3s per request)"
    else:
        performance_tier = "Standard Performance"
        estimated_speed = "Moderate (2-6s per request)"

    performance_stats = {
        "tier": performance_tier,
        "estimated_speed": estimated_speed,
        "optimal_batch_size": system_config.optimal_batch_size,
        "optimal_workers": system_config.optimal_workers,
        "recommended_concurrent_users": min(system_config.optimal_workers * 2, 10),
        "memory_usage": memory_usage
    }

    return SystemInfoResponse(
        system_config={
            "os_type": system_config.os_type,
            "architecture": system_config.architecture,
            "device": system_config.device,
            "device_name": system_config.device_name,
            "cpu_cores": system_config.cpu_cores,
            "total_memory_gb": system_config.total_memory_gb,
            "available_memory_gb": system_config.available_memory_gb,
            "gpu_memory_gb": system_config.gpu_memory_gb,
            "cache_dir": system_config.cache_dir
        },
        model_info=model_info,
        api_config=api_config,
        performance_stats=performance_stats
    )


@app.get("/capabilities")
async def get_capabilities():
    """Get system-specific capabilities and limits"""
    return {
        "supported_modes": ["auto", "efficient", "quality", "universal"],
        "max_candidates": api_config["max_candidates"],
        "max_batch_size": api_config["max_batch_size"],
        "max_text_length": 5000 if system_config.device == "cpu" else 10000,
        "supported_devices": [system_config.device],
        "auto_optimization": True,
        "concurrent_requests": True,
        "reasoning_engine": True,
        "similarity_scoring": True,
        "fluency_scoring": True,
        "adaptive_caching": True,
        "system_optimization": {
            "device_specific": True,
            "memory_adaptive": True,
            "threading_optimized": True,
            "auto_cleanup": True
        }
    }

# Utility Endpoints


@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Universal AI Paraphraser - Auto-adaptive for any system",
        "system": get_system_metadata(),
        "docs": "/docs",
        "health": "/health",
        "system_info": "/system_info"
    }


@app.get("/docs_info")
async def docs_info():
    """Information about available endpoints"""
    return {
        "endpoints": {
            "/paraphrase": "Main paraphrasing endpoint",
            "/paraphrase_advanced": "Advanced paraphrasing with detailed controls",
            "/health": "System health check",
            "/system_info": "Comprehensive system information",
            "/capabilities": "System capabilities and limits",
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation"
        },
        "modes": {
            "auto": "Automatically selects best mode for your system",
            "efficient": "Memory-efficient mode for resource-constrained systems",
            "quality": "High-quality mode for powerful systems",
            "universal": "Balanced mode that works on any system"
        },
        "system_optimization": f"Optimized for {system_config.device.upper()} on {system_config.os_type.title()}"
    }

# Startup event


@app.on_event("startup")
async def startup_event():
    """Startup event with system detection and optimization"""
    logger.info("🚀 Starting Universal AI Paraphraser...")
    log_system_info()

    # Verify model is loaded
    model_info = get_model_info()
    if not model_info["model_loaded"]:
        logger.warning(
            "⚠️ Model not loaded during startup, will load on first request")
    else:
        logger.info("✅ Model loaded and ready")

    logger.info(
        f"🌐 Service optimized for {system_config.device.upper()} on {system_config.os_type.title()}")
    logger.info(
        f"📊 Performance tier: {api_config.get('performance_tier', 'Standard')}")
    logger.info("🎯 Universal AI Paraphraser ready to serve!")

# Shutdown event


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down Universal AI Paraphraser...")

    try:
        from AI.paraphraser.shared_model import cleanup_model
        cleanup_model()
        logger.info("✅ Model cleanup completed")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

    logger.info("👋 Universal AI Paraphraser stopped")

if __name__ == "__main__":
    import uvicorn

    # Auto-configure uvicorn based on system
    uvicorn_config = {
        "app": "AI.paraphraser.service:app",
        "host": "0.0.0.0",
        "port": 8000,
        "workers": api_config["workers"],
        "timeout_keep_alive": api_config["timeout_keep_alive"],
        "log_level": "info"
    }

    # System-specific uvicorn optimizations
    if system_config.device == "cuda":
        uvicorn_config["worker_class"] = "uvicorn.workers.UvicornWorker"
    elif system_config.os_type == "windows":
        uvicorn_config["loop"] = "asyncio"

    logger.info(f"🚀 Starting server with configuration: {uvicorn_config}")
    uvicorn.run(**uvicorn_config)
