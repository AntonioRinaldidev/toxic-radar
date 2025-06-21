# AI/classifier/classifier.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detoxify import Detoxify
import torch
import logging
from typing import List, Dict, Optional
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ToxicRadar Classifier")

# Model initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    model = Detoxify('unbiased', device=device)
    logger.info("Detoxify model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Detoxify model: {e}")
    raise


class TextRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "You're such an idiot!"
            }
        }


class BatchTextRequest(BaseModel):
    texts: List[str]
    max_batch_size: Optional[int] = 32

    class Config:
        json_schema_extra = {
            "example": {
                "texts": ["Hello world", "You're terrible"],
                "max_batch_size": 32
            }
        }


class ClassificationResponse(BaseModel):
    is_success: bool
    message: str
    data: Optional[Dict[str, float]] = None


class BatchClassificationResponse(BaseModel):
    is_success: bool
    message: str
    data: Optional[List[Dict[str, float]]] = None
    processed_count: int
    failed_count: int


def validate_text(text: str) -> str:
    """
    Validate and clean input text.

    Args:
        text: Input text to validate

    Returns:
        Cleaned text

    Raises:
        ValueError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValueError("Text must be a string")

    cleaned = text.strip()

    if not cleaned:
        raise ValueError("Text cannot be empty")

    if len(cleaned) > 10000:  # Reasonable limit
        raise ValueError("Text too long (max 10,000 characters)")

    return cleaned


@lru_cache(maxsize=1000)
def cached_predict_single(text: str) -> Dict[str, float]:
    """
    Cached prediction for single texts to avoid recomputation.

    Args:
        text: Text to classify

    Returns:
        Dictionary of toxicity scores
    """
    try:
        results = model.predict(text)
        return {k: float(v) for k, v in results.items()}
    except Exception as e:
        logger.error(f"Prediction failed for text: {e}")
        raise


@app.post("/classify", response_model=ClassificationResponse)
def classify_text(request: TextRequest):
    """
    Classify a single text for toxicity.

    Args:
        request: Text classification request

    Returns:
        Classification results with toxicity scores
    """
    try:
        text = validate_text(request.text)
        results = cached_predict_single(text)

        return ClassificationResponse(
            is_success=True,
            message="Classification completed successfully.",
            data=results
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return ClassificationResponse(
            is_success=False,
            message=f"Classification failed: {str(e)}",
            data=None
        )


@app.post("/classify_batch", response_model=BatchClassificationResponse)
def classify_batch(request: BatchTextRequest):
    """
    Classify multiple texts for toxicity in batch.

    Args:
        request: Batch text classification request

    Returns:
        Batch classification results
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    if len(request.texts) > request.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.texts)} exceeds maximum {request.max_batch_size}"
        )

    results = []
    failed_count = 0

    try:
        # Validate all texts first
        validated_texts = []
        for i, text in enumerate(request.texts):
            try:
                validated_texts.append(validate_text(text))
            except ValueError as e:
                logger.warning(f"Text {i} validation failed: {e}")
                validated_texts.append(None)
                failed_count += 1

        # Process valid texts in batch
        if validated_texts:
            valid_texts = [t for t in validated_texts if t is not None]
            if valid_texts:
                batch_results = model.predict(valid_texts)

                # Convert batch results to list of dictionaries
                batch_dicts = [
                    {label: batch_results[label][i] for label in batch_results}
                    for i in range(len(valid_texts))
                ]

                # Merge results back with None placeholders
                result_idx = 0
                for validated_text in validated_texts:
                    if validated_text is not None:
                        results.append(
                            {k: float(v) for k, v in batch_dicts[result_idx].items()})
                        result_idx += 1
                    else:
                        results.append(None)

        processed_count = len(request.texts) - failed_count

        return BatchClassificationResponse(
            is_success=True,
            message=f"Batch classification completed. {processed_count} successful, {failed_count} failed.",
            data=results,
            processed_count=processed_count,
            failed_count=failed_count
        )

    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        return BatchClassificationResponse(
            is_success=False,
            message=f"Batch classification failed: {str(e)}",
            data=None,
            processed_count=0,
            failed_count=len(request.texts)
        )


def get_toxicity_scores(texts: List[str]) -> List[Dict[str, float]]:
    """
    Apply Detoxify in batch and return per-sentence toxicity scores.
    Each output is a dict: label -> score.

    Args:
        texts: List of texts to classify

    Returns:
        List of dictionaries with toxicity scores for each text
    """
    if not isinstance(texts, list):
        texts = [texts]

    if not texts:
        return []

    try:
        # Filter out empty texts and track their positions
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)

        if not valid_texts:
            logger.warning("No valid texts to process")
            return [{"toxicity": 0.0} for _ in texts]

        # Process valid texts
        raw_output = model.predict(valid_texts)

        # Convert from {label: [v1, v2, ...]} to [{label: v1}, {label: v2}, ...]
        results = [None] * len(texts)

        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = {
                label: float(raw_output[label][i])
                for label in raw_output
            }

        # Fill None values with default scores
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "toxicity": 0.0,
                    "severe_toxicity": 0.0,
                    "obscene": 0.0,
                    "threat": 0.0,
                    "insult": 0.0,
                    "identity_attack": 0.0
                }

        return results

    except Exception as e:
        logger.error(f"Error in get_toxicity_scores: {e}")
        # Return default scores on error
        return [{"toxicity": 0.0} for _ in texts]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ToxicRadar Classifier",
        "device": device,
        "model": "detoxify-unbiased"
    }


@app.get("/info")
def get_info():
    """Get service information."""
    return {
        "title": "ToxicRadar Classifier",
        "description": "Toxicity classification service using Detoxify",
        "version": "1.0.0",
        "model": "detoxify-unbiased",
        "device": device,
        "endpoints": [
            "/classify - POST: Classify single text",
            "/classify_batch - POST: Classify multiple texts",
            "/health - GET: Service status",
            "/info - GET: Service information"
        ],
        "toxicity_labels": [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack"
        ]
    }
