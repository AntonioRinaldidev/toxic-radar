# AI/paraphraser/scorer.py

from typing import List, Dict, Tuple
from AI.classifier.classifier import get_toxicity_scores
from sentence_transformers import SentenceTransformer, util
from AI.paraphraser.shared_model import polish_model as fluency_model, polish_tokenizer as fluency_tokenizer
from AI.paraphraser.voting import custom_utility_score
import torch
import re

# Load Sentence-BERT model
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")


def score_toxicity(candidates: List[str]) -> List[Dict[str, float]]:
    """
    Returns a list of dictionaries, where each dictionary contains
    all raw toxicity scores (e.g., 'toxicity', 'severe_toxicity', etc.) for a candidate.

    Args:
        candidates: List of candidate texts to score

    Returns:
        List of dictionaries with toxicity scores
    """
    if not candidates:
        return []
    # Assuming get_toxicity_scores already returns a list of dictionaries
    # like [{"toxicity": 0.8, "insult": 0.7, ...}, {...}]
    raw = get_toxicity_scores(candidates)
    return raw  # Return the full raw scores



def taunt_equivalence_score(original: str, candidates: List[str]) -> List[float]:
    """
    Returns a proxy for 'intent alignment' using semantic similarity,
    but caps the penalty to avoid bias against soft rephrasings.

    Args:
        original: Original text to compare against
        candidates: List of candidate paraphrases

    Returns:
        List of similarity scores (0.4-1.0 range)
    """

    all_sentences = [original] + candidates
    embeddings = similarity_model.encode(all_sentences, convert_to_tensor=True)
    ref_embedding, candidate_embeddings = embeddings[0], embeddings[1:]
    similarities = util.cos_sim(
        ref_embedding, candidate_embeddings)[0].tolist()

    # Soft floor: anything above 0.4 is considered "acceptable intent match"
    adjusted = [min(1.0, max(0.4, float(s))) for s in similarities]
    return adjusted


def score_fluency(candidates: List[str]) -> List[float]:
    """
    Average log-probability per token (higher = more fluent).

    Args:
        candidates: List of candidate texts

    Returns:
        List of fluency scores
    """
    scores: List[float] = []
    for sentence in candidates:
        inputs = fluency_tokenizer(
            sentence, return_tensors="pt").to(fluency_model.device)
        with torch.no_grad():
            output = fluency_model(**inputs, labels=inputs["input_ids"])
            loss = output.loss.item()
        length = inputs["input_ids"].shape[1]
        scores.append(-loss / length)
    return scores


