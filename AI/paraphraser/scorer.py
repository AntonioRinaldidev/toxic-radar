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


def score_identity_hate(candidates: List[str]) -> List[float]:
    """
    Returns identity_hate score so you can filter out
    protected-class insults if desired.

    Args:
        candidates: List of candidate texts

    Returns:
        List of identity hate scores
    """
    if not candidates:
        return []
    raw = get_toxicity_scores(candidates)
    return [d.get("identity_hate", 0.0) for d in raw]


def score_severe_toxicity(candidates: List[str]) -> List[float]:
    """
    Returns severe_toxicity for more aggressive content.

    Args:
        candidates: List of candidate texts

    Returns:
        List of severe toxicity scores
    """
    if not candidates:
        return []
    raw = get_toxicity_scores(candidates)
    return [d.get("severe_toxicity", 0.0) for d in raw]


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


def rank_candidates(
    candidates: List[str]
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Runs all scorers, computes a combined utility score, and
    returns candidates sorted by that score descending.

    Args:
        candidates: List of candidate texts to rank

    Returns:
        List of tuples: (candidate, {toxicity, similarity, fluency, utility})
    """
    # Gather raw data
    toxicity_scores = score_toxicity(candidates)
    similarity_scores = taunt_equivalence_score(
        candidates[0] if candidates else "", candidates)
    fluency_scores = score_fluency(candidates)

    # Extract main toxicity scores for utility calculation
    main_toxicity_scores = [tox.get('toxicity', 0.0)
                            for tox in toxicity_scores]

    # Build score dict for voting utility
    score_lists = {
        "toxicity": main_toxicity_scores,
        "similarity": similarity_scores,
        "fluency": fluency_scores
    }

    # custom_utility_score returns ranked indices
    ranked_indices = custom_utility_score(score_lists)

    results = []
    for idx in ranked_indices:
        utility_score = (1 - main_toxicity_scores[idx]) * 0.5 + \
            similarity_scores[idx] * 0.3 + fluency_scores[idx] * 0.2
        results.append((
            candidates[idx],
            {
                "toxicity": main_toxicity_scores[idx],
                "similarity": similarity_scores[idx],
                "fluency": fluency_scores[idx],
                "utility": utility_score
            }
        ))
    return results
