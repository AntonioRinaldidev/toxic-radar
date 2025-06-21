# AI/paraphraser/scorer.py

from typing import List, Dict, Tuple
from AI.classifier.classifier import get_toxicity_scores
from sentence_transformers import SentenceTransformer, util
from AI.paraphraser._legacy_backup.shared_model import polish_model as fluency_model, polish_tokenizer as fluency_tokenizer
from AI.paraphraser._legacy_backup.voting import custom_utility_score
import torch
import re

# Load Sentence-BERT model
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# SLUR_MAP: same as in generator, to clean original before similarity
SLUR_MAP = {
    r"\bretard(?:ed)?\b": "person with a different perspective",
    r"\bfuck(?:ing)?\b": "",
    r"\bdickhead\b": "unpleasant person",
    r"\basshole\b": "person being disrespectful",
}


def normalize_reference(text: str) -> str:
    """
    Remove slurs and profanity so similarity
    scoring is based on intent, not hateful tokens.
    """
    cleaned = text
    for pat, sub in SLUR_MAP.items():
        cleaned = re.sub(pat, sub, cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def score_toxicity(candidates: List[str]) -> List[float]:
    """
    Returns basic 'toxicity' score in [0,1].
    """
    if not candidates:
        return []
    raw = get_toxicity_scores(candidates)
    return [d.get("toxicity", 0.0) for d in raw]


def score_identity_hate(candidates: List[str]) -> List[float]:
    """
    Returns identity_hate score so you can filter out
    protected-class insults if desired.
    """
    if not candidates:
        return []
    raw = get_toxicity_scores(candidates)
    return [d.get("identity_hate", 0.0) for d in raw]


def score_severe_toxicity(candidates: List[str]) -> List[float]:
    """
    Returns severe_toxicity for more aggressive content.
    """
    if not candidates:
        return []
    raw = get_toxicity_scores(candidates)
    return [d.get("severe_toxicity", 0.0) for d in raw]


def taunt_equivalence_score(original: str, candidates: List[str]) -> List[float]:
    """
    Returns a proxy for 'intent alignment' using semantic similarity,
    but caps the penalty to avoid bias against soft rephrasings.
    """
    ref = normalize_reference(original)
    all_sents = [ref] + candidates
    emb = similarity_model.encode(all_sents, convert_to_tensor=True)
    ref_emb, cand_embs = emb[0], emb[1:]
    sims = util.cos_sim(ref_emb, cand_embs)[0].tolist()

    # Soft floor: anything above 0.4 is considered "acceptable intent match"
    adjusted = [min(1.0, max(0.4, float(s))) for s in sims]
    return adjusted


def score_fluency(candidates: List[str]) -> List[float]:
    """
    Average log-probability per token (higher = more fluent).
    """
    scores: List[float] = []
    for sent in candidates:
        inputs = fluency_tokenizer(
            sent, return_tensors="pt").to(fluency_model.device)
        with torch.no_grad():
            out = fluency_model(**inputs, labels=inputs["input_ids"])
            loss = out.loss.item()
        length = inputs["input_ids"].shape[1]
        scores.append(-loss / length)
    return scores


def rank_candidates(
    candidates: List[str]
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Runs all scorers, computes a combined utility score, and
    returns candidates sorted by that score descending.

    Returns:
      [ (candidate, {toxicity, similarity, fluency, utility}), ... ]
    """
    # Gather raw data
    tox = score_toxicity(candidates)
    sim = taunt_equivalence_score(
        candidates[0] if candidates else "", candidates)
    flu = score_fluency(candidates)
    # Build score dict for voting util
    score_lists = {
        "toxicity": tox,
        "similarity": sim,
        "fluency": flu
    }
    # custom_utility_score returns ranked indices
    ranked_idxs = custom_utility_score(score_lists)

    results = []
    for idx in ranked_idxs:
        util_score = (1 - tox[idx]) * 0.5 + sim[idx] * 0.3 + flu[idx] * 0.2
        results.append((
            candidates[idx],
            {
                "toxicity": tox[idx],
                "similarity": sim[idx],
                "fluency": flu[idx],
                "utility": util_score
            }
        ))
    return results
