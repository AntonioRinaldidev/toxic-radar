# AI/paraphraser/voting.py

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def custom_utility_score(
    score_lists: Dict[str, List[float]],
    weights: Optional[Dict[str, float]] = None
) -> List[int]:
    """
    Custom utility score: (1 - toxicity) * weight_tox + similarity * weight_sim + fluency * weight_flu
    Returns ranked indices ordered by utility score (best first).

    Args:
        score_lists: Dictionary with keys 'toxicity', 'similarity', 'fluency' 
                    and values as lists of scores
        weights: Optional custom weights for each component.
                Default: {'toxicity': 0.5, 'similarity': 0.3, 'fluency': 0.2}

    Returns:
        List of indices ordered by utility score (descending)
    """
    # Default weights - toxicity reduction is most important
    if weights is None:
        weights = {
            'toxicity': 0.5,    # Higher weight for toxicity reduction
            'similarity': 0.3,   # Preserve meaning
            'fluency': 0.2      # Language quality
        }

    # Validate weights sum to 1.0
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(
            f"Weights sum to {weight_sum:.3f}, not 1.0. Normalizing...")
        weights = {k: v/weight_sum for k, v in weights.items()}

    # Get number of candidates
    num_candidates = len(next(iter(score_lists.values())))

    # Validate all score lists have same length
    for metric, scores in score_lists.items():
        if len(scores) != num_candidates:
            raise ValueError(
                f"Score list '{metric}' has {len(scores)} items, expected {num_candidates}")

    utilities = []

    for i in range(num_candidates):
        # Get scores for this candidate (with defaults if missing)
        toxicity = score_lists.get("toxicity", [1.0] * num_candidates)[i]
        similarity = score_lists.get("similarity", [0.0] * num_candidates)[i]
        fluency = score_lists.get("fluency", [0.0] * num_candidates)[i]

        # Validate score ranges
        toxicity = max(0.0, min(1.0, toxicity))
        similarity = max(0.0, min(1.0, similarity))
        # Fluency can be negative (log prob)
        fluency = max(-10.0, min(10.0, fluency))

        # Calculate utility score
        # Note: (1 - toxicity) because lower toxicity is better
        score = (
            (1 - toxicity) * weights['toxicity'] +
            similarity * weights['similarity'] +
            fluency * weights['fluency']
        )

        utilities.append((i, score))

    # Sort by utility score (descending - higher is better)
    utilities.sort(key=lambda x: x[1], reverse=True)

    # Return just the indices
    ranked_indices = [idx for idx, _ in utilities]

    logger.debug(
        f"Utility scores: {[(i, round(score, 3)) for i, score in utilities]}")

    return ranked_indices




def adaptive_scoring(
    score_lists: Dict[str, List[float]],
    original_toxicity: float = 0.5
) -> List[int]:
    """
    Adaptive scoring that adjusts weights based on original text toxicity.

    Args:
        score_lists: Dictionary of score lists
        original_toxicity: Toxicity score of original text

    Returns:
        List of indices ranked adaptively
    """
    # Adjust weights based on original toxicity level
    if original_toxicity > 0.8:
        # Very toxic: prioritize toxicity reduction heavily
        weights = {'toxicity': 0.7, 'similarity': 0.2, 'fluency': 0.1}
    elif original_toxicity > 0.5:
        # Moderately toxic: balanced approach
        weights = {'toxicity': 0.5, 'similarity': 0.3, 'fluency': 0.2}
    else:
        # Mildly toxic: preserve meaning and fluency more
        weights = {'toxicity': 0.3, 'similarity': 0.4, 'fluency': 0.3}

    logger.info(
        f"Using adaptive weights for toxicity {original_toxicity:.2f}: {weights}")

    return custom_utility_score(score_lists, weights)
