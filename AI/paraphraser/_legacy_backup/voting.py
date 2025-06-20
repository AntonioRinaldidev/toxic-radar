# AI/paraphraser/voting.py

from typing import List, Dict


def custom_utility_score(score_lists: Dict[str, List[float]]) -> List[int]:
    """
    Custom utility score: (1 - toxicity) * 0.5 + similarity * 0.3 + fluency * 0.2
    Returns ranked indices.
    """
    num_candidates = len(next(iter(score_lists.values())))
    utilities = []

    for i in range(num_candidates):
        tox = score_lists.get("toxicity", [1.0] * num_candidates)[i]
        sim = score_lists.get("similarity", [0.0] * num_candidates)[i]
        flu = score_lists.get("fluency", [0.0] * num_candidates)[i]

        score = (1 - tox) * 0.5 + sim * 0.3 + flu * 0.2
        utilities.append((i, score))

    return [idx for idx, _ in sorted(utilities, key=lambda x: x[1], reverse=True)]
