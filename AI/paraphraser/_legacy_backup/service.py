# AI/paraphraser/service.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from AI.paraphraser.voting import custom_utility_score
from AI.paraphraser.generator import generate_paraphrases
from AI.paraphraser.scorer import score_toxicity, taunt_equivalence_score, score_fluency

app = FastAPI(title="ToxicRadar Paraphraser")


# ==== Request & Response Schemas ====

class ParaphraseRequest(BaseModel):
    text: str
    num_candidates: int = 10
    # optional: you can also expose threshold and max_length
    toxicity_threshold: float = 0.5
    max_length: int = 120


class ParaphraseCandidate(BaseModel):
    text: str
    toxicity: float
    similarity: float
    fluency: float
    rank: int


class ParaphraseResponse(BaseModel):
    original: str
    candidates: List[ParaphraseCandidate]


# ==== Internal Utilities ====

def format_candidates(
    candidates: List[str],
    toxicity: List[float],
    similarity: List[float],
    fluency: List[float],
    ranking: List[int]
) -> List[ParaphraseCandidate]:
    """
    Build and return a list of ParaphraseCandidate objects,
    ordering them according to the computed ranking.
    """
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


# ==== Endpoint ====

@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase(request: ParaphraseRequest):
    # 1) Generation (detox → polish) with requested parameters
    candidates = generate_paraphrases(
        text=request.text,
        num_return_sequences=request.num_candidates,
        max_length=request.max_length,
        toxicity_threshold=request.toxicity_threshold,
    )

    # 2) If no clean candidates, return 422 Unprocessable Entity
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail="All candidates were too toxic to consider."
        )

    # 3) Score the filtered candidates
    tox_scores = score_toxicity(candidates)
    sim_scores = taunt_equivalence_score(request.text, candidates)
    flu_scores = score_fluency(candidates)

    # 4) Pre-filter: keep only those below the toxicity threshold
    filtered = [
        (c, t, s, f)
        for c, t, s, f in zip(candidates, tox_scores, sim_scores, flu_scores)
        if t <= request.toxicity_threshold
    ]

    # If still no candidates after scoring, return 422
    if not filtered:
        raise HTTPException(
            status_code=422,
            detail="All candidates were too toxic after scoring."
        )

    # 5) Prepare the score lists for ranking
    candidates_f, tox_f, sim_f, flu_f = zip(*filtered)
    score_lists: Dict[str, List[float]] = {
        "toxicity": list(tox_f),
        "similarity": list(sim_f),
        "fluency": list(flu_f)
    }

    # 6) Compute the ranking using the custom utility score function
    ranking = custom_utility_score(score_lists)

    # 7) Build the final ordered output
    ranked_candidates = format_candidates(
        list(candidates_f),
        list(tox_f),
        list(sim_f),
        list(flu_f),
        ranking
    )

    return ParaphraseResponse(
        original=request.text,
        candidates=ranked_candidates
    )
