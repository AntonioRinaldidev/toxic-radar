import re
from collections import OrderedDict
from typing import List
from transformers import pipeline
from AI.paraphraser.shared_model import (
    polish_model, polish_tokenizer,

)
from AI.paraphraser.scorer import score_toxicity,  taunt_equivalence_score

# 1) Normalize known slurs so our similarity reference is neutral
SLUR_MAP = {
    r"\bretard(?:ed)?\b": "person with a different perspective",
    r"\bfuck(?:ing)?\b": "",
    r"\bdickhead\b": "unpleasant person",
    r"\basshole\b": "person being disrespectful",
}


def lexical_clean(text: str) -> str:
    cleaned = text
    for pat, sub in SLUR_MAP.items():
        cleaned = re.sub(pat, sub, cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# Polish model: we’ll use sampling so it can rewrite the entire insult
polish_pipe = pipeline(
    "text2text-generation",
    model=polish_model,
    tokenizer=polish_tokenizer,
    # no beams here—sampling gives more varied full-sentence rewrites
)


def build_polish_prompt(text: str) -> str:
    return (
        f"Instruction: Rewrite the following toxic sentence to be polite and respectful.\n"
        f"Input: {text.strip()}\nPolite:"
    )


def generate_paraphrases(
    text: str,
    max_length=96,
    num_return_sequences=3,  # Quante risposte vuoi per ogni input
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=0.8,
    toxicity_threshold=0.5
) -> List[str]:

    # STEP 1: Build prompt & generate N rewrites via sampling
    prompt = build_polish_prompt(text)
    raw_outs = polish_pipe(
        prompt,
        do_sample=do_sample,         # usa greedy decoding o beam search
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        repetition_penalty=1.8,
    )

    # STEP 2: Clean quotes & collect unique candidates
    polished = []
    seen = set()
    for o in raw_outs:
        s = o["generated_text"].strip()
        s = s.strip('"“”\'’')
        if s and s not in seen:
            seen.add(s)
            polished.append(s)
    print(f"[DEBUG] polished candidates: {polished}")

    # STEP 3: Score toxicity and semantic similarity to the *cleaned* reference
    # We compare to lexical_clean(text) so that mild rewrites (e.g. "You are behind.")
    # don’t get unfairly penalized for being too different from the raw slur.
    ref = lexical_clean(text)
    tox_scores = score_toxicity(polished)
    sim_scores = taunt_equivalence_score(ref, polished)
    for i, (p, tox, sim) in enumerate(zip(polished, tox_scores, sim_scores)):
        print(
            f"[DEBUG] Candidate {i+1}: '{p}' | Toxicity: {tox:.3f} | Similarity: {sim:.3f}")

    # STEP 4: Hard-filter out anything still over the toxicity threshold
    survivors = [
        (p, tox, sim)
        for p, tox, sim in zip(polished, tox_scores, sim_scores)
        if tox <= toxicity_threshold
    ]

    # STEP 5: Rank survivors by similarity (so we preserve meaning) and take top N
    survivors.sort(key=lambda x: x[2], reverse=True)
    final = [p for p, _, _ in survivors][:num_return_sequences]

    # STEP 6: Dedupe (just in case) and return
    return list(OrderedDict.fromkeys(final))
