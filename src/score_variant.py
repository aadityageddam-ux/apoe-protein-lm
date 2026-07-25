"""Zero-shot masked-marginal scoring of protein variants with ESM-2.

Runs against the free Hugging Face Inference API `fill-mask` endpoint — no local
model weights, no GPU. Generic: takes any sequence and any set of 1-based
positions, not specific to APOE.

Method reference: Lin Z, Akin H, Rao R, et al. Science. 2023;379(6637):1123-1130.
PMID: 36927031 (ESM-2).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import requests

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"
HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models/{model}"

# ESM-2's vocabulary is 33 tokens; requesting all of them gives the full
# softmax distribution at the masked position rather than a truncated top-k.
ESM2_VOCAB_SIZE = 33

MASK_TOKEN = "<mask>"
DEFAULT_TIMEOUT = 120
MAX_RETRIES = 5


class InferenceAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class PositionScore:
    position: int
    residue: str
    log_prob: float
    prob: float


def get_api_token(token: str | None = None) -> str:
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise InferenceAPIError(
            "No Hugging Face token found. Set HF_TOKEN in the environment "
            "(a free token from https://huggingface.co/settings/tokens is sufficient)."
        )
    return token


def _spaced(sequence: str) -> list[str]:
    """ESM's tokenizer splits on whitespace, so residues are passed space-separated."""
    return list(sequence)


def masked_distribution(
    sequence: str,
    position: int,
    model: str = DEFAULT_MODEL,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, float]:
    """Return the model's full probability distribution over residues at `position`.

    `position` is 1-based. The residue at that position is replaced by the mask
    token and the model predicts it from the surrounding sequence context.
    """
    if not 1 <= position <= len(sequence):
        raise IndexError(f"Position {position} outside sequence of length {len(sequence)}")

    residues = _spaced(sequence)
    residues[position - 1] = MASK_TOKEN
    payload = {
        "inputs": " ".join(residues),
        "parameters": {"top_k": ESM2_VOCAB_SIZE},
        "options": {"wait_for_model": True},
    }

    headers = {
        "Authorization": f"Bearer {get_api_token(token)}",
        "Content-Type": "application/json",
    }
    url = HF_ROUTER_URL.format(model=model)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            predictions = response.json()
            if not isinstance(predictions, list) or not predictions:
                raise InferenceAPIError(f"Unexpected response payload: {predictions!r}")
            return {p["token_str"]: float(p["score"]) for p in predictions}

        # 503 = model cold-starting; 429 = free-tier rate limit. Both are retryable.
        if response.status_code in (429, 503):
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            time.sleep(5 * (attempt + 1))
            continue

        raise InferenceAPIError(f"HTTP {response.status_code}: {response.text[:500]}")

    raise InferenceAPIError(f"Inference API failed after {MAX_RETRIES} attempts: {last_error}")


def score_positions(
    sequence: str,
    positions: list[int],
    model: str = DEFAULT_MODEL,
    token: str | None = None,
) -> list[PositionScore]:
    """Masked-marginal score of the residues actually present at `positions`.

    For each position: mask it, ask the model for its distribution given the rest
    of this sequence, and record the log-probability it assigns to the residue
    that is really there. Higher (less negative) means the model finds that
    residue more expected in that context.
    """
    scores = []
    for position in positions:
        distribution = masked_distribution(sequence, position, model=model, token=token)
        residue = sequence[position - 1]
        if residue not in distribution:
            raise InferenceAPIError(
                f"Residue {residue!r} absent from returned distribution at position {position}"
            )
        prob = distribution[residue]
        scores.append(
            PositionScore(
                position=position,
                residue=residue,
                log_prob=math.log(prob),
                prob=prob,
            )
        )
    return scores


def total_score(scores: list[PositionScore]) -> float:
    """Sum of per-position log-probabilities."""
    return sum(s.log_prob for s in scores)


def substitution_llr(
    reference_sequence: str,
    substitutions: dict[int, str],
    model: str = DEFAULT_MODEL,
    token: str | None = None,
    cache: dict[int, dict[str, float]] | None = None,
) -> float:
    """Wild-type-context log-likelihood ratio, summed over substitutions.

    For each 1-based position -> mutant residue, computes
    log p(mutant | masked reference context) - log p(wild-type | same context).
    This is the standard ESM zero-shot mutation-effect score (masked marginals
    evaluated in the wild-type background). Positive favours the mutant.

    `cache` maps position -> distribution so repeated calls reuse API responses.
    """
    cache = cache if cache is not None else {}
    llr = 0.0
    for position, mutant in substitutions.items():
        if position not in cache:
            cache[position] = masked_distribution(
                reference_sequence, position, model=model, token=token
            )
        distribution = cache[position]
        wild_type = reference_sequence[position - 1]
        llr += math.log(distribution[mutant]) - math.log(distribution[wild_type])
    return llr
