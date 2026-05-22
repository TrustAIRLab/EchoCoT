"""Token counting + scoring metrics for GEPA trajectory optimization.

Scoring metric (drives optimization):
  token_count_ratio = final_output_tokens / target_token_count
  distance_to_1     = abs(token_count_ratio - 1)

Diagnostic metrics (logged, NOT shown to optimizer LLM):
  token_overlap_pct = |recon_set & gt_set| / |gt_set|
  jaccard           = |recon_set & gt_set| / |recon_set | gt_set|
"""

import re

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _tokenize_words(text: str) -> list:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _metric_token_ids(text: str) -> list:
    if not text:
        return []
    if _ENC is not None:
        return _ENC.encode(text)
    return _tokenize_words(text)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(_tokenize_words(text))


def token_count_ratio(final_output_tokens: int, target_token_count: int) -> float:
    if target_token_count <= 0:
        return 0.0
    return final_output_tokens / target_token_count


def distance_to_1(ratio: float) -> float:
    return abs(ratio - 1.0)


def token_overlap_pct(reconstructed: str, ground_truth: str) -> float:
    gt = set(_metric_token_ids(ground_truth))
    if not gt:
        return 0.0
    recon = set(_metric_token_ids(reconstructed))
    return len(recon & gt) / len(gt)


def jaccard(reconstructed: str, ground_truth: str) -> float:
    recon = set(_metric_token_ids(reconstructed))
    gt = set(_metric_token_ids(ground_truth))
    union = recon | gt
    if not union:
        return 0.0
    return len(recon & gt) / len(union)
