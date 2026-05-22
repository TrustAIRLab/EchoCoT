"""
Metrics and trajectory data structures.

Training phase:
  - token_count_ratio computed locally (range tracking)
  - Success is controlled by optimizer STOP decision.

Evaluation phase:
  - token_count_ratio + token_overlap_pct + jaccard (needs ground-truth CoT)
"""

import re
from dataclasses import dataclass, field

from config import RATIO_LOWER_BOUND, RATIO_UPPER_BOUND

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _tokenize_words(text: str) -> list:
    """Split into lowercase word tokens."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def count_tokens(text: str) -> int:
    """Model-agnostic token count (tiktoken fallback to word count)."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return len(_tokenize_words(text))


def _metric_token_ids(text: str) -> list:
    """Token ids for overlap/jaccard metrics, with word fallback."""
    if not text:
        return []
    if _ENC is not None:
        return _ENC.encode(text)
    return _tokenize_words(text)


def compute_token_count_ratio(scratchpad_tokens: int, gt_tokens: int) -> float:
    """token_count_ratio of scratchpad word count to ground-truth reasoning word count."""
    if gt_tokens == 0:
        return 0.0
    return scratchpad_tokens / gt_tokens


def token_count_ratio_passes_threshold(token_count_ratio: float) -> bool:
    """Return True if token_count_ratio is within configured bounds."""
    return RATIO_LOWER_BOUND <= token_count_ratio <= RATIO_UPPER_BOUND


# ---------------------------------------------------------------------------
# Evaluation metrics (jaccard + overlap, needs ground-truth content)
# ---------------------------------------------------------------------------

def compute_eval_metrics(reconstructed: str, ground_truth: str) -> dict:
    """Full metrics for evaluation phase. Needs ground-truth CoT content."""
    recon_tokens = _metric_token_ids(reconstructed)
    gt_tokens = _metric_token_ids(ground_truth)

    if not gt_tokens:
        return {"token_count_ratio": 0.0, "token_overlap_pct": 0.0, "jaccard": 0.0}

    recon_set = set(recon_tokens)
    gt_set = set(gt_tokens)
    intersection = recon_set & gt_set
    union = recon_set | gt_set

    return {
        "token_count_ratio": len(recon_tokens) / len(gt_tokens),
        "token_overlap_pct": len(intersection) / len(gt_set) if gt_set else 0.0,
        "jaccard": len(intersection) / len(union) if union else 0.0,
    }


# ---------------------------------------------------------------------------
# Trajectory data structures
# ---------------------------------------------------------------------------


@dataclass
class InjectionTurn:
    turn_idx: int
    scratchpad_in: str
    scratchpad_in_tokens: int
    token_count_ratio: float = 0.0
    token_overlap_pct: float = 0.0
    jaccard: float = 0.0
    injection_out: str = ""
    stopped: bool = False


@dataclass
class TrajectoryResult:
    turns: list = field(default_factory=list)
    ground_truth_cot: str = ""      # first-turn thinking (kept for eval, never shown to optimizer)
    ground_truth_tokens: int = 0
    final_output: str = ""
    success: bool = False           # optimizer said STOP AND token_count_ratio passed
    success_turn: int = -1          # first turn that succeeded (-1 = never)

    @property
    def final_scratchpad(self) -> str:
        return self.turns[-1].scratchpad_in if self.turns else ""

    @property
    def final_scratchpad_tokens(self) -> int:
        return self.turns[-1].scratchpad_in_tokens if self.turns else 0

    @property
    def final_token_count_ratio(self) -> float:
        return self.turns[-1].token_count_ratio if self.turns else 0.0

    @property
    def num_injection_turns(self) -> int:
        return max(0, len(self.turns) - 1)

    def trajectory_log(self) -> str:
        """Full trajectory with content — shown to optimizer in REFLECT."""
        lines = []
        for t in self.turns:
            tag = "[INITIAL]" if t.turn_idx == 0 else f"[TURN {t.turn_idx}]"
            lines.append(
                f"{tag} token_count_ratio={t.token_count_ratio:.3f}"
            )
            if t.injection_out and not t.stopped:
                lines.append(f"  INJECTION SENT:\n  {t.injection_out}..."
                             f"Received Scores: token_count_ratio={t.token_count_ratio:.3f}")
            if t.stopped:
                lines.append("  -> STOPPED"
                             f"Final Scores: token_count_ratio={t.token_count_ratio:.3f}")
                
        return "\n\n".join(lines)
