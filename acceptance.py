"""
Acceptance criteria — the discriminator gate adapted from AutoData for MT.

A source sentence is accepted only when ALL four conditions hold:

  ┌──────────────────────────────┬──────────────────────────────────────────┐
  │ weak_avg  ≤ weak_max         │ weak MT model generally fails            │
  │ weak_avg  ≥ weak_min_floor   │ sentence is at least translatable        │
  │ strong_avg ≥ strong_min      │ strong MT model succeeds                 │
  │ gap        ≥ min_gap         │ meaningful discrimination between models  │
  └──────────────────────────────┴──────────────────────────────────────────┘

Thresholds (from PoCConfig.acceptance):
  weak_max       = 0.55   (mirrors AutoData's 0.65 for CS-QA; tighter for MT)
  weak_min_floor = 0.05   (sentence must be at least minimally translatable)
  strong_min     = 0.68
  min_gap        = 0.20
"""

from dataclasses import dataclass
from statistics import mean

from .config import AcceptanceCriteria
from .judge import TranslationScore


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceResult:
    accepted: bool
    weak_avg: float
    strong_avg: float
    gap: float
    fail_reasons: list[str]  # empty when accepted


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------

def check_acceptance(
    weak_scores: list[TranslationScore],
    strong_scores: list[TranslationScore],
    criteria: AcceptanceCriteria,
) -> AcceptanceResult:
    """
    Compute aggregate scores and apply the four-condition gate.

    Returns an AcceptanceResult with `accepted=True` only when all pass.
    """
    weak_avg = mean(s.weighted_score for s in weak_scores)
    strong_avg = mean(s.weighted_score for s in strong_scores)
    gap = strong_avg - weak_avg

    reasons: list[str] = []

    if weak_avg > criteria.weak_max:
        reasons.append(
            f"weak_avg={weak_avg:.3f} > {criteria.weak_max} "
            "(sentence too easy — weak model can also translate it)"
        )
    if weak_avg < criteria.weak_min_floor:
        reasons.append(
            f"weak_avg={weak_avg:.3f} < {criteria.weak_min_floor} "
            "(sentence untranslatable — even weak model produces zero output)"
        )
    if strong_avg < criteria.strong_min:
        reasons.append(
            f"strong_avg={strong_avg:.3f} < {criteria.strong_min} "
            "(even the strong model fails — sentence too difficult/obscure)"
        )
    if gap < criteria.min_gap:
        reasons.append(
            f"gap={gap:.3f} < {criteria.min_gap} "
            "(insufficient discrimination: both models perform similarly)"
        )

    return AcceptanceResult(
        accepted=len(reasons) == 0,
        weak_avg=weak_avg,
        strong_avg=strong_avg,
        gap=gap,
        fail_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Feedback formatter
# ---------------------------------------------------------------------------

def format_rejection_feedback(
    result: AcceptanceResult,
    weak_scores: list[TranslationScore],
    strong_scores: list[TranslationScore],
) -> str:
    """
    Build a structured feedback string for the Challenger to improve on.

    Includes:
    - What failed and why
    - Actionable guidance specific to the failure mode
    - Sample translations from both models for context
    """
    lines: list[str] = [
        "REJECTION — all conditions must hold simultaneously:",
        *[f"  ✗ {r}" for r in result.fail_reasons],
        "",
        f"Scores: weak_avg={result.weak_avg:.3f}, "
        f"strong_avg={result.strong_avg:.3f}, gap={result.gap:.3f}",
        "",
        "GUIDANCE:",
    ]

    # Specific actionable advice per failure mode
    if result.weak_avg > 0.55:
        lines.append(
            "  → The sentence was too EASY. Use a stronger challenge: deeper idiom, "
            "more ambiguous lexical choice, rare domain term, or complex morphology."
        )
    if result.strong_avg < 0.68:
        lines.append(
            "  → The sentence was too HARD even for the strong model. "
            "Make it more translatable: the strong model should handle the challenge well."
        )
    if result.gap < 0.20:
        lines.append(
            "  → Gap too small. The challenge must SPECIFICALLY exploit weaknesses of the "
            "small MarianMT model: it struggles with idioms, gender agreement, separable "
            "verbs, compound-noun splitting, and domain vocabulary."
        )

    # Show the actual translations for context
    lines.append("")
    if weak_scores:
        w_t = weak_scores[0].candidate
        w_c = weak_scores[0].overall_comment
        lines.append(f"  Weak model output  : '{w_t}'")
        lines.append(f"  Judge comment      : {w_c}")
    if strong_scores:
        s_t = strong_scores[0].candidate
        s_c = strong_scores[0].overall_comment
        lines.append(f"  Strong model output: '{s_t}'")
        lines.append(f"  Judge comment      : {s_c}")

    return "\n".join(lines)
