"""
Judge agent — uses Gemma 4:31b (Ollama) to score MT outputs against a rubric.

The judge receives:
  • The source English sentence
  • A reference (gold) translation produced by the Challenger
  • The MT model's candidate translation
  • The rubric (list of weighted criteria)

It returns per-criterion scores in [0, 1] and a weighted aggregate score.
Scores are clamped and fail-closed on out-of-range values (same hardening
strategy as the Lauorie/Autodata repo's judge normalisation).
"""

import logging
from dataclasses import dataclass
from statistics import mean
from typing import Optional

from .config import LANG_NAMES, OllamaConfig
from .utils import call_ollama, compute_weighted_score, extract_json_object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert machine translation quality evaluator.

You will receive:
  • A source English sentence
  • A REFERENCE translation (what a perfect translation should say)
  • A CANDIDATE translation (produced by an MT model — to be scored)
  • A RUBRIC with N evaluation criteria

Your task: for EACH criterion assign a score in [0.0, 1.0].

Scoring guide:
  1.0  — fully satisfies the criterion
  0.7  — mostly satisfies, minor issue
  0.5  — partially satisfies (meaning preserved but awkward / partially wrong)
  0.3  — barely satisfies (key error but some correct elements)
  0.0  — does not satisfy at all (wrong meaning, missing, or ungrammatical)

Be STRICT and CALIBRATED:
  • A wrong idiom rendering (literal translation) → 0.0–0.2 on that criterion.
  • Correct meaning but unnatural phrasing → 0.4–0.6.
  • Near-perfect output → 0.8–1.0.

Output ONLY this JSON object (no prose before or after):
{"per_criterion": [<float>, ...], "overall_comment": "<one sentence>"}

"per_criterion" MUST have EXACTLY the same number of elements as the rubric.\
"""


def _build_judge_prompt(
    source: str,
    reference: str,
    candidate: str,
    rubric: list[dict],
    tgt_lang_name: str,
) -> str:
    criteria_block = "\n".join(
        f"  {i + 1}. [weight={r['weight']}] {r['criterion']}"
        for i, r in enumerate(rubric)
    )
    return (
        f"Evaluate this {tgt_lang_name} translation.\n\n"
        f"SOURCE (English)   : {source}\n"
        f"REFERENCE          : {reference}\n"
        f"CANDIDATE (to score): {candidate}\n\n"
        f"RUBRIC ({len(rubric)} criteria):\n{criteria_block}\n\n"
        f"Output per-criterion scores as a JSON array of {len(rubric)} floats:\n"
        f'{{ "per_criterion": [<score_1>, <score_2>, ...], "overall_comment": "<assessment>" }}'
    )


# ---------------------------------------------------------------------------
# Score normalisation (fail-closed, same logic as Lauorie/Autodata)
# ---------------------------------------------------------------------------

_CLAMP_TOLERANCE = 0.25  # accept values in [-0.25, 1.25]; clamp to [0, 1]


def _normalise_scores(raw: list, rubric_len: int) -> list[float]:
    """
    Clamp per-criterion scores to [0, 1] with tolerance, fail-closed on
    grossly out-of-range or non-numeric values.
    """
    scores: list[float] = []
    for s in raw:
        try:
            v = float(s)
        except (ValueError, TypeError):
            scores.append(0.0)  # fail-closed: non-numeric
            continue

        if -_CLAMP_TOLERANCE <= v <= 1.0 + _CLAMP_TOLERANCE:
            scores.append(max(0.0, min(1.0, v)))
        else:
            scores.append(0.0)  # fail-closed: wildly out of range (e.g. 2.5)

    # Align length with rubric
    if len(scores) < rubric_len:
        scores.extend([0.0] * (rubric_len - len(scores)))
    return scores[:rubric_len]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class TranslationScore:
    candidate: str          # the MT output that was scored
    per_criterion: list[float]
    weighted_score: float   # 0–1, weight-normalised aggregate
    overall_comment: str


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

class Judge:
    """Calls Gemma (via Ollama) to score one MT candidate against a rubric."""

    def __init__(self, cfg: OllamaConfig, tgt_lang: str) -> None:
        self.cfg = cfg
        self.tgt_lang_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())

    def score(
        self,
        source: str,
        reference: str,
        candidate: str,
        rubric: list[dict],
    ) -> TranslationScore:
        """
        Score *candidate* against *rubric* and return a TranslationScore.

        On any failure (LLM error, JSON parse error) the method returns a
        zero-score result so the caller always receives a valid object.
        """
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {
                "role": "user",
                "content": _build_judge_prompt(
                    source, reference, candidate, rubric, self.tgt_lang_name
                ),
            },
        ]

        # ── Call LLM ──────────────────────────────────────────────────────
        try:
            raw = call_ollama(
                messages=messages,
                model=self.cfg.model,
                temperature=self.cfg.judge_temperature,
                base_url=self.cfg.base_url,
                timeout_s=self.cfg.timeout_s,
            )
        except RuntimeError as e:
            logger.error(f"Judge LLM call failed: {e}")
            return self._zero_score(candidate, len(rubric), "LLM call failed")

        # ── Parse JSON ────────────────────────────────────────────────────
        data = extract_json_object(raw)
        if data is None:
            logger.warning(f"Judge returned non-JSON: {raw[:200]!r}")
            return self._zero_score(candidate, len(rubric), "JSON parse failed")

        # ── Normalise scores ──────────────────────────────────────────────
        raw_scores = data.get("per_criterion", [])
        if not isinstance(raw_scores, list):
            raw_scores = []

        per_criterion = _normalise_scores(raw_scores, len(rubric))
        weights = [r["weight"] for r in rubric]
        weighted = compute_weighted_score(per_criterion, weights)
        comment = str(data.get("overall_comment", "")).strip()[:300]

        logger.debug(
            f"Judge scored {weighted:.3f} | candidate='{candidate[:60]}' | {comment}"
        )
        return TranslationScore(
            candidate=candidate,
            per_criterion=per_criterion,
            weighted_score=weighted,
            overall_comment=comment,
        )

    @staticmethod
    def _zero_score(candidate: str, rubric_len: int, reason: str) -> TranslationScore:
        return TranslationScore(
            candidate=candidate,
            per_criterion=[0.0] * rubric_len,
            weighted_score=0.0,
            overall_comment=reason,
        )

    def generate_feedback(
        self,
        source: str,
        reference: str,
        rubric: list[dict],
        weak_scores: list["TranslationScore"],
        strong_scores: list["TranslationScore"],
        fail_reasons: list[str],
    ) -> Optional[str]:
        """
        Ask the Judge LLM to produce rich, sentence-specific rejection feedback
        for the Challenger.

        Unlike the template-based format_rejection_feedback(), this method
        examines the actual per-criterion breakdowns and the translations
        themselves, then writes targeted advice on WHAT to change and WHY.

        Returns a feedback string, or None on LLM/parse failure (caller should
        fall back to format_rejection_feedback()).
        """
        weak_avg = mean(s.weighted_score for s in weak_scores)
        strong_avg = mean(s.weighted_score for s in strong_scores)
        gap = strong_avg - weak_avg

        # Build per-criterion breakdown for the prompt
        def _breakdown(scores: list["TranslationScore"], label: str) -> str:
            lines = [f"{label} translation: '{scores[0].candidate}'"]
            for i, (crit, sc) in enumerate(
                zip(rubric, scores[0].per_criterion), start=1
            ):
                lines.append(
                    f"  criterion {i} [{crit['weight']}w] '{crit['criterion']}' "
                    f"→ {sc:.2f}"
                )
            lines.append(
                f"  weighted_score={scores[0].weighted_score:.3f}  "
                f"judge_comment='{scores[0].overall_comment}'"
            )
            return "\n".join(lines)

        breakdown = (
            _breakdown(weak_scores, "WEAK model") + "\n\n"
            + _breakdown(strong_scores, "STRONG model")
        )

        conditions_block = "\n".join(f"  ✗ {r}" for r in fail_reasons)

        system = (
            "You are an expert machine translation data curator. "
            "Analyze a FAILED MT discrimination attempt and write precise, "
            "actionable feedback to help the Challenger LLM design a better sentence. "
            "Be specific about WHICH linguistic property to target and HOW."
        )

        user = (
            f"A source sentence was REJECTED because it did not discriminate "
            f"between a weak MT model (Helsinki-NLP MarianMT, 74M params) and a "
            f"strong MT model (Meta NLLB-200, 1.3B params).\n\n"
            f"SOURCE SENTENCE: {source}\n"
            f"REFERENCE      : {reference}\n"
            f"SCORES         : weak_avg={weak_avg:.3f}  strong_avg={strong_avg:.3f}  "
            f"gap={gap:.3f}\n\n"
            f"FAILED CONDITIONS:\n{conditions_block}\n\n"
            f"PER-CRITERION BREAKDOWN:\n{breakdown}\n\n"
            f"Write a feedback message for the Challenger LLM (3–6 sentences) that:\n"
            f"  1. Explains the ROOT CAUSE of the discrimination failure for THIS specific sentence.\n"
            f"  2. Identifies WHICH specific criteria the weak model failed/passed unexpectedly.\n"
            f"  3. Gives a CONCRETE recommendation: what linguistic property, "
            f"construct, or domain to target in the NEXT sentence.\n"
            f"  4. If the weak model scored too HIGH: suggest a HARDER challenge "
            f"(more idiomatic, rarer term, deeper morphology).\n"
            f"  5. If the gap was too SMALL: suggest a challenge that more specifically "
            f"exploits MarianMT's known weaknesses (separable verbs, idioms, "
            f"compound-noun splitting, register switching, rare domain terms).\n\n"
            f"Write ONLY the plain-text feedback message — no JSON, no headers."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        try:
            raw = call_ollama(
                messages=messages,
                model=self.cfg.model,
                temperature=0.4,   # slightly creative but focused
                base_url=self.cfg.base_url,
                timeout_s=self.cfg.timeout_s,
            )
        except RuntimeError as e:
            logger.warning(f"Judge.generate_feedback LLM call failed: {e}")
            return None

        feedback = raw.strip()
        if not feedback:
            return None

        logger.debug(f"Judge feedback ({len(feedback)} chars): {feedback[:120]}")
        return feedback
