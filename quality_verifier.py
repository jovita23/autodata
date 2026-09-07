"""
Quality Verifier (QV) — LLM-based structural gate on Challenger output.

Adapted from the Lauorie/Autodata reference implementation's five-axis check
(`pipeline/inner_loop.py::_qv_step`, Kimi-K2.6 with reasoning_effort=none).

For MT, the five axes are:

  A. TRANSLATABLE      — sentence is genuine natural English (not gibberish,
                         not already in the target language, not a fragment)
  B. CHALLENGE_VALID   — the sentence genuinely exemplifies the claimed
                         challenge type (e.g., actually contains an idiom if
                         type='idiom'; actually requires a register choice if
                         type='register_formality')
  C. REFERENCE_QUALITY — the reference translation is correct and idiomatic,
                         not a word-for-word literal rendering
  D. RUBRIC_COHERENCE  — each criterion is (i) a POSITIVE statement, (ii)
                         specific and verifiable from the translation alone,
                         (iii) weight is an integer in [1, 5]
  E. DIFFICULTY_FIT    — the challenge targets a known MarianMT weakness
                         (idioms, morphological agreement, separable verbs,
                         compound-noun splitting, domain vocabulary, register);
                         NOT just generic sentence difficulty

All five must pass. On failure the QV returns structured per-axis feedback
so the Challenger can fix exactly what was wrong.

The QV is designed to be fast: low temperature, short output (JSON only).
"""

import logging
from dataclasses import dataclass

from .config import LANG_NAMES, OllamaConfig
from .utils import call_ollama, extract_json_object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QV_SYSTEM = """\
You are a quality verifier for a machine translation data generation pipeline.

You receive a Challenger's output (a source English sentence, a reference \
translation, and a rubric) and must check it on FIVE axes. Be strict but fair.

For each axis, output:
  "pass": true   — if it clearly passes
  "pass": false  — if it fails (even partially)
  "reason": "<one concise sentence explaining the decision>"

Output ONLY a valid JSON object. No prose before or after.\
"""


def _build_qv_prompt(
    source: str,
    reference: str,
    challenge_type: str,
    challenge_explanation: str,
    rubric: list[dict],
    tgt_lang_name: str,
) -> str:
    rubric_block = "\n".join(
        f"  {i+1}. [weight={r['weight']}] {r['criterion']}"
        for i, r in enumerate(rubric)
    )
    return (
        f"Verify this Challenger output for a {tgt_lang_name} translation task.\n\n"
        f"SOURCE SENTENCE   : {source}\n"
        f"REFERENCE TRANSL. : {reference}\n"
        f"CHALLENGE TYPE    : {challenge_type}\n"
        f"CHALLENGER EXPLAINS: {challenge_explanation}\n\n"
        f"RUBRIC ({len(rubric)} criteria):\n{rubric_block}\n\n"
        f"Check all FIVE axes and output exactly this JSON:\n"
        f"{{\n"
        f'  "A_translatable":     {{"pass": <bool>, "reason": "<str>"}},\n'
        f'  "B_challenge_valid":  {{"pass": <bool>, "reason": "<str>"}},\n'
        f'  "C_reference_quality":{{"pass": <bool>, "reason": "<str>"}},\n'
        f'  "D_rubric_coherence": {{"pass": <bool>, "reason": "<str>"}},\n'
        f'  "E_difficulty_fit":   {{"pass": <bool>, "reason": "<str>"}}\n'
        f"}}\n\n"
        f"Axis definitions:\n"
        f"  A. TRANSLATABLE: The source is genuine, complete, natural English "
        f"(8-40 words). Not gibberish, not already in {tgt_lang_name}, not truncated.\n"
        f"  B. CHALLENGE_VALID: The sentence genuinely exhibits the challenge type "
        f"'{challenge_type}'. The challenge_explanation must match what is actually "
        f"linguistically present in the sentence.\n"
        f"  C. REFERENCE_QUALITY: The reference is a correct, idiomatic {tgt_lang_name} "
        f"translation — NOT word-for-word literal. For idiom/proverb types, the "
        f"target-language equivalent must be used.\n"
        f"  D. RUBRIC_COHERENCE: Every criterion is a POSITIVE statement "
        f"('The translation correctly renders…'). Each is specific and verifiable "
        f"from the translation text alone. Weights are integers in [1, 5]. "
        f"No negative criteria ('does not use…' is a FAIL).\n"
        f"  E. DIFFICULTY_FIT: The challenge specifically targets a known weakness "
        f"of small MarianMT models (idioms, morphological agreement, separable verbs, "
        f"compound-noun splitting, domain terms, register). Not just generic difficulty."
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class QVVerdict:
    passed: bool                   # True only when ALL five axes pass
    axis_results: dict[str, dict]  # {"A_translatable": {"pass": bool, "reason": str}, …}
    feedback: str                  # Human-readable summary for the Challenger

    # Strict bool coercion — "false" must never become True
    # (matches the codex-flagged fix in the reference repo)
    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return False


def _parse_verdict(data: dict) -> "QVVerdict | None":
    """Parse the QV JSON response into a QVVerdict."""
    axes = [
        "A_translatable",
        "B_challenge_valid",
        "C_reference_quality",
        "D_rubric_coherence",
        "E_difficulty_fit",
    ]
    axis_results: dict[str, dict] = {}
    failed_axes: list[str] = []

    for ax in axes:
        entry = data.get(ax)
        if not isinstance(entry, dict):
            # Missing axis — treat as failed
            axis_results[ax] = {"pass": False, "reason": "axis missing from QV output"}
            failed_axes.append(ax)
            continue

        passed = QVVerdict._coerce_bool(entry.get("pass", False))
        reason = str(entry.get("reason", "")).strip()[:200]
        axis_results[ax] = {"pass": passed, "reason": reason}
        if not passed:
            failed_axes.append(ax)

    all_passed = len(failed_axes) == 0

    if failed_axes:
        lines = ["Quality Verifier FAILED on:"]
        for ax in failed_axes:
            lines.append(f"  {ax}: {axis_results[ax]['reason']}")
        lines.append(
            "\nFix the above issues in your next attempt. "
            "All five axes must pass before the sentence is evaluated by MT models."
        )
        feedback = "\n".join(lines)
    else:
        feedback = "Quality Verifier PASSED all five axes."

    return QVVerdict(
        passed=all_passed,
        axis_results=axis_results,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# QualityVerifier class
# ---------------------------------------------------------------------------

class QualityVerifier:
    """
    Calls Gemma (via Ollama) to run the five-axis structural check on a
    ChallengeOutput before the MT solvers are invoked.

    A failing QV verdict aborts the current iteration and sends structured
    per-axis feedback to the Challenger — the same pattern used in the
    reference repo's `_qv_step`.
    """

    def __init__(self, cfg: OllamaConfig, tgt_lang: str) -> None:
        self.cfg = cfg
        self.tgt_lang_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())

    def verify(
        self,
        source: str,
        reference: str,
        challenge_type: str,
        challenge_explanation: str,
        rubric: list[dict],
    ) -> QVVerdict:
        """
        Run the five-axis check on one ChallengeOutput.

        Returns a QVVerdict. On LLM or parse failure, returns a failed
        verdict with a descriptive reason so the inner loop can retry.
        """
        messages = [
            {"role": "system", "content": _QV_SYSTEM},
            {
                "role": "user",
                "content": _build_qv_prompt(
                    source, reference, challenge_type,
                    challenge_explanation, rubric, self.tgt_lang_name,
                ),
            },
        ]

        try:
            raw = call_ollama(
                messages=messages,
                model=self.cfg.model,
                temperature=0.1,      # low temperature: this is a yes/no structural check
                base_url=self.cfg.base_url,
                timeout_s=self.cfg.timeout_s,
            )
        except RuntimeError as e:
            logger.error(f"QualityVerifier LLM call failed: {e}")
            return self._fail_verdict(f"LLM call failed: {e}")

        data = extract_json_object(raw)
        if data is None:
            logger.warning(f"QV returned non-JSON: {raw[:200]!r}")
            return self._fail_verdict("Quality Verifier returned non-JSON output.")

        verdict = _parse_verdict(data)
        if verdict is None:
            return self._fail_verdict("Quality Verifier output could not be parsed.")

        log_level = logging.INFO if verdict.passed else logging.WARNING
        logger.log(
            log_level,
            f"QV {'PASSED' if verdict.passed else 'FAILED'} | "
            + " | ".join(
                f"{ax.split('_', 1)[1]}={'✓' if v['pass'] else '✗'}"
                for ax, v in verdict.axis_results.items()
            ),
        )
        return verdict

    @staticmethod
    def _fail_verdict(reason: str) -> QVVerdict:
        axes = [
            "A_translatable", "B_challenge_valid",
            "C_reference_quality", "D_rubric_coherence", "E_difficulty_fit",
        ]
        return QVVerdict(
            passed=False,
            axis_results={ax: {"pass": False, "reason": reason} for ax in axes},
            feedback=f"Quality Verifier error: {reason}\nRetry with a new sentence.",
        )
