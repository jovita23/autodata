"""
AutoData inner loop adapted for Machine Translation.

Implements the 8-step Agentic Self-Instruct protocol:

  REPEAT until ACCEPTED or max_iterations exhausted:
    1.  Challenger generates  (source_sentence, reference_translation, rubric)
    2.  Quality Verifier (LLM) checks five axes:
          A. TRANSLATABLE      — genuine natural English
          B. CHALLENGE_VALID   — sentence actually exhibits the challenge type
          C. REFERENCE_QUALITY — reference is correct and idiomatic
          D. RUBRIC_COHERENCE  — positive criteria, correct weights
          E. DIFFICULTY_FIT    — targets a known MarianMT weakness
    3.  If QV fails            → iterate with per-axis feedback
    4.  Weak solver translates the sentence  ×  N
    5.  Judge scores each weak translation against the rubric
    6.  Strong solver translates the sentence  ×  N
    7.  Judge scores each strong translation against the rubric
    8.  Acceptance check:
          weak_avg ≤ 0.55  AND  strong_avg ≥ 0.68  AND  gap ≥ 0.20
        If ACCEPTED → return AcceptedSample
        Else        → Judge generates targeted rejection feedback (→ step 1)
                      Falls back to template feedback on LLM failure.

Improvements over the original design:
  • QualityVerifier (LLM) — replaces the cheap length/rubric-count check.
    Gemma runs a five-axis structural pass on every Challenger output before
    any MT inference is triggered, matching the reference repo's _qv_step.
  • Judge.generate_feedback() — Gemma analyzes the actual per-criterion
    breakdowns and writes sentence-specific guidance for the Challenger,
    replacing the generic template-based rejection message.
  • seen_sources deduplication — previously generated sentences are tracked
    and injected into Challenger prompts as negative examples.
  • accepted_types history — Challenger is told which challenge types already
    succeeded so it diversifies across the run.
"""

import logging
from dataclasses import dataclass, field

from .acceptance import AcceptanceResult, check_acceptance, format_rejection_feedback
from .challenger import Challenger, ChallengeOutput
from .config import PoCConfig
from .judge import Judge, TranslationScore
from .mt_solvers import WeakMTSolver, StrongMTSolver
from .quality_verifier import QualityVerifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class AcceptedSample:
    """A source sentence that passed the AutoData acceptance gate."""
    challenge: ChallengeOutput
    weak_scores: list[TranslationScore]
    strong_scores: list[TranslationScore]
    acceptance: AcceptanceResult
    rounds_taken: int               # which iteration was accepted (1-based)

    # Convenience properties
    @property
    def weak_translation(self) -> str:
        return self.weak_scores[0].candidate if self.weak_scores else ""

    @property
    def strong_translation(self) -> str:
        return self.strong_scores[0].candidate if self.strong_scores else ""


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------

def run_inner_loop(
    challenger: Challenger,
    weak_solver: WeakMTSolver,
    strong_solver: StrongMTSolver,
    judge: Judge,
    quality_verifier: QualityVerifier,
    cfg: PoCConfig,
    sample_idx: int = 0,
    seen_sources: "set[str] | None" = None,
    accepted_types: "list[str] | None" = None,
) -> "AcceptedSample | None":
    """
    Run the 8-step AutoData inner loop for one source-sentence attempt.

    Args:
        quality_verifier: LLM-based five-axis structural gate (Steps 2–3).
        seen_sources:     Shared set of already-generated source sentences for
                          deduplication. Updated in-place when a new sentence is
                          generated (accepted or not).
        accepted_types:   List of challenge types that have already been accepted
                          across the whole run. Passed to the Challenger to
                          encourage diversity.

    Returns an AcceptedSample if the criteria are met within the allowed
    iterations, or None if all iterations are exhausted.
    """
    if seen_sources is None:
        seen_sources = set()
    if accepted_types is None:
        accepted_types = []

    feedback = ""
    max_iter = cfg.acceptance.max_iterations
    n_samples = cfg.num_solver_samples

    for iteration in range(max_iter):
        tag = f"[#{sample_idx} iter {iteration + 1}/{max_iter}]"

        # ── Step 1: Challenger generates a source sentence + rubric ────────
        logger.info(f"{tag} Calling Challenger…")
        challenge = challenger.generate(
            feedback=feedback,
            iteration=iteration,
            seen_sources=seen_sources,
            accepted_types=accepted_types,
        )

        if challenge is None:
            feedback = (
                "Previous attempt returned invalid JSON or an empty sentence. "
                "Ensure the output is a single valid JSON object."
            )
            logger.warning(f"{tag} Challenger returned None — retrying")
            continue

        logger.info(
            f"{tag} challenge_type={challenge.challenge_type!r} | "
            f"src='{challenge.source_sentence[:70]}'"
        )

        # ── Steps 2–3: Quality Verifier (LLM) ────────────────────────────────
        logger.info(f"{tag} Calling Quality Verifier…")
        verdict = quality_verifier.verify(
            source=challenge.source_sentence,
            reference=challenge.reference_translation,
            challenge_type=challenge.challenge_type,
            challenge_explanation=challenge.challenge_explanation,
            rubric=challenge.rubric,
        )

        if not verdict.passed:
            feedback = verdict.feedback
            logger.warning(f"{tag} QV failed — feedback sent to Challenger")
            # Still track the sentence so the Challenger doesn't regenerate it
            seen_sources.add(challenge.source_sentence.strip().lower())
            continue

        # Track the source sentence for cross-iteration deduplication
        seen_sources.add(challenge.source_sentence.strip().lower())

        # ── Steps 4–5: Weak solver ×N + Judge ──────────────────────────────
        weak_scores: list[TranslationScore] = []
        for k in range(n_samples):
            translation = weak_solver.translate(challenge.source_sentence)
            score = judge.score(
                source=challenge.source_sentence,
                reference=challenge.reference_translation,
                candidate=translation,
                rubric=challenge.rubric,
            )
            weak_scores.append(score)
            logger.debug(
                f"{tag} weak[{k}]: '{translation[:70]}' score={score.weighted_score:.3f}"
            )

        # ── Steps 6–7: Strong solver ×N + Judge ────────────────────────────
        strong_scores: list[TranslationScore] = []
        for k in range(n_samples):
            translation = strong_solver.translate(challenge.source_sentence)
            score = judge.score(
                source=challenge.source_sentence,
                reference=challenge.reference_translation,
                candidate=translation,
                rubric=challenge.rubric,
            )
            strong_scores.append(score)
            logger.debug(
                f"{tag} strong[{k}]: '{translation[:70]}' score={score.weighted_score:.3f}"
            )

        # ── Step 8: Acceptance check ────────────────────────────────────────
        result = check_acceptance(weak_scores, strong_scores, cfg.acceptance)

        status = "ACCEPTED ✓" if result.accepted else "rejected ✗"
        logger.info(
            f"{tag} weak_avg={result.weak_avg:.3f}  "
            f"strong_avg={result.strong_avg:.3f}  "
            f"gap={result.gap:.3f}  {status}"
        )

        if result.accepted:
            return AcceptedSample(
                challenge=challenge,
                weak_scores=weak_scores,
                strong_scores=strong_scores,
                acceptance=result,
                rounds_taken=iteration + 1,
            )

        # Not accepted — ask the Judge for targeted, sentence-specific feedback.
        # Fall back to the template formatter if the LLM call fails.
        logger.info(
            f"{tag} Rejection reasons: "
            + " | ".join(result.fail_reasons)
        )
        logger.info(f"{tag} Calling Judge for rejection feedback…")
        judge_feedback = judge.generate_feedback(
            source=challenge.source_sentence,
            reference=challenge.reference_translation,
            rubric=challenge.rubric,
            weak_scores=weak_scores,
            strong_scores=strong_scores,
            fail_reasons=result.fail_reasons,
        )
        if judge_feedback:
            logger.info(f"{tag} Judge feedback: {judge_feedback[:120]}…")
            feedback = judge_feedback
        else:
            logger.warning(f"{tag} Judge feedback unavailable — using template fallback")
            feedback = format_rejection_feedback(result, weak_scores, strong_scores)

    logger.warning(
        f"[#{sample_idx}] Exhausted {max_iter} iterations — no accepted sample produced."
    )
    return None
