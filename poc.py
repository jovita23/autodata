"""
AutoData MT PoC — main entry point.

Usage
-----
  python -m autodata_mt.poc --target-pairs 20 --lang-pair en-de
  python -m autodata_mt.poc --target-pairs 20 --lang-pair en-es --device cpu
  python -m autodata_mt.poc --help
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

from .challenger import Challenger
from .config import (
    AcceptanceCriteria,
    LANG_NAMES,
    MTConfig,
    OllamaConfig,
    PoCConfig,
    STRONG_MODELS,
    WEAK_MODELS,
)
from .inner_loop import AcceptedSample, run_inner_loop
from .judge import Judge
from .mt_solvers import load_solvers
from .quality_verifier import QualityVerifier
from .utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "AutoData Agentic Self-Instruct PoC — Machine Translation\n"
            "Generates discriminative EN↔{DE,ES} sentence pairs where\n"
            "a strong MT model succeeds and a weak MT model fails."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core settings
    p.add_argument(
        "--lang-pair", choices=["en-de", "en-es"], default="en-de",
        help="Language pair (default: en-de)",
    )
    p.add_argument(
        "--target-pairs", type=int, default=20,
        help="Number of accepted sentence pairs to collect (default: 20)",
    )

    # Inner-loop tuning
    p.add_argument(
        "--max-iterations", type=int, default=6,
        help="Max Challenger iterations per sample (default: 6)",
    )
    p.add_argument(
        "--num-solver-samples", type=int, default=2,
        help="Times each solver runs per iteration — reduces variance (default: 2)",
    )

    # Acceptance thresholds
    p.add_argument("--weak-max",   type=float, default=0.55, metavar="F",
                   help="Weak solver max score threshold (default: 0.55)")
    p.add_argument("--strong-min", type=float, default=0.68, metavar="F",
                   help="Strong solver min score threshold (default: 0.68)")
    p.add_argument("--min-gap",    type=float, default=0.20, metavar="F",
                   help="Min strong−weak gap required (default: 0.20)")

    # Model configuration
    p.add_argument(
        "--ollama-model", default="gemma4:31b",
        help="Ollama model for Challenger + Judge (default: gemma4:31b)",
    )
    p.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for HuggingFace MT models (default: auto)",
    )

    # Output
    p.add_argument(
        "--output-dir", default="output_poc",
        help="Directory for saved results JSON (default: output_poc/)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 74


def _print_banner(cfg: PoCConfig, src: str, tgt: str) -> None:
    print()
    print("═" * 74)
    print("  AutoData — Agentic Self-Instruct PoC  |  Machine Translation")
    print("═" * 74)
    print(f"  Language pair    : {src.upper()} → {tgt.upper()}")
    print(f"  Target pairs     : {cfg.target_accepted}")
    print(f"  Challenger/Judge : {cfg.ollama.model}  (Ollama)")
    print(f"  Weak solver      : {cfg.mt.weak_model_id}")
    print(f"  Strong solver    : {cfg.mt.strong_model_id}")
    print(
        f"  Acceptance gate  : "
        f"weak ≤ {cfg.acceptance.weak_max}  |  "
        f"strong ≥ {cfg.acceptance.strong_min}  |  "
        f"gap ≥ {cfg.acceptance.min_gap}"
    )
    print("═" * 74)


def _print_accepted(sample: AcceptedSample, idx: int, tgt: str) -> None:
    tgt_name = LANG_NAMES.get(tgt, tgt.upper())
    print(f"\n{_SEP}")
    print(
        f"  Sample #{idx:<3}  |  "
        f"Type: {sample.challenge.challenge_type:<25}  |  "
        f"Rounds: {sample.rounds_taken}"
    )
    print(_SEP)
    print(f"  EN     : {sample.challenge.source_sentence}")
    print(f"  REF    : {sample.challenge.reference_translation}")
    print(f"  WEAK  ({tgt_name:>7}) : {sample.weak_translation}")
    print(f"          score = {sample.acceptance.weak_avg:.3f}  ← {sample.weak_scores[0].overall_comment[:80]}")
    print(f"  STRONG ({tgt_name:>7}) : {sample.strong_translation}")
    print(f"          score = {sample.acceptance.strong_avg:.3f}  ← {sample.strong_scores[0].overall_comment[:80]}")
    print(f"  Gap    : {sample.acceptance.gap:.3f}")
    print(f"  Why?   : {sample.challenge.challenge_explanation[:110]}")


def _print_summary(samples: list[AcceptedSample], elapsed_s: float) -> None:
    if not samples:
        print("\nNo samples accepted.")
        return

    weak_avgs   = [s.acceptance.weak_avg   for s in samples]
    strong_avgs = [s.acceptance.strong_avg for s in samples]
    gaps        = [s.acceptance.gap        for s in samples]
    rounds      = [s.rounds_taken          for s in samples]

    def _sd(vals: list[float]) -> float:
        return stdev(vals) if len(vals) > 1 else 0.0

    print(f"\n{'═' * 74}")
    print("  FINAL SUMMARY")
    print(f"{'═' * 74}")
    print(f"  Accepted samples     : {len(samples)}")
    print(f"  Total wall time      : {elapsed_s:.1f}s  ({elapsed_s / len(samples):.1f}s/sample)")
    print()
    print(f"  Weak  avg score      : {mean(weak_avgs):.3f} ± {_sd(weak_avgs):.3f}")
    print(f"  Strong avg score     : {mean(strong_avgs):.3f} ± {_sd(strong_avgs):.3f}")
    print(f"  Gap (strong − weak)  : {mean(gaps):.3f} ± {_sd(gaps):.3f}   "
          f"[min={min(gaps):.3f}, max={max(gaps):.3f}]")
    print(f"  Avg rounds to accept : {mean(rounds):.1f}")
    print()

    # Challenge type histogram
    type_counts: dict[str, int] = {}
    for s in samples:
        type_counts[s.challenge.challenge_type] = (
            type_counts.get(s.challenge.challenge_type, 0) + 1
        )
    print("  Challenge type distribution:")
    for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        pct = 100.0 * count / len(samples)
        print(f"    {ct:<28} {bar:<20} {count:>3}  ({pct:.0f}%)")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _avg_per_criterion(scores: list) -> list[float]:
    """Average per-criterion scores across all N solver runs."""
    if not scores:
        return []
    n = len(scores)
    rubric_len = len(scores[0].per_criterion)
    return [
        round(sum(s.per_criterion[i] for s in scores) / n, 4)
        for i in range(rubric_len)
    ]


def _to_record(sample: AcceptedSample, tgt: str) -> dict:
    return {
        "source_en": sample.challenge.source_sentence,
        f"reference_{tgt}": sample.challenge.reference_translation,
        f"weak_mt_{tgt}": sample.weak_translation,
        f"strong_mt_{tgt}": sample.strong_translation,
        "weak_score": round(sample.acceptance.weak_avg, 4),
        "strong_score": round(sample.acceptance.strong_avg, 4),
        "gap": round(sample.acceptance.gap, 4),
        "challenge_type": sample.challenge.challenge_type,
        "challenge_explanation": sample.challenge.challenge_explanation,
        "rubric": sample.challenge.rubric,
        "weak_per_criterion": _avg_per_criterion(sample.weak_scores),
        "strong_per_criterion": _avg_per_criterion(sample.strong_scores),
        "rounds_taken": sample.rounds_taken,
    }


def _save(samples: list[AcceptedSample], path: Path, tgt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [_to_record(s, tgt) for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(records)} records to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    setup_logging(args.log_level)

    src, tgt = args.lang_pair.split("-")

    cfg = PoCConfig(
        ollama=OllamaConfig(
            base_url=args.ollama_url,
            model=args.ollama_model,
        ),
        mt=MTConfig(
            src_lang=src,
            tgt_lang=tgt,
            weak_model_id=WEAK_MODELS[args.lang_pair],
            strong_model_id=STRONG_MODELS[args.lang_pair],
            device=args.device,
        ),
        acceptance=AcceptanceCriteria(
            weak_max=args.weak_max,
            strong_min=args.strong_min,
            min_gap=args.min_gap,
            max_iterations=args.max_iterations,
        ),
        target_accepted=args.target_pairs,
        num_solver_samples=args.num_solver_samples,
        output_dir=args.output_dir,
    )

    _print_banner(cfg, src, tgt)

    # ── Load HuggingFace MT models ────────────────────────────────────────
    logger.info("Loading HuggingFace MT models…")
    try:
        weak_solver, strong_solver = load_solvers(cfg.mt)
    except Exception as exc:
        logger.error(f"Failed to load MT models: {exc}")
        sys.exit(1)

    # ── Initialise Ollama agents ──────────────────────────────────────────
    challenger       = Challenger(cfg.ollama, tgt)
    judge            = Judge(cfg.ollama, tgt)
    quality_verifier = QualityVerifier(cfg.ollama, tgt)

    # ── Output path ───────────────────────────────────────────────────────
    output_path = Path(cfg.output_dir) / f"{args.lang_pair}_autodata_poc.json"

    # ── Main generation loop ──────────────────────────────────────────────
    accepted: list[AcceptedSample] = []
    seen_sources: set[str] = set()       # cross-attempt deduplication
    accepted_types: list[str] = []       # diversity tracking
    attempts = 0
    start = time.time()

    while len(accepted) < cfg.target_accepted:
        attempts += 1
        logger.info(
            f"\n{'─'*60}\n"
            f"Attempt {attempts} | Accepted so far: {len(accepted)}/{cfg.target_accepted}"
        )

        sample = run_inner_loop(
            challenger=challenger,
            weak_solver=weak_solver,
            strong_solver=strong_solver,
            judge=judge,
            quality_verifier=quality_verifier,
            cfg=cfg,
            sample_idx=len(accepted) + 1,
            seen_sources=seen_sources,
            accepted_types=accepted_types,
        )

        if sample is not None:
            accepted.append(sample)
            accepted_types.append(sample.challenge.challenge_type)
            _print_accepted(sample, len(accepted), tgt)
            # Incremental save after every accepted sample
            _save(accepted, output_path, tgt)

        # Safety valve: stop if too many outer attempts fail
        max_outer_attempts = cfg.target_accepted * 8
        if attempts >= max_outer_attempts:
            logger.warning(
                f"Reached {max_outer_attempts} outer attempts with only "
                f"{len(accepted)} accepted — stopping early."
            )
            break

    elapsed = time.time() - start
    _print_summary(accepted, elapsed)

    if accepted:
        print(f"\n  Results saved to: {output_path}")
    else:
        print("\n  No samples were accepted. Check Ollama connectivity and model availability.")


if __name__ == "__main__":
    main()
