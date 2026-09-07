"""
Challenger agent — uses Gemma 4:31b (Ollama) to generate English source
sentences specifically designed to challenge weak MT systems while remaining
tractable for strong ones.

Each generated challenge includes:
- The source English sentence
- A reference (gold) translation the judge can compare against
- The challenge type (idiom, morphological, etc.)
- A rubric (weighted criteria for evaluating the translation)
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

from .config import LANG_NAMES, OllamaConfig
from .utils import call_ollama, extract_json_object

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Challenge type catalogue
# ---------------------------------------------------------------------------
# Each entry: (type_name, description_for_prompt)
# The Challenger cycles through all types to ensure diversity.
CHALLENGE_TYPES: list[tuple[str, str]] = [
    (
        "idiom",
        "An English idiom or fixed expression with NO word-for-word equivalent in the "
        "target language (e.g., 'kick the bucket', 'spill the beans', 'bite the bullet'). "
        "The weak model will translate literally; the strong model finds the idiomatic equivalent.",
    ),
    (
        "morphological",
        "A sentence requiring precise grammatical gender, case agreement, or noun "
        "inflection in the target language that small MT models frequently get wrong "
        "(e.g., adjective-noun agreement, genitive constructions, pronoun gender).",
    ),
    (
        "syntactic_inversion",
        "A sentence with clause fronting, topicalization, complex embedding, or a "
        "relative clause that forces major word-order restructuring in the target language. "
        "Weak models produce ungrammatical or literal word-for-word output.",
    ),
    (
        "lexical_ambiguity",
        "A sentence containing a word or phrase with multiple distinct target-language "
        "equivalents depending on context (e.g., English 'bank' = Finanzbank vs. Flussufer; "
        "'address' = Adresse vs. ansprechen vs. Rede). Context must resolve which sense.",
    ),
    (
        "negation_scope",
        "A sentence where negation scope, double negation, or litotes is semantically "
        "critical. Weak models often misplace or drop the negation, inverting the meaning.",
    ),
    (
        "cultural_reference",
        "A sentence referencing a culture-specific institution, practice, unit, or concept "
        "that lacks a direct target-language equivalent and requires a functional translation "
        "(e.g., 'call 911', 'file for Chapter 11', 'taking the Fifth').",
    ),
    (
        "register_formality",
        "A sentence where the English register (casual vs. formal) must map correctly to "
        "the target language's grammaticalized politeness system "
        "(e.g., German du vs. Sie, Spanish tú vs. usted). "
        "Weak models default to one register regardless of context.",
    ),
    (
        "tense_aspect",
        "A sentence where English tense or aspect maps non-trivially onto the target "
        "language (e.g., English present-perfect vs. German Perfekt/Präteritum distinction; "
        "progressive aspect implying ongoing action that German lacks).",
    ),
    (
        "technical_domain",
        "A sentence from a specialized domain (medical, legal, engineering, finance) "
        "requiring precise domain-specific terminology. The weak model uses generic "
        "vocabulary; the strong model renders the correct technical term.",
    ),
    (
        "proverb_equivalent",
        "An English proverb or maxim that must be rendered as the TARGET language's "
        "OWN proverb with an equivalent meaning — NOT a literal translation. "
        "Example: 'It's raining cats and dogs' → German 'Es regnet in Strömen' (correct), "
        "NOT 'Es regnet Katzen und Hunde' (wrong literal rendering).",
    ),
]

# Cycle index — advanced by each call to _next_challenge_type()
_CYCLE_IDX: int = 0


def _next_challenge_type() -> tuple[str, str]:
    """Return the next challenge type in round-robin order."""
    global _CYCLE_IDX
    ct = CHALLENGE_TYPES[_CYCLE_IDX % len(CHALLENGE_TYPES)]
    _CYCLE_IDX += 1
    return ct


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SYSTEM = """\
You are a computational linguistics expert who designs machine translation stress tests.

Your goal: create English source sentences that DISCRIMINATE between a weak MT system \
(Helsinki-NLP MarianMT, ~74M params) and a strong MT system (Meta NLLB-200, 1.3B params).

A GOOD challenge sentence:
  • Makes the WEAK model score ≤ 0.55/1.0  (it fails significantly)
  • Makes the STRONG model score ≥ 0.68/1.0 (it handles it well)
  • Is grammatically correct, natural, everyday English (8–30 words)
  • Exemplifies the specified challenge type clearly

You MUST output ONLY a valid JSON object — no prose, no markdown, no explanation outside the JSON.\
"""


def _build_user_prompt(
    tgt_lang_name: str,
    ct_name: str,
    ct_desc: str,
    feedback: str,
    iteration: int,
    seen_sources: set[str] | None = None,
    accepted_types: list[str] | None = None,
) -> str:
    fb_block = ""
    if feedback:
        fb_block = (
            f"\n\nFEEDBACK FROM ITERATION {iteration} (address this in your new sentence):\n"
            f"{feedback}\n"
            "Generate a substantially DIFFERENT sentence that resolves the above issue.\n"
        )

    # Negative-example block — prevents the Challenger from repeating sentences
    dedup_block = ""
    if seen_sources and len(seen_sources) >= 3:
        import random
        sample = random.sample(sorted(seen_sources), k=min(6, len(seen_sources)))
        lines = "\n".join(f'  - "{s}"' for s in sample)
        dedup_block = (
            f"\nDO NOT generate any of the following sentences (or close variants):\n"
            f"{lines}\n"
        )

    # Diversity hint — encourage new challenge types when some are already accepted
    diversity_block = ""
    if accepted_types:
        already = ", ".join(f"'{t}'" for t in accepted_types[-5:])
        diversity_block = (
            f"\nNOTE: You have already generated ACCEPTED sentences of types: {already}. "
            f"Prioritise a DIFFERENT angle within '{ct_name}' — do not repeat the same "
            f"sentence structure or lexical pattern.\n"
        )

    return f"""\
{fb_block}
{dedup_block}{diversity_block}
Generate ONE English sentence as a {tgt_lang_name} translation challenge.

CHALLENGE TYPE : {ct_name}
DESCRIPTION    : {ct_desc}

Also provide:
  1. The REFERENCE translation — what a perfect {tgt_lang_name} translation must be.
  2. A RUBRIC with 3–5 evaluation criteria.

Rubric rules:
  • Each criterion is a POSITIVE statement ("The translation correctly renders…").
  • Integer weight in [1, 5].
  • NO negative criteria ("does not use wrong word" is forbidden).

Output EXACTLY this JSON (no other text):
{{
  "source_sentence": "<natural English, 8–30 words>",
  "reference_translation": "<ideal {tgt_lang_name} translation>",
  "challenge_type": "{ct_name}",
  "challenge_explanation": "<1–2 sentences: WHY the weak model will fail>",
  "rubric": [
    {{"criterion": "<positive statement>", "weight": <1-5>}},
    {{"criterion": "<positive statement>", "weight": <1-5>}}
  ]
}}

Iteration: {iteration + 1}. Be creative — each call should produce a UNIQUE sentence.\
"""


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class ChallengeOutput:
    source_sentence: str
    reference_translation: str
    challenge_type: str
    challenge_explanation: str
    rubric: list[dict]  # [{"criterion": str, "weight": int}, ...]


def _parse_challenge(data: dict) -> Optional["ChallengeOutput"]:
    """Validate and convert a parsed JSON dict into a ChallengeOutput."""
    try:
        src = str(data.get("source_sentence", "")).strip()
        ref = str(data.get("reference_translation", "")).strip()
        ct = str(data.get("challenge_type", "unknown")).strip()
        expl = str(data.get("challenge_explanation", "")).strip()
        rubric_raw = data.get("rubric", [])
    except (AttributeError, TypeError):
        return None

    # Basic sanity on source sentence
    word_count = len(src.split())
    if word_count < 4 or word_count > 80:
        logger.debug(f"Source sentence rejected (word_count={word_count}): '{src[:60]}'")
        return None
    if not ref:
        logger.debug("Empty reference translation")
        return None
    if not isinstance(rubric_raw, list) or len(rubric_raw) < 2:
        logger.debug("Rubric missing or has < 2 criteria")
        return None

    rubric: list[dict] = []
    for item in rubric_raw:
        if not isinstance(item, dict):
            continue
        criterion = str(item.get("criterion", "")).strip()
        try:
            weight = max(1, min(5, int(item.get("weight", 2))))
        except (ValueError, TypeError):
            weight = 2
        if criterion:
            rubric.append({"criterion": criterion, "weight": weight})

    if len(rubric) < 2:
        logger.debug("Parsed rubric has < 2 valid criteria")
        return None

    return ChallengeOutput(
        source_sentence=src,
        reference_translation=ref,
        challenge_type=ct,
        challenge_explanation=expl,
        rubric=rubric,
    )


# ---------------------------------------------------------------------------
# Challenger class
# ---------------------------------------------------------------------------

class Challenger:
    """Calls Gemma (via Ollama) to generate MT-discriminative source sentences."""

    def __init__(self, cfg: OllamaConfig, tgt_lang: str) -> None:
        self.cfg = cfg
        self.tgt_lang_name = LANG_NAMES.get(tgt_lang, tgt_lang.upper())
        self._last_ct: Optional[tuple[str, str]] = None  # allows retry with same type

    def generate(
        self,
        feedback: str = "",
        iteration: int = 0,
        force_challenge_type: Optional[tuple[str, str]] = None,
        seen_sources: "set[str] | None" = None,
        accepted_types: "list[str] | None" = None,
    ) -> Optional[ChallengeOutput]:
        """
        Ask the Challenger LLM to produce one ChallengeOutput.

        Args:
            feedback:             Rejection feedback from the previous iteration.
            iteration:            Current inner-loop iteration count.
            force_challenge_type: Override the round-robin type selection.
            seen_sources:         Previously generated source sentences (dedup).
            accepted_types:       Challenge types already accepted (diversity).
        """
        if force_challenge_type is not None:
            ct_name, ct_desc = force_challenge_type
        elif feedback and self._last_ct is not None:
            # Retry with the same challenge type so feedback is relevant
            ct_name, ct_desc = self._last_ct
        else:
            ct_name, ct_desc = _next_challenge_type()

        self._last_ct = (ct_name, ct_desc)

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _build_user_prompt(
                    self.tgt_lang_name, ct_name, ct_desc, feedback, iteration,
                    seen_sources=seen_sources,
                    accepted_types=accepted_types,
                ),
            },
        ]

        try:
            raw = call_ollama(
                messages=messages,
                model=self.cfg.model,
                temperature=self.cfg.challenger_temperature,
                base_url=self.cfg.base_url,
                timeout_s=self.cfg.timeout_s,
            )
        except RuntimeError as e:
            logger.error(f"Challenger LLM call failed: {e}")
            return None

        data = extract_json_object(raw)
        if data is None:
            logger.warning(
                f"Challenger returned non-JSON (iteration={iteration}): {raw[:200]!r}"
            )
            return None

        challenge = _parse_challenge(data)
        if challenge is None:
            logger.warning(f"Challenger output failed validation (iteration={iteration}): {data}")
        else:
            logger.debug(
                f"Challenger generated: type={challenge.challenge_type!r} "
                f"src='{challenge.source_sentence[:70]}'"
            )
        return challenge
