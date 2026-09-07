# AutoData MT — Agentic Self-Instruct for Machine Translation

A Proof-of-Concept implementation of Meta FAIR's **AutoData** framework applied
to **Machine Translation (MT)** discriminative data generation.

> **Paper / blog**: [Autodata: an automatic data scientist to create high-quality data](https://facebookresearch.github.io/RAM/blogs/autodata/) — Kulikov et al., Meta FAIR 2026  
> **Reference implementation**: [Lauorie/Autodata](https://github.com/Lauorie/Autodata)

---

## Table of Contents

- [Version History](#version-history)
1. [What is AutoData?](#1-what-is-autodata)
2. [How This code Adapts AutoData for MT](#2-how-this-code-adapts-autodata-for-mt)
3. [Architecture](#3-architecture)
4. [The Inner Loop (8 Steps)](#4-the-inner-loop-8-steps)
5. [Acceptance Criteria](#5-acceptance-criteria)
6. [Challenge Types](#6-challenge-types)
7. [Installation](#7-installation)
8. [Running the code](#8-running-the-code)
9. [Expected Output](#9-expected-output)
10. [Code Structure](#10-code-structure)
11. [Design Decisions and Trade-offs](#11-design-decisions-and-trade-offs)
12. [What We Changed vs. the Blog](#12-what-we-changed-vs-the-blog)
13. [Challenges and Limitations](#13-challenges-and-limitations)

---

## Version History

### v1 — Base Implementation
First working inner loop. Rejection feedback is **template-generated** — hardcoded strings based on which acceptance condition failed. Quality gate is a cheap length + rubric-count check (no LLM call).

| Component | v1 |
|---|---|
| Challenger | Gemma 4:31b — generates source sentence + rubric |
| Quality gate | Length check (`≥ 4 words`) + rubric count (`≥ 2 criteria`) |
| Weak solver | Helsinki-NLP/opus-mt-en-de (MarianMT 74M) |
| Strong solver | NLLB-200-distilled-1.3B |
| Judge | Gemma 4:31b — rubric scoring |
| Rejection feedback | Template strings keyed on which threshold failed |
| Deduplication | None |
| Diversity tracking | None |

---

### v2 — Judge-Generated Feedback + Deduplication
Rejection feedback is now **written by the Judge LLM** — Gemma receives the actual per-criterion score breakdown for both models and writes sentence-specific diagnosis and guidance. Falls back to v1 template on LLM failure. Added cross-attempt deduplication and challenge-type diversity tracking.

| Component | v2 change |
|---|---|
| Rejection feedback | `Judge.generate_feedback()` — per-criterion breakdown → targeted advice |
| Deduplication | `seen_sources` set injected into Challenger prompt as negative examples |
| Diversity tracking | `accepted_types` list — Challenger told which types already succeeded |

---

### v3 — LLM Quality Verifier
The cheap length check is replaced by a **full LLM Quality Verifier** (new module: [autodata_mt/quality_verifier.py](autodata_mt/quality_verifier.py)). Gemma runs a five-axis structural pass on every Challenger output *before* any MT inference is triggered — matching the reference repo's `_qv_step`.

| Component | v3 change |
|---|---|
| Quality gate | `QualityVerifier.verify()` — 5-axis LLM call at temperature 0.1 |
| QV axis A | **Translatable** — genuine natural English |
| QV axis B | **Challenge_Valid** — sentence actually exhibits the claimed challenge type |
| QV axis C | **Reference_Quality** — reference is idiomatic, not word-for-word |
| QV axis D | **Rubric_Coherence** — positive criteria, integer weights in [1,5] |
| QV axis E | **Difficulty_Fit** — challenge targets a specific MarianMT weakness |
| Strict bool coercion | `"false"` string → `False` (never coerced to `True`) |

---

## 1. What is AutoData?

**AutoData** converts inference compute into training-data quality. Instead of
generating synthetic data in a single shot (Self-Instruct style), an ensemble of
agents iteratively *proposes*, *critiques*, and *validates* each example until it
satisfies a **discriminative signal**:

> *Data where a strong model succeeds but a weak model fails.*

The core result from the blog: on CS-research QA, single-shot CoT Self-Instruct
yields a weak/strong gap of only **1.9 percentage points**. The Agentic
Self-Instruct loop widens this gap to **34 pp** (weak ≈ 44 %, strong ≈ 78 %).
The harder data also produces measurably better GRPO-trained models.

### The Agentic Self-Instruct Loop

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                       AutoData Inner Loop                           │
 └─────────────────────────────────────────────────────────────────────┘
        ▲  feedback (if rejected)
        │
  ┌─────────────┐   source + rubric   ┌─────────────────────────────┐
  │  CHALLENGER │ ──────────────────► │    Quality Verification     │
  │  (LLM)      │                     └──────────────┬──────────────┘
  └─────────────┘                                    │ pass
                                     ┌──────────────┴──────────────┐
                                     │                              │
                              ┌──────▼──────┐              ┌────────▼──────┐
                              │ WEAK SOLVER │              │ STRONG SOLVER │
                              └──────┬──────┘              └───────┬───────┘
                                     │                              │
                              ┌──────▼──────┐              ┌───────▼───────┐
                              │    JUDGE    │              │     JUDGE     │
                              └──────┬──────┘              └───────┬───────┘
                                     │ weak_avg                    │ strong_avg
                                     └──────────────┬──────────────┘
                                                    ▼
                                         ┌──────────────────┐
                                         │ ACCEPTANCE CHECK │
                                         │ weak  ≤ 0.55     │
                                         │ strong ≥ 0.68    │
                                         │ gap   ≥ 0.20     │
                                         └────────┬─────────┘
                                                  │
                                   ┌──────────────┴──────────────┐
                                   ▼                             ▼
                             ✓ ACCEPTED                   ✗ REJECTED
                           save sample               feedback → Challenger
```

---

## 2. How This  Adapts AutoData for MT

The original AutoData paper uses **CS-research QA** as the task (Challenger
writes questions; solvers answer them). This  re-maps every role to MT:

| AutoData role | Original task | **This MT ** |
|---|---|---|
| **Challenger** | Generates questions + rubric | Gemma 4:31b (Ollama) — generates *hard-to-translate English sentences* + rubric |
| **Weak Solver** | Small LM (Qwen-4B) | `Helsinki-NLP/opus-mt-en-de` — small MarianMT (~74 M params) |
| **Strong Solver** | Frontier LM (Qwen-397B) | `facebook/nllb-200-distilled-1.3B` — Meta's NLLB-200 MT model |
| **Judge** | Kimi-K2.6 scoring rubric | Gemma 4:31b (Ollama) — LLM-as-judge against a translation rubric |
| **Quality Verifier** | Structural QA check | Five-axis LLM call (v3): Translatable · Challenge_Valid · Reference_Quality · Rubric_Coherence · Difficulty_Fit |

### Why these MT models?

**Weak — `Helsinki-NLP/opus-mt-en-de`**
- ~74 M parameters, MarianMT architecture, OPUS-100 training data
- Excellent at simple/literal sentences
- Known weaknesses: idioms (translates literally), morphological agreement,
  register distinctions (du/Sie), rare domain terms, complex embedded clauses

**Strong — `facebook/nllb-200-distilled-1.3B`**
- ~1.3 B parameters, NLLB-200 multilingual Transformer
- Trained on 54.5 B sentence pairs across 200 languages
- Handles idioms, complex syntax, domain vocabulary, and register correctly
- ~18× larger than the weak model → meaningful capability gap

### What makes an MT sentence "discriminative"?

The Challenger is prompted to generate sentences that exploit the *specific
weaknesses* of small MT models:

| Challenge type | Weak model fails because… | Strong model succeeds because… |
|---|---|---|
| **idiom** | Translates "kick the bucket" literally | Knows "den Löffel abgeben" |
| **morphological** | Wrong gender agreement, wrong case | Correct inflection throughout |
| **syntactic_inversion** | Rigid word order, ungrammatical | Restructures correctly |
| **lexical_ambiguity** | Picks the wrong sense of a polysemous word | Context-aware disambiguation |
| **negation_scope** | Misplaces or drops negation | Correct scope |
| **cultural_reference** | Translates "call 911" literally | Functional equivalent |
| **register_formality** | Defaults to one register | du/Sie correctly applied |
| **tense_aspect** | Maps English tense 1:1 | Correct Perfekt/Präteritum selection |
| **technical_domain** | Generic vocabulary | Correct domain term |
| **proverb_equivalent** | Literal rendering | Target-language equivalent proverb |

---

## 3. Architecture

```
/data/autodata/
├── README.md                  ← This file
├── requirements_.txt       ← Python dependencies
├── generate_samples.py        ← Pre-existing bulk sampler (unrelated)
└── autodata_mt/               ←  package
    ├── __init__.py
    ├── config.py              ← Configuration dataclasses
    ├── utils.py               ← Ollama client, JSON extraction, scoring
    ├── challenger.py          ← Challenger agent (Gemma → hard source sentences)
    ├── mt_solvers.py          ← Weak + Strong HuggingFace MT model wrappers
    ├── judge.py               ← Judge agent (Gemma → rubric scores)
    ├── acceptance.py          ← Acceptance gate + rejection feedback builder
    ├── inner_loop.py          ← 8-step AutoData inner loop
    └── .py                 ← CLI entry point + reporting
```

### Data flow for one accepted sample

```
Gemma (Ollama)
  ┌─ Challenger prompt ──────────────────────────────────────────────────┐
  │ "Generate an English sentence with challenge type: idiom             │
  │  that a small MarianMT model would mistranslate…"                   │
  └─────────────────────────────────────────────────────────────────────┘
                         │
                         ▼  JSON response
  {
    "source_sentence":       "She finally kicked the bucket.",
    "reference_translation": "Sie hat endlich den Löffel abgegeben.",
    "challenge_type":        "idiom",
    "challenge_explanation": "The weak model translates literally: …",
    "rubric": [
      {"criterion": "Idiom 'kick the bucket' rendered as the German equivalent 'den Löffel abgeben'", "weight": 5},
      {"criterion": "Grammatically correct German feminine pronoun 'sie'", "weight": 3},
      …
    ]
  }
                         │
           ┌─────────────┴──────────────┐
           │                            │
     WeakMTSolver                  StrongMTSolver
  "Sie hat endlich              "Sie hat endlich den
   den Eimer getreten."          Löffel abgegeben."
           │                            │
     Judge (Gemma)               Judge (Gemma)
     score: 0.10                 score: 0.82
           │                            │
           └─────────────┬──────────────┘
                         │
                  gap = 0.82 − 0.10 = 0.72 ≥ 0.20  ✓
                  ACCEPTED
```

---

## 4. The Inner Loop (8 Steps)

Each *attempt* to generate one accepted sample runs the following loop
(see [autodata_mt/inner_loop.py](autodata_mt/inner_loop.py)):

```
for iteration in 1 .. max_iterations:

  Step 1   Challenger.generate(feedback, iteration)
             → ChallengeOutput {source_sentence, reference_translation,
                                challenge_type, challenge_explanation, rubric}

  Steps 2-3  Quality Verifier (LLM) — five axes:
               A. Translatable, B. Challenge_Valid, C. Reference_Quality,
               D. Rubric_Coherence, E. Difficulty_Fit
             If fails → per-axis feedback → continue

  Steps 4-5  for k in 1..N:
               weak_translation_k  = WeakMTSolver.translate(source_sentence)
               weak_score_k        = Judge.score(source, ref, weak_translation_k, rubric)

  Steps 6-7  for k in 1..N:
               strong_translation_k = StrongMTSolver.translate(source_sentence)
               strong_score_k       = Judge.score(source, ref, strong_translation_k, rubric)

  Step 8     result = check_acceptance(weak_scores, strong_scores)
             if result.accepted:
                 return AcceptedSample
             else:
                 feedback = format_rejection_feedback(result, weak_scores, strong_scores)
                 # → loop with feedback

return None   # exhausted
```

**N = `--num-solver-samples`** (default 2). Running each solver twice and
averaging reduces score variance caused by deterministic beam search.

---

## 5. Acceptance Criteria

All four conditions must hold simultaneously (mirrors AutoData Table 1):

| Condition | Threshold | Meaning |
|---|---|---|
| `weak_avg ≤ weak_max` | 0.55 | Weak MT model fails on this sentence |
| `weak_avg ≥ weak_min_floor` | 0.05 | Sentence is at least minimally translatable |
| `strong_avg ≥ strong_min` | 0.68 | Strong MT model handles it well |
| `gap ≥ min_gap` | 0.20 | Meaningful discrimination |

When rejected, `format_rejection_feedback()` builds structured feedback:

```
REJECTION — all conditions must hold simultaneously:
  ✗ weak_avg=0.72 > 0.55 (sentence too easy — weak model can also translate it)

Scores: weak_avg=0.720, strong_avg=0.880, gap=0.160

GUIDANCE:
  → The sentence was too EASY. Use a stronger challenge: deeper idiom, …
  → Gap too small. The challenge must SPECIFICALLY exploit weak-model weaknesses …

  Weak model output  : 'Die Lage war sehr kompliziert.'
  Judge comment      : Translation is literal and mostly correct.
  Strong model output: 'Die Lage war außerordentlich heikel.'
  Judge comment      : Accurately renders the nuanced domain vocabulary.
```

---

## 6. Challenge Types

The Challenger cycles through 10 challenge types in round-robin order,
retrying with the same type when feedback is non-empty:

| # | Type | Example English | Weak model error | Strong model |
|---|---|---|---|---|
| 1 | `idiom` | "It's raining cats and dogs." | "Es regnet Katzen und Hunde." | "Es regnet in Strömen." |
| 2 | `morphological` | "She introduced herself to the new female colleague." | Wrong pronoun gender | Correct `sich … ihr` |
| 3 | `syntactic_inversion` | "Never have I seen such chaos." | Literal SVO order | Correct verb-second inversion |
| 4 | `lexical_ambiguity` | "We need to address this issue at the board." | Wrong sense of "address" | Context-correct `ansprechen` |
| 5 | `negation_scope` | "I didn't say she stole the money." | Drops or misplaces negation | Exact scope preserved |
| 6 | `cultural_reference` | "She threatened to call the DMV." | Literal "DMV" | Functional equivalent |
| 7 | `register_formality` | "Could you pass the salt?" (formal setting) | Casual `du`-form | Correct `Sie`-form |
| 8 | `tense_aspect` | "By the time you read this, I will have left." | Wrong tense | Correct Futur II |
| 9 | `technical_domain` | "The patient presented with acute myocardial infarction." | Generic vocabulary | Precise medical term |
| 10 | `proverb_equivalent` | "Every cloud has a silver lining." | Literal rendering | German proverb equivalent |

---

## 7. Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with `gemma4:31b` pulled
- CUDA GPU (optional but recommended for NLLB-1.3B; CPU fallback is available)

### 1. Verify Ollama is running with the model

```bash
ollama list          # should show gemma4:31b
ollama run gemma4:31b --version   # optional smoke-test
```

### 2. Install Python dependencies

```bash
cd /data/autodata
pip install -r requirements_.txt
```

> **Memory note**: The NLLB-1.3B model needs ~5 GB VRAM on GPU (or ~5 GB RAM on CPU).
> Ollama runs Gemma 4:31b in a separate process and releases GPU context between
> calls, so both can coexist on most multi-GPU setups. Use `--device cpu` if VRAM
> is tight.

---

## 8. Running the 

All commands are run from `/data/autodata/`.

### Quick start — 20 accepted EN→DE pairs (default)

```bash
python -m autodata_mt.
```

### EN→ES with 50 pairs on CPU

```bash
python -m autodata_mt. --lang-pair en-es --target-pairs 50 --device cpu
```

### Relaxed acceptance criteria (easier to accept, faster )

```bash
python -m autodata_mt. \
    --weak-max 0.65 \
    --strong-min 0.60 \
    --min-gap 0.15 \
    --target-pairs 20
```

### Stricter acceptance (matches AutoData blog thresholds)

```bash
python -m autodata_mt. \
    --weak-max 0.50 \
    --strong-min 0.75 \
    --min-gap 0.25 \
    --target-pairs 20
```

### Debug mode (verbose inner-loop traces)

```bash
python -m autodata_mt. --log-level DEBUG --target-pairs 5
```

### Full CLI reference

```
python -m autodata_mt. --help

optional arguments:
  --lang-pair          {en-de,en-es}   Language pair (default: en-de)
  --target-pairs N     Accepted pairs to collect (default: 20)
  --max-iterations N   Challenger iterations per sample (default: 6)
  --num-solver-samples N Times each solver runs per iteration (default: 2)
  --weak-max F         Weak solver max score (default: 0.55)
  --strong-min F       Strong solver min score (default: 0.68)
  --min-gap F          Min strong−weak gap (default: 0.20)
  --ollama-model STR   Ollama model name (default: gemma4:31b)
  --ollama-url URL     Ollama base URL (default: http://localhost:11434)
  --device             {auto,cuda,cpu} (default: auto)
  --output-dir DIR     Output directory (default: output_/)
  --log-level          {DEBUG,INFO,WARNING,ERROR}
```

---

## 9. Expected Output

### Terminal output (during generation)

```
══════════════════════════════════════════════════════════════════════════
  AutoData — Agentic Self-Instruct   |  Machine Translation
══════════════════════════════════════════════════════════════════════════
  Language pair    : EN → DE
  Target pairs     : 20
  Challenger/Judge : gemma4:31b  (Ollama)
  Weak solver      : Helsinki-NLP/opus-mt-en-de
  Strong solver    : facebook/nllb-200-distilled-1.3B
  Acceptance gate  : weak ≤ 0.55  |  strong ≥ 0.68  |  gap ≥ 0.20
══════════════════════════════════════════════════════════════════════════

── Attempt 1 ──────────────────────────────────────────────────────────
[#1 iter 1/6] Calling Challenger…
[#1 iter 1/6] challenge_type='idiom' | src='She finally kicked the bucket...'
[#1 iter 1/6] weak_avg=0.112  strong_avg=0.834  gap=0.722  ACCEPTED ✓

──────────────────────────────────────────────────────────────────────────
  Sample #1   |  Type: idiom                      |  Rounds: 1
──────────────────────────────────────────────────────────────────────────
  EN     : She finally kicked the bucket last Tuesday.
  REF    : Sie hat endlich letzten Dienstag den Löffel abgegeben.
  WEAK  (German) : Sie hat endlich den Eimer letzten Dienstag getreten.
          score = 0.112  ← Idiom rendered literally; wrong meaning
  STRONG (German) : Sie hat letzten Dienstag endlich den Löffel abgegeben.
          score = 0.834  ← Correctly uses the German idiom equivalent
  Gap    : 0.722
  Why?   : The weak model has no idiom knowledge and translates…
```

### Saved JSON output

`output_/en-de_autodata_.json`:

```json
[
  {
    "source_en": "She finally kicked the bucket last Tuesday.",
    "reference_de": "Sie hat endlich letzten Dienstag den Löffel abgegeben.",
    "weak_mt_de": "Sie hat endlich den Eimer letzten Dienstag getreten.",
    "strong_mt_de": "Sie hat letzten Dienstag endlich den Löffel abgegeben.",
    "weak_score": 0.112,
    "strong_score": 0.834,
    "gap": 0.722,
    "challenge_type": "idiom",
    "challenge_explanation": "The weak model has no idiom knowledge and translates 'kick the bucket' literally as 'den Eimer treten', producing nonsensical German.",
    "rubric": [
      {"criterion": "Idiom 'kick the bucket' correctly rendered as 'den Löffel abgeben'", "weight": 5},
      {"criterion": "Correct past tense (Perfekt) construction", "weight": 3},
      {"criterion": "Correct temporal phrase 'letzten Dienstag' placement", "weight": 2}
    ],
    "weak_per_criterion": [0.0, 0.5, 0.3],
    "strong_per_criterion": [1.0, 0.9, 0.8],
    "rounds_taken": 1
  },
  …
]
```

### Final summary

```
══════════════════════════════════════════════════════════════════════════
  FINAL SUMMARY
══════════════════════════════════════════════════════════════════════════
  Accepted samples     : 20
  Total wall time      : 843.2s  (42.2s/sample)

  Weak  avg score      : 0.201 ± 0.089
  Strong avg score     : 0.791 ± 0.062
  Gap (strong − weak)  : 0.590 ± 0.114   [min=0.213, max=0.831]
  Avg rounds to accept : 2.4

  Challenge type distribution:
    idiom                        ████████████         8  (40%)
    morphological                ████                 4  (20%)
    syntactic_inversion          ██                   2  (10%)
    tense_aspect                 ██                   2  (10%)
    technical_domain             █                    1   (5%)
    register_formality           █                    1   (5%)
    lexical_ambiguity            █                    1   (5%)
    proverb_equivalent           █                    1   (5%)
```

---

## 10. Code Structure

```
autodata_mt/
├── config.py        Configuration dataclasses
│                      AcceptanceCriteria, OllamaConfig, MTConfig, Config
│                      NLLB_LANG_CODES, WEAK_MODELS, STRONG_MODELS
│
├── utils.py         Shared utilities
│                      call_ollama()          — Ollama REST POST /api/chat
│                      extract_json_object()  — Robust JSON from messy LLM output
│                        1. Markdown fence scan
│                        2. Balanced-brace scanner (respects string literals)
│                        3. Direct json.loads fallback
│                        4. Trailing-comma repair
│                      compute_weighted_score() — normalised rubric aggregate
│
├── challenger.py    Challenger agent
│                      CHALLENGE_TYPES        — 10 MT-specific challenge descriptors
│                      Challenger.generate()  — prompts Gemma, parses ChallengeOutput
│                      ChallengeOutput        — source, reference, challenge_type, rubric
│
├── mt_solvers.py    Translation model wrappers
│                      WeakMTSolver           — Helsinki-NLP MarianMT (74M)
│                      StrongMTSolver         — Meta NLLB-200-distilled-1.3B
│                      load_solvers()         — convenience factory
│
├── judge.py         Judge agent
│                      Judge.score()          — prompts Gemma, returns TranslationScore
│                      TranslationScore       — per_criterion, weighted_score, comment
│                      _normalise_scores()    — clamp ±0.25 tolerance, fail-closed
│
├── acceptance.py    Acceptance gate
│                      check_acceptance()     — four-condition discriminator gate
│                      AcceptanceResult       — accepted, weak_avg, strong_avg, gap
│                      format_rejection_feedback() — structured guidance for Challenger
│
├── inner_loop.py    AutoData 8-step inner loop
│                      run_inner_loop()       — orchestrates Steps 1–8
│                      AcceptedSample         — challenge + scores + acceptance result
│
└── .py           CLI entry point
                       main()                 — arg parsing, model loading, generation loop
                       _print_accepted()      — per-sample console output
                       _print_summary()       — aggregate statistics + histogram
                       _save()                — incremental JSON save
```

---

## 11. Design Decisions and Trade-offs

### Why Gemma 4:31b as both Challenger and Judge?

The blog uses separate frontier models for different roles (Kimi-K2.6 for
Challenger/Judge, smaller local model for Weak Solver). Since only `gemma4:31b`
is available locally via Ollama, it doubles as both. This introduces a
**role-conflict risk** — the same model proposes and evaluates — but is
acceptable for a . In production, use separate models or a smaller model
for judging.

### Why NLLB-200-distilled-1.3B as the strong solver?

NLLB-200 was chosen over a frontier LLM (e.g., calling Gemma for translation)
because:
1. It is a purpose-built MT model, making the weak/strong gap reflect **MT
   architecture capability** rather than general LLM reasoning
2. The 18× parameter gap over opus-mt-en-de provides a consistent capability
   split across all challenge types
3. It is deterministic (beam search), avoiding LLM stochasticity in translations

### LLM-as-Judge vs. Automatic Metrics

AutoData uses an LLM judge because BLEU/chrF do not capture idiom correctness,
register appropriateness, or semantic accuracy against a rubric. The  follows
this design. Automatic metrics can be added as supplementary signals by
uncommenting `sacrebleu` in `requirements_.txt`.

### JSON robustness

LLM outputs are often messy (fences, trailing commas, prose before JSON). The
`extract_json_object()` function uses a balanced-brace scanner that respects
string literals, preventing a `}` inside a quoted value from terminating the
parse prematurely — identical in design to `llm/client.py::extract_json` in the
Lauorie/Autodata reference implementation.

### Judge score clamping (fail-closed)

Judge per-criterion scores are clamped to [0, 1] with a ±0.25 tolerance.
Values beyond ±0.25 (e.g., a confused judge emitting `2.5`) return 0.0
rather than being normalised — the same fail-closed strategy used in
`pipeline/evaluate_rubric.py::_normalise_judge_scores` in the reference repo.
This prevents a misbehaving judge from pushing a poor translation over the
acceptance threshold.

### Sequential execution

Unlike the reference repo's nested `ThreadPoolExecutor` concurrency, the 
runs sequentially. This is intentional: both HuggingFace models share GPU with
Ollama, and concurrent inference could cause OOM. For throughput, enable
`--device cpu` for the HuggingFace models and run Ollama with GPU.

---

## 12. What We Changed vs. the Blog

### New / different

| Area | Blog | This  |
|---|---|---|
| **Task** | CS-research QA from paper passages | Machine Translation (EN→DE / EN→ES) |
| **Challenger output** | Question + reference answer + rubric | Hard source sentence + reference translation + rubric |
| **Weak solver** | Qwen3.5-4B (general LLM) | `Helsinki-NLP/opus-mt-en-de` — MarianMT, 74 M params |
| **Strong solver** | Qwen3.5-397B-A17B (general LLM) | `facebook/nllb-200-distilled-1.3B` — NLLB-200 MT model |
| **Models** | Kimi-K2.6 via cloud API | **Gemma 4:31b fully local via Ollama** — no API key |
| **Challenge taxonomy** | Single task type | 10 typed MT challenges targeting specific MarianMT weaknesses (`idiom`, `morphological`, `syntactic_inversion`, `lexical_ambiguity`, `negation_scope`, `cultural_reference`, `register_formality`, `tense_aspect`, `technical_domain`, `proverb_equivalent`) |
| **QV axis: context leakage** | Question must not quote the passage | **Challenge_Valid** — sentence must actually exhibit the claimed challenge type |
| **QV axis: answerability** | Answer grounded in passage | **Reference_Quality** — reference must be idiomatic, not word-for-word literal |
| **QV axis (new)** | — | **Difficulty_Fit** — challenge must specifically exploit a known MarianMT weakness, not just be generically hard |
| **Rejection feedback** | Main agent re-prompts from judge output | **`Judge.generate_feedback()`** — Gemma gets the full per-criterion score breakdown for both models and writes sentence-specific diagnosis; template fallback on failure |
| **Deduplication** | Not described | `seen_sources` set shared across all attempts; injected into Challenger prompt as negative examples |
| **Diversity tracking** | Not described | `accepted_types` list; Challenger told which types already succeeded to prevent overrepresentation |
| **Weak score floor** | None | `weak_avg ≥ 0.05` — rejects untranslatable noise sentences |
| **Weak score ceiling** | `weak_avg ≤ 0.65` | **`weak_avg ≤ 0.55`** — tighter; MT models are more deterministic than LLM solvers |

### Not implemented (out of scope for a )

| Blog feature | Reason skipped |
|---|---|
| Outer meta-optimization loop (harness evolution, 12.8% → 42.4%) | Requires multi-generation AST-level code mutation |
| GRPO / SFT fine-tuning of the weak model |  stops at data generation |
| 10,000+ source documents (S2ORC corpus) | MT task generates source sentences from scratch, no corpus needed |
| Boltzmann population sampling | Tied to the outer loop |
