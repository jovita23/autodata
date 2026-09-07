"""Shared utilities: Ollama API client, robust JSON extraction, scoring helpers."""

import json
import logging
import re
import sys
from statistics import mean
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama REST client
# ---------------------------------------------------------------------------

def call_ollama(
    messages: list[dict],
    model: str,
    temperature: float,
    base_url: str,
    timeout_s: int,
) -> str:
    """POST to Ollama /api/chat and return the assistant content string."""
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout_s}s")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. "
            "Is `ollama serve` running?"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}") from e


# ---------------------------------------------------------------------------
# Robust JSON extraction
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> Optional[dict]:
    """
    Robustly extract a JSON *object* from messy LLM output.

    Strategy (in order):
    1. Fenced ```json ... ``` block.
    2. Balanced-brace scanner that respects string literals so that
       a stray `}` inside a quoted value does not terminate too early.
    3. Direct json.loads of the whole stripped text.
    4. Trailing-comma repair + retry.
    """
    text = text.strip()

    # ── 1. Markdown fence ──────────────────────────────────────────────────
    fence = re.search(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # ── 2. Balanced-brace scan ─────────────────────────────────────────────
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Try trailing-comma repair
                            repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                            try:
                                return json.loads(repaired)
                            except json.JSONDecodeError:
                                break  # best candidate found, but unparseable

    # ── 3. Direct parse ─────────────────────────────────────────────────────
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Weighted scoring
# ---------------------------------------------------------------------------

def compute_weighted_score(per_criterion: list[float], weights: list[int]) -> float:
    """Return a weight-normalised score in [0, 1]."""
    if not per_criterion or not weights or len(per_criterion) != len(weights):
        return 0.0
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    raw = sum(s * w for s, w in zip(per_criterion, weights)) / total_w
    return max(0.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
