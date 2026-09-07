"""
MT solver wrappers — weak and strong translation models.

Weak  : Helsinki-NLP/opus-mt-en-{de,es}   (~74 M params, MarianMT)
Strong: facebook/nllb-200-distilled-1.3B  (~1.3 B params, NLLB-200)

Both are loaded lazily via transformers and run on the chosen device.
CPU is perfectly fine for a PoC; set --device cuda for faster inference
if sufficient VRAM is available alongside Ollama's Gemma instance.
"""

import logging
from typing import Optional

import torch

from .config import NLLB_LANG_CODES, MTConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _resolve_device(preferred: str) -> str:
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preferred


def _param_count(model) -> str:
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    return f"{n / 1e6:.0f}M"


# ---------------------------------------------------------------------------
# Weak MT solver  (Helsinki-NLP MarianMT)
# ---------------------------------------------------------------------------

class WeakMTSolver:
    """
    Small MarianMT model (~74 M params).

    Strengths : fast, handles simple/literal sentences well.
    Weaknesses: idioms, morphology, complex syntax, rare vocabulary,
                register distinctions (du/Sie), domain terms.
    """

    def __init__(self, model_id: str, device: str = "auto") -> None:
        from transformers import MarianMTModel, MarianTokenizer

        self.device = _resolve_device(device)
        self.model_id = model_id
        logger.info(f"[WeakMT] Loading '{model_id}' on {self.device} …")

        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        logger.info(f"[WeakMT] Loaded — {_param_count(self.model)} params")

    @torch.inference_mode()
    def translate(self, text: str, num_beams: int = 4, max_new_tokens: int = 256) -> str:
        """Translate *text* and return the decoded target string."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Strong MT solver  (Meta NLLB-200-distilled-1.3B)
# ---------------------------------------------------------------------------

class StrongMTSolver:
    """
    Meta's NLLB-200-distilled-1.3B (~1.3 B params).

    Covers 200 languages; handles idioms, complex syntax, domain terms,
    and register distinctions significantly better than MarianMT.
    """

    def __init__(
        self,
        model_id: str,
        src_lang: str,
        tgt_lang: str,
        device: str = "auto",
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = _resolve_device(device)
        self.model_id = model_id
        self.src_code = NLLB_LANG_CODES[src_lang]
        self.tgt_code = NLLB_LANG_CODES[tgt_lang]

        logger.info(f"[StrongMT] Loading '{model_id}' on {self.device} …")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, src_lang=self.src_code
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

        # Cache the forced BOS token id for the target language
        self._tgt_lang_id: int = self.tokenizer.convert_tokens_to_ids(self.tgt_code)
        if self._tgt_lang_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"NLLB tokenizer does not recognise language code '{self.tgt_code}'. "
                "Check NLLB_LANG_CODES in config.py."
            )

        logger.info(
            f"[StrongMT] Loaded — {_param_count(self.model)} params | "
            f"src={self.src_code} → tgt={self.tgt_code} (id={self._tgt_lang_id})"
        )

    @torch.inference_mode()
    def translate(self, text: str, num_beams: int = 4, max_new_tokens: int = 256) -> str:
        """Translate *text* and return the decoded target string."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            forced_bos_token_id=self._tgt_lang_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def load_solvers(cfg: MTConfig) -> tuple[WeakMTSolver, StrongMTSolver]:
    """Load both solvers from a MTConfig and return (weak, strong)."""
    weak = WeakMTSolver(model_id=cfg.weak_model_id, device=cfg.device)
    strong = StrongMTSolver(
        model_id=cfg.strong_model_id,
        src_lang=cfg.src_lang,
        tgt_lang=cfg.tgt_lang,
        device=cfg.device,
    )
    return weak, strong
