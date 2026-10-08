"""Text normalization and OCR metrics.

Every metric works on the same normalized form of both texts:
  * NFC Unicode, zero-width characters removed, `char_map` applied (typographic unification)
  * paragraphs = blocks separated by blank lines (or by single newlines when the text has no blank lines)
  * whitespace inside a paragraph collapsed to one space, so handwritten line wraps never count as errors

Metrics:
  cer          character error rate on the flat text (paragraphs joined by a space) - main accuracy number
  cer_lenient  CER ignoring case, punctuation and `lenient_char_map` variants (e.g. ё/е) - pure letter reading
  wer          word error rate, punctuation ignored, case-sensitive
  punctuation  precision / recall / F1 per mark, from a character alignment of reference and output
  paragraphs   precision / recall / F1 of paragraph breaks, from the same alignment
  char classes recall of letters / digits / punctuation / spaces
  case / diacritic errors  substitutions that differ only by case (a/A) or only by a diacritic (е/ё, e/é)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz.distance import Levenshtein

DEFAULT_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-",
    "–": "—", "―": "—",
    "…": "...",
}
DEFAULT_LENIENT_MAP = {"ё": "е", "Ё": "Е"}

CHAR_CLASSES = ("letters", "digits", "punctuation", "spaces", "other")
PARA = "\n"  # paragraph separator in the "structured" text

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
WORD_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")


@dataclass
class NormConfig:
    char_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CHAR_MAP))
    lenient_char_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LENIENT_MAP))
    strip_markdown: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> NormConfig:
        d = d or {}
        cfg = cls()
        if "char_map" in d:
            cfg.char_map = dict(d["char_map"] or {})
        if "lenient_char_map" in d:
            cfg.lenient_char_map = dict(d["lenient_char_map"] or {})
        if "strip_markdown" in d:
            cfg.strip_markdown = bool(d["strip_markdown"])
        return cfg


def is_punct(c: str) -> bool:
    return unicodedata.category(c).startswith("P")


def char_class(c: str) -> str:
    if c == " ":
        return "spaces"
    cat = unicodedata.category(c)[0]
    if cat in "LM":
        return "letters"
    if cat == "N":
        return "digits"
    if cat == "P":
        return "punctuation"
    return "other"


def _base_letter(c: str) -> str:
    return unicodedata.normalize("NFD", c)[:1].casefold()


def strip_markdown(text: str) -> str:
    text = text.strip()
    fence = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", text, re.S)
    if fence:
        text = fence.group(1)
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    text = re.sub(r"__(.+?)__", r"\1", text, flags=re.S)
    return text


def clean(text: str, cfg: NormConfig, *, is_output: bool) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n").translate(_ZERO_WIDTH)
    if is_output and cfg.strip_markdown:
        text = strip_markdown(text)
    for src, dst in cfg.char_map.items():
        text = text.replace(src, dst)
    return text


def split_paragraphs(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if re.search(r"\n[^\S\n]*\n", text):
        blocks = re.split(r"\n\s*\n", text)
    else:  # no blank lines: treat every line as a paragraph
        blocks = text.split("\n")
    paragraphs = (re.sub(r"\s+", " ", b).strip() for b in blocks)
    return [p for p in paragraphs if p]


def lenient_form(flat: str, cfg: NormConfig) -> str:
    for src, dst in cfg.lenient_char_map.items():
        flat = flat.replace(src, dst)
    flat = "".join(c for c in flat.casefold() if not is_punct(c))
    return re.sub(r"\s+", " ", flat).strip()


@dataclass
class Normalized:
    paragraphs: list[str]

    @property
    def structured(self) -> str:
        return PARA.join(self.paragraphs)

    @property
    def flat(self) -> str:
        return " ".join(self.paragraphs)


def normalize(text: str, cfg: NormConfig, *, is_output: bool) -> Normalized:
    return Normalized(split_paragraphs(clean(text, cfg, is_output=is_output)))


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    if p is None and r is None:
        f1 = None
    elif p is None or r is None:
        f1 = 0.0
    else:
        f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_pair(reference: str, output: str, cfg: NormConfig) -> dict[str, Any]:
    ref = normalize(reference, cfg, is_output=False)
    hyp = normalize(output, cfg, is_output=True)
    ref_f, hyp_f = ref.flat, hyp.flat

    # --- characters (flat text) ---
    sub = dele = ins = case_err = diac_err = 0
    char_conf: Counter[tuple[str, str]] = Counter()
    for op in Levenshtein.editops(ref_f, hyp_f):
        if op.tag == "replace":
            a, b = ref_f[op.src_pos], hyp_f[op.dest_pos]
            sub += 1
            if a.casefold() == b.casefold():
                case_err += 1
            elif _base_letter(a) == _base_letter(b):
                diac_err += 1
            char_conf[(a, b)] += 1
        elif op.tag == "delete":
            dele += 1
            char_conf[(ref_f[op.src_pos], "")] += 1
        else:
            ins += 1
            char_conf[("", hyp_f[op.dest_pos])] += 1

    ref_l, hyp_l = lenient_form(ref_f, cfg), lenient_form(hyp_f, cfg)
    lenient_edits = Levenshtein.distance(ref_l, hyp_l)

    # --- words (punctuation ignored) ---
    ref_w, hyp_w = WORD_RE.findall(ref_f), WORD_RE.findall(hyp_f)
    w_sub = w_del = w_ins = 0
    for op in Levenshtein.editops(ref_w, hyp_w):
        if op.tag == "replace":
            w_sub += 1
        elif op.tag == "delete":
            w_del += 1
        else:
            w_ins += 1
    word_conf: Counter[tuple[str, str]] = Counter()
    for op in Levenshtein.opcodes(ref_w, hyp_w):
        if op.tag != "equal":
            word_conf[(" ".join(ref_w[op.src_start:op.src_end]), " ".join(hyp_w[op.dest_start:op.dest_end]))] += 1

    # --- char classes (flat text alignment, so layout never counts) ---
    class_total: Counter[str] = Counter(char_class(c) for c in ref_f)
    class_hit: Counter[str] = Counter()
    for op in Levenshtein.opcodes(ref_f, hyp_f):
        if op.tag == "equal":
            class_hit.update(char_class(c) for c in ref_f[op.src_start:op.src_end])

    # --- punctuation and paragraph breaks (structured text alignment) ---
    ref_s, hyp_s = ref.structured, hyp.structured
    punct: dict[str, list[int]] = {}  # mark -> [tp, fp, fn]
    para = [0, 0, 0]

    def bucket(c: str) -> list[int] | None:
        if c == PARA:
            return para
        if is_punct(c):
            return punct.setdefault(c, [0, 0, 0])
        return None

    for op in Levenshtein.opcodes(ref_s, hyp_s):
        a, b = ref_s[op.src_start:op.src_end], hyp_s[op.dest_start:op.dest_end]
        if op.tag == "equal":
            for c in a:
                if (bk := bucket(c)) is not None:
                    bk[0] += 1
        else:
            for c in a:
                if (bk := bucket(c)) is not None:
                    bk[2] += 1
            for c in b:
                if (bk := bucket(c)) is not None:
                    bk[1] += 1

    char_edits = sub + dele + ins
    word_edits = w_sub + w_del + w_ins
    p_tp, p_fp, p_fn = (sum(v[i] for v in punct.values()) for i in range(3))
    return {
        "ref_chars": len(ref_f),
        "hyp_chars": len(hyp_f),
        "char_edits": char_edits, "char_sub": sub, "char_del": dele, "char_ins": ins,
        "cer": char_edits / max(len(ref_f), 1),
        "lenient_ref_chars": len(ref_l),
        "lenient_edits": lenient_edits,
        "cer_lenient": lenient_edits / max(len(ref_l), 1),
        "ref_words": len(ref_w),
        "hyp_words": len(hyp_w),
        "word_edits": word_edits, "word_sub": w_sub, "word_del": w_del, "word_ins": w_ins,
        "wer": word_edits / max(len(ref_w), 1),
        "case_errors": case_err,
        "diacritic_errors": diac_err,
        "punct_by_mark": {m: {"tp": v[0], "fp": v[1], "fn": v[2]} for m, v in sorted(punct.items())},
        "punct_tp": p_tp, "punct_fp": p_fp, "punct_fn": p_fn,
        "punct": _prf(p_tp, p_fp, p_fn),
        "para_ref": len(ref.paragraphs),
        "para_hyp": len(hyp.paragraphs),
        "para_tp": para[0], "para_fp": para[1], "para_fn": para[2],
        "para": _prf(*para),
        "class_total": dict(class_total),
        "class_hit": dict(class_hit),
        "char_confusions": [[a, b, n] for (a, b), n in char_conf.most_common()],
        "word_confusions": [[a, b, n] for (a, b), n in word_conf.most_common()],
    }


def aggregate(items: list[dict[str, Any]], top_n: int = 15) -> dict[str, Any]:
    """Micro-averaged metrics over many samples (sum of edits / sum of reference length)."""
    if not items:
        return {"n": 0}

    def total(key: str) -> int:
        return sum(m[key] for m in items)

    ref_chars, ref_words, len_ref = total("ref_chars"), total("ref_words"), total("lenient_ref_chars")
    marks: dict[str, list[int]] = {}
    class_total: Counter[str] = Counter()
    class_hit: Counter[str] = Counter()
    char_conf: Counter[tuple[str, str]] = Counter()
    word_conf: Counter[tuple[str, str]] = Counter()
    for m in items:
        for mark, v in m["punct_by_mark"].items():
            acc = marks.setdefault(mark, [0, 0, 0])
            acc[0] += v["tp"]; acc[1] += v["fp"]; acc[2] += v["fn"]
        class_total.update(m["class_total"])
        class_hit.update(m["class_hit"])
        char_conf.update({(a, b): n for a, b, n in m["char_confusions"]})
        word_conf.update({(a, b): n for a, b, n in m["word_confusions"]})

    cers = sorted(m["cer"] for m in items)
    return {
        "n": len(items),
        "ref_chars": ref_chars,
        "char_edits": total("char_edits"),
        "char_sub": total("char_sub"), "char_del": total("char_del"), "char_ins": total("char_ins"),
        "cer": total("char_edits") / max(ref_chars, 1),
        "cer_mean": sum(cers) / len(cers),
        "cer_median": cers[len(cers) // 2] if len(cers) % 2 else (cers[len(cers) // 2 - 1] + cers[len(cers) // 2]) / 2,
        "cer_worst": cers[-1],
        "cer_lenient": total("lenient_edits") / max(len_ref, 1),
        "wer": total("word_edits") / max(ref_words, 1),
        "case_errors": total("case_errors"),
        "diacritic_errors": total("diacritic_errors"),
        "punct": _prf(total("punct_tp"), total("punct_fp"), total("punct_fn")),
        "punct_by_mark": {mk: {"tp": v[0], "fp": v[1], "fn": v[2], **_prf(*v)} for mk, v in sorted(marks.items())},
        "para": _prf(total("para_tp"), total("para_fp"), total("para_fn")),
        "para_ref": total("para_ref"),
        "para_hyp": total("para_hyp"),
        "para_tp": total("para_tp"), "para_fp": total("para_fp"), "para_fn": total("para_fn"),
        "class_recall": {c: (class_hit[c] / class_total[c]) if class_total[c] else None for c in CHAR_CLASSES},
        "class_total": dict(class_total),
        "char_confusions": [[a, b, n] for (a, b), n in char_conf.most_common(top_n)],
        "word_confusions": [[a, b, n] for (a, b), n in word_conf.most_common(top_n)],
    }
