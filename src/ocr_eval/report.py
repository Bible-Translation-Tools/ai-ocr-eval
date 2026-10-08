"""Self-contained HTML report."""

from __future__ import annotations

import base64
import html
import json
import os
import re
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from .dataset import Sample
from .metrics import CHAR_CLASSES, PARA, NormConfig, normalize

# CER bins for the heatmap (upper bounds), each mapped to a step of a single-hue ramp.
CER_BINS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, float("inf")]
CER_BIN_LABELS = ["<1%", "1–2%", "2–5%", "5–10%", "10–20%", "20–40%", "≥40%"]

_DIFF_TOKEN = re.compile(r"\n|[^\S\n]+|\w+|[^\w\s]")


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def num(x: float | None, digits: int = 0) -> str:
    return "—" if x is None else f"{x:,.{digits}f}"


def cer_bin(cer: float) -> int:
    return next(i for i, upper in enumerate(CER_BINS) if cer < upper)


# ---------------------------------------------------------------- diff

def _diff_text(s: str, kind: str) -> str:
    if kind != "eq" and s.strip(" ") == "":
        s = s.replace(" ", "␣")
    return esc(s).replace(PARA, '<span class="pb" title="paragraph break">¶</span><br>')


def render_diff(ref: str, hyp: str, level: str) -> str:
    a = _DIFF_TOKEN.findall(ref) if level == "word" else list(ref)
    b = _DIFF_TOKEN.findall(hyp) if level == "word" else list(hyp)
    parts: list[str] = []
    for op in Levenshtein.opcodes(a, b):
        old = "".join(a[op.src_start:op.src_end])
        new = "".join(b[op.dest_start:op.dest_end])
        if op.tag == "equal":
            parts.append(_diff_text(old, "eq"))
            continue
        if old:
            parts.append(f"<del>{_diff_text(old, 'del')}</del>")
        if new:
            parts.append(f"<ins>{_diff_text(new, 'ins')}</ins>")
    return "".join(parts)


# ---------------------------------------------------------------- page pieces

CSS = """
:root{
  color-scheme:light;
  --page:#f9f9f7;--surface:#fcfcfb;--surface-2:#f0efec;
  --ink:#0b0b0b;--ink-2:#52514e;--muted:#6f6e69;
  --grid:#e1e0d9;--border:rgba(11,11,11,.10);
  --accent:#2a78d6;--bar:#2a78d6;
  --del-bg:#fde2e1;--del-ink:#8a1c1c;--ins-bg:#d9f2d9;--ins-ink:#0a5a0a;
  --h0:#cde2fb;--h1:#9ec5f4;--h2:#6da7ec;--h3:#3987e5;--h4:#256abf;--h5:#184f95;--h6:#0d366b;
  --h0-ink:#0b0b0b;--h1-ink:#0b0b0b;--h2-ink:#0b0b0b;--h3-ink:#fff;--h4-ink:#fff;--h5-ink:#fff;--h6-ink:#fff;
  --warn:#8a5a00;--warn-bg:#fdf1d6;--err:#a32020;--err-bg:#fbe3e3;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --page:#0d0d0d;--surface:#1a1a19;--surface-2:#262624;
    --ink:#fff;--ink-2:#c3c2b7;--muted:#9a9890;
    --grid:#2c2c2a;--border:rgba(255,255,255,.10);
    --accent:#6da7ec;--bar:#3987e5;
    --del-bg:#4a1f1f;--del-ink:#ffb8b8;--ins-bg:#173d1c;--ins-ink:#a8eda8;
    --h0:#0d366b;--h1:#104281;--h2:#184f95;--h3:#256abf;--h4:#3987e5;--h5:#6da7ec;--h6:#9ec5f4;
    --h0-ink:#fff;--h1-ink:#fff;--h2-ink:#fff;--h3-ink:#fff;--h4-ink:#fff;--h5-ink:#0b0b0b;--h6-ink:#0b0b0b;
    --warn:#f3c86b;--warn-bg:#3a2e12;--err:#ff9d9d;--err-bg:#3d1717;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d;--surface:#1a1a19;--surface-2:#262624;
  --ink:#fff;--ink-2:#c3c2b7;--muted:#9a9890;
  --grid:#2c2c2a;--border:rgba(255,255,255,.10);
  --accent:#6da7ec;--bar:#3987e5;
  --del-bg:#4a1f1f;--del-ink:#ffb8b8;--ins-bg:#173d1c;--ins-ink:#a8eda8;
  --h0:#0d366b;--h1:#104281;--h2:#184f95;--h3:#256abf;--h4:#3987e5;--h5:#6da7ec;--h6:#9ec5f4;
  --h0-ink:#fff;--h1-ink:#fff;--h2-ink:#fff;--h3-ink:#fff;--h4-ink:#fff;--h5-ink:#0b0b0b;--h6-ink:#0b0b0b;
  --warn:#f3c86b;--warn-bg:#3a2e12;--err:#ff9d9d;--err-bg:#3d1717;
}
.t-dark{display:none}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .t-dark{display:inline}:root:not([data-theme="light"]) .t-light{display:none}}
:root[data-theme="dark"] .t-dark{display:inline}
:root[data-theme="dark"] .t-light{display:none}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent)}
header.top{position:sticky;top:0;z-index:5;background:var(--page);border-bottom:1px solid var(--grid)}
.bar-inner{max-width:1280px;margin:0 auto;padding:10px 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.bar-inner strong{font-size:16px}
nav{display:flex;gap:14px;flex-wrap:wrap;font-size:14px}
nav a{color:var(--ink-2);text-decoration:none}
nav a:hover{color:var(--ink)}
.spacer{flex:1}
button,.seg label{font:inherit;font-size:13px;color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:4px 10px;cursor:pointer}
.seg{display:inline-flex;gap:0}
.seg input{position:absolute;opacity:0;pointer-events:none}
.seg label{border-radius:0}
.seg label:first-of-type{border-radius:6px 0 0 6px}
.seg label:last-of-type{border-radius:0 6px 6px 0;border-left:0}
.seg input:checked+label{background:var(--ink);color:var(--page)}
.seg input:focus-visible+label{outline:2px solid var(--accent);outline-offset:1px}
main{max-width:1280px;margin:0 auto;padding:8px 16px 64px}
h1{font-size:26px;margin:24px 0 4px}
h2{font-size:20px;margin:40px 0 4px}
h3{font-size:16px;margin:0}
.sub{color:var(--ink-2);margin:0 0 16px;overflow-wrap:anywhere}
.muted{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-size:12px;font-weight:600;color:var(--ink-2);vertical-align:bottom}
th[title]{text-decoration:underline dotted;text-underline-offset:3px;cursor:help}
td:first-child,th:first-child{text-align:left}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:var(--surface-2)}
.best{font-weight:700}
.model-name{font-weight:600}
.model-sub{display:block;font-size:12px;color:var(--muted);font-weight:400}
.meter{display:flex;align-items:center;gap:8px;justify-content:flex-end}
.meter .track{width:90px;height:8px;border-radius:4px;background:var(--surface-2);overflow:hidden}
.meter .fill{height:100%;background:var(--bar);border-radius:0 4px 4px 0}
.small{font-size:12px;color:var(--muted)}
.heat td.h{text-align:center;min-width:76px;padding:0;border:2px solid var(--surface)}
.heat td.h a{display:block;padding:8px 6px;text-decoration:none;color:inherit}
.heat td.h a:hover{outline:2px solid var(--ink);outline-offset:-2px}
.heat td.na{background:var(--surface-2);color:var(--muted)}
.heat th{text-align:center}
.heat th:first-child{text-align:left}
.legend{display:flex;gap:4px;align-items:center;flex-wrap:wrap;font-size:12px;color:var(--ink-2);margin:10px 0 0}
.legend span.sw{display:inline-block;width:28px;height:12px;border-radius:2px;vertical-align:middle}
""" + "".join(
    f".b{i}{{background:var(--h{i});color:var(--h{i}-ink)}}" for i in range(7)
) + """
.explain{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(240px,100%),1fr));gap:8px 24px;padding:14px 16px;margin-top:12px;font-size:13px;color:var(--ink-2)}
.explain b{color:var(--ink)}
details>summary{cursor:pointer;color:var(--ink-2);font-size:14px}
details[open]>summary{margin-bottom:8px}
.conf{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(300px,100%),1fr));gap:12px}
.conf .card{padding:12px 14px}
.conf table td,.conf table th{padding:3px 6px;font-size:13px}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}
.chip-c{background:var(--surface-2);border-radius:4px;padding:0 4px}
.sample{display:grid;grid-template-columns:300px minmax(0,1fr);gap:20px;margin-top:16px;padding-top:24px;border-top:1px solid var(--grid)}
.sample .side{position:sticky;top:64px;align-self:start}
.sample .side img{width:100%;border-radius:8px;border:1px solid var(--border);cursor:zoom-in;display:block;background:#fff}
.sample h3{font-size:18px;margin-bottom:8px;word-break:break-all}
.ref{white-space:pre-wrap;font-size:14px;padding:10px 12px;max-height:320px;overflow:auto}
.pred{padding:14px 16px;margin-bottom:12px}
.pred-head{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:baseline;margin-bottom:8px}
.stats{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.stats b{color:var(--ink);font-weight:600}
.badge{font-size:12px;border-radius:4px;padding:1px 6px;font-weight:600}
.badge.cer{border:1px solid var(--border)}
.badge.warn{background:var(--warn-bg);color:var(--warn)}
.badge.err{background:var(--err-bg);color:var(--err)}
.diff{font-size:15px;line-height:1.75;padding:10px 12px;background:var(--page);border-radius:6px;border:1px solid var(--grid);overflow-wrap:anywhere}
del{background:var(--del-bg);color:var(--del-ink);text-decoration:line-through;text-decoration-thickness:1px;border-radius:2px}
ins{background:var(--ins-bg);color:var(--ins-ink);text-decoration:underline;text-decoration-thickness:1px;text-underline-offset:3px;border-radius:2px}
.pb{color:var(--muted);font-size:12px}
del .pb,ins .pb{color:inherit}
body.diff-char .dw{display:none}
body:not(.diff-char) .dc{display:none}
pre.raw{white-space:pre-wrap;font-size:13px;padding:10px 12px;background:var(--page);border:1px solid var(--grid);border-radius:6px;max-height:420px;overflow:auto}
.errbox{background:var(--err-bg);color:var(--err);padding:10px 12px;border-radius:6px;font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}
dialog{border:0;padding:0;background:transparent;max-width:96vw;max-height:96vh}
dialog::backdrop{background:rgba(0,0,0,.75)}
dialog img{max-width:96vw;max-height:96vh;display:block;border-radius:6px;background:#fff}
pre.prompt{white-space:pre-wrap;font-size:13px;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:8px}
@media (max-width:820px){
  .sample{grid-template-columns:minmax(0,1fr)}
  .bar-inner{gap:8px}
  .sample .side{position:static}
  .sample .side img{max-height:360px;object-fit:contain}
  nav{display:none}
}
"""

JS = """
(function(){
  var root=document.documentElement, KEY='ocr-eval-theme', DKEY='ocr-eval-diff';
  function get(k){try{return localStorage.getItem(k)}catch(e){return null}}
  function set(k,v){try{localStorage.setItem(k,v)}catch(e){}}
  var t=get(KEY); if(t) root.setAttribute('data-theme',t);
  var btn=document.getElementById('theme');
  function label(){var v=root.getAttribute('data-theme')||'auto';btn.textContent='Theme: '+v;}
  label();
  btn.addEventListener('click',function(){
    var order=['auto','light','dark'],cur=root.getAttribute('data-theme')||'auto';
    var next=order[(order.indexOf(cur)+1)%3];
    if(next==='auto'){root.removeAttribute('data-theme');set(KEY,'')}else{root.setAttribute('data-theme',next);set(KEY,next)}
    label();
  });
  function setDiff(v){document.body.classList.toggle('diff-char',v==='char');var r=document.getElementById('d-'+v);if(r)r.checked=true;set(DKEY,v);}
  setDiff(get(DKEY)||'word');
  document.querySelectorAll('input[name=diff]').forEach(function(r){r.addEventListener('change',function(){setDiff(r.value)})});
  var dlg=document.getElementById('viewer'), big=document.getElementById('viewer-img');
  document.querySelectorAll('img.thumb').forEach(function(img){
    img.addEventListener('click',function(){big.src=img.src;big.alt=img.alt;dlg.showModal();});
  });
  dlg.addEventListener('click',function(){dlg.close()});
})();
"""


def _image_src(sample: Sample, embed: bool, report_dir: Path) -> str:
    if embed:
        b64 = base64.b64encode(sample.image_path.read_bytes()).decode("ascii")
        return f"data:{sample.mime};base64,{b64}"
    return Path(os.path.relpath(sample.image_path, report_dir)).as_posix()


def _anchor(sample_id: str) -> str:
    return "s-" + re.sub(r"[^\w-]+", "-", sample_id)


def _best(values: list[float | None], higher_is_better: bool) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return max(vals) if higher_is_better else min(vals)


def _leaderboard(models: list[dict]) -> str:
    cols: list[tuple[str, str, str, bool]] = [
        # key, header, tooltip, higher_is_better
        ("cer", "CER", "Character error rate: (substitutions + deletions + insertions) / reference characters, "
                       "micro-averaged over all images. Line wraps are ignored. Lower is better.", False),
        ("cer_lenient", "CER lenient", "CER ignoring case, punctuation and configured variants such as ё/е.", False),
        ("wer", "WER", "Word error rate, punctuation ignored, case-sensitive.", False),
        ("punct_f1", "Punct. F1", "F1 of punctuation marks aligned to the reference.", True),
        ("para_f1", "¶ F1", "F1 of paragraph breaks aligned to the reference.", True),
        ("case_errors", "Case err.", "Characters wrong only by upper/lower case.", False),
        ("diacritic_errors", "Diacr. err.", "Characters wrong only by a diacritic (е/ё, e/é).", False),
    ]

    def value(m: dict, key: str) -> float | None:
        a = m["agg"]
        if not a.get("n"):
            return None
        if key == "punct_f1":
            return a["punct"]["f1"]
        if key == "para_f1":
            return a["para"]["f1"]
        return a.get(key)

    best = {k: _best([value(m, k) for m in models], hib) for k, _, _, hib in cols}
    max_cer = max([value(m, "cer") or 0 for m in models] + [1e-9])

    head = "".join(f'<th title="{esc(tip)}">{esc(h)}</th>' for _, h, tip, _ in cols)
    rows = []
    for rank, m in enumerate(models, 1):
        cells = []
        for key, _, _, _ in cols:
            v = value(m, key)
            cls = ' class="best"' if v is not None and v == best[key] and len(models) > 1 else ""
            if key == "cer" and v is not None:
                w = 100 * v / max_cer
                cells.append(f'<td{cls}><div class="meter"><span>{pct(v, 2)}</span>'
                             f'<span class="track" aria-hidden="true"><span class="fill" style="width:{w:.1f}%;display:block"></span></span></div></td>')
            elif key in ("case_errors", "diacritic_errors"):
                cells.append(f"<td{cls}>{num(v)}</td>")
            else:
                cells.append(f"<td{cls}>{pct(v, 1)}</td>")
        ok = f'{m["n_ok"]}/{m["n_total"]}'
        if m["n_err"]:
            ok += f' <span class="badge err">{m["n_err"]} failed</span>'
        tokens = "—" if not m["tokens_in"] and not m["tokens_out"] else f'{num(m["tokens_in"])} / {num(m["tokens_out"])}'
        cost = "—" if m["cost"] is None else f'${m["cost"]:.4f}'
        rows.append(
            f'<tr><td>{rank}. <span class="model-name">{esc(m["label"])}</span>'
            f'<span class="model-sub">{esc(m["provider"])} · {esc(m["model"] or "—")}</span></td>'
            + "".join(cells)
            + f'<td>{ok}</td><td>{num(m["latency_avg"], 1)}{"s" if m["latency_avg"] is not None else ""}</td>'
              f'<td>{tokens}</td><td>{cost}</td></tr>'
        )
    return (
        '<div class="card scroll"><table><thead><tr><th>Model</th>' + head
        + '<th title="Images transcribed successfully / total">Images</th>'
          '<th title="Mean request latency">Latency</th>'
          '<th title="Total input / output tokens (Gemini output includes thinking tokens)">Tokens in / out</th>'
          '<th title="Total cost from the price set in config.yaml">Cost</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
    )


def _heatmap(samples: list[Sample], models: list[dict], results: dict) -> str:
    head = "".join(f"<th>{esc(m['label'])}</th>" for m in models)
    rows = []
    for s in samples:
        cells = []
        for m in models:
            r = results[m["label"]].get(s.id)
            if not r or not r.get("metrics"):
                why = "failed" if r and r.get("error") else "not run"
                cells.append(f'<td class="h na"><a href="#{_anchor(s.id)}" title="{esc(m["label"])} · {esc(s.id)}: {why}">{why}</a></td>')
                continue
            cer = r["metrics"]["cer"]
            tip = f'{m["label"]} · {s.id}: CER {pct(cer, 2)}, WER {pct(r["metrics"]["wer"], 1)}'
            cells.append(f'<td class="h b{cer_bin(cer)}"><a href="#{_anchor(s.id)}" title="{esc(tip)}">{pct(cer, 1)}</a></td>')
        rows.append(f'<tr><td><a href="#{_anchor(s.id)}">{esc(s.id)}</a></td>{"".join(cells)}</tr>')
    legend = "".join(f'<span><span class="sw b{i}"></span> {lab}</span>' for i, lab in enumerate(CER_BIN_LABELS))
    return (f'<div class="card scroll"><table class="heat"><thead><tr><th>Image</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            f'<div class="legend"><span>CER:</span>{legend}</div>')


def _prf_cell(d: dict | None, extra: str = "") -> str:
    if not d or d.get("f1") is None:
        return '<td class="muted">—</td>'
    return (f'<td title="precision {pct(d["precision"])} · recall {pct(d["recall"])}{extra}">{pct(d["f1"])}'
            f'<span class="model-sub">P {pct(d["precision"], 0)} · R {pct(d["recall"], 0)}</span></td>')


def _punctuation(models: list[dict]) -> str:
    live = [m for m in models if m["agg"].get("n")]
    if not live:
        return '<p class="muted">No results.</p>'
    ref_counts: dict[str, int] = {}
    for m in live:
        for mark, v in m["agg"]["punct_by_mark"].items():
            ref_counts[mark] = max(ref_counts.get(mark, 0), v["tp"] + v["fn"])
    marks = sorted(ref_counts, key=lambda k: (-ref_counts[k], k))
    head = "".join(f"<th>{esc(m['label'])}</th>" for m in live)
    rows = [
        "<tr><td><b>All punctuation</b></td><td>"
        + num(sum(ref_counts.values())) + "</td>" + "".join(_prf_cell(m["agg"]["punct"]) for m in live) + "</tr>",
        f'<tr><td><b>¶ Paragraph breaks</b></td><td>{num(max(m["agg"]["para_tp"] + m["agg"]["para_fn"] for m in live))}</td>'
        + "".join(_prf_cell(m["agg"]["para"], f' · {m["agg"]["para_hyp"]} paragraphs vs {m["agg"]["para_ref"]} in reference') for m in live)
        + "</tr>",
    ]
    for mark in marks:
        name = {" ": "space", '"': 'quote "', "'": "apostrophe '", "—": "dash —", "-": "hyphen -"}.get(mark, mark)
        cells = "".join(_prf_cell(m["agg"]["punct_by_mark"].get(mark)) for m in live)
        rows.append(f'<tr><td><span class="mono chip-c">{esc(mark)}</span> <span class="small">{esc(name) if name != mark else ""}</span></td>'
                    f"<td>{num(ref_counts[mark])}</td>{cells}</tr>")
    return (f'<div class="card scroll"><table><thead><tr><th>Mark</th><th title="Occurrences in the reference">In ref.</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="small">Cells show F1 with precision (how many marks the model wrote were right) and recall '
            '(how many reference marks the model got). A mark that appears only in model output has no recall.</p>')


def _char_classes(models: list[dict]) -> str:
    live = [m for m in models if m["agg"].get("n")]
    if not live:
        return '<p class="muted">No results.</p>'
    totals = live[0]["agg"]["class_total"]
    classes = [c for c in CHAR_CLASSES if totals.get(c)]
    head = "".join(f'<th>{c.capitalize()}<span class="model-sub">{num(totals.get(c))} chars</span></th>' for c in classes)
    best = {c: _best([m["agg"]["class_recall"][c] for m in live], True) for c in classes}
    rows = []
    for m in live:
        a = m["agg"]
        cells = "".join(
            f'<td{" class=best" if len(live) > 1 and a["class_recall"][c] == best[c] else ""}>{pct(a["class_recall"][c])}</td>'
            for c in classes)
        rows.append(f'<tr><td class="model-name">{esc(m["label"])}</td>{cells}'
                    f'<td>{num(a["char_sub"])}</td><td>{num(a["char_del"])}</td><td>{num(a["char_ins"])}</td></tr>')
    return (f'<div class="card scroll"><table><thead><tr><th>Model</th>{head}'
            '<th title="Characters replaced by a wrong one">Substituted</th>'
            '<th title="Reference characters missing from the output">Deleted</th>'
            '<th title="Extra characters not in the reference">Inserted</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="small">Recall: share of reference characters of each class that the model reproduced exactly at the aligned position.</p>')


def _conf_label(s: str) -> str:
    if s == "":
        return '<span class="muted">∅</span>'
    return f'<span class="mono chip-c">{esc(s.replace(" ", "␣"))}</span>'


def _confusions(models: list[dict]) -> str:
    cards = []
    for m in models:
        a = m["agg"]
        if not a.get("n"):
            continue

        def table(rows: list, title: str) -> str:
            if not rows:
                return f'<p class="small">{title}: none</p>'
            body = "".join(f"<tr><td>{_conf_label(x)} → {_conf_label(y)}</td><td>{n}</td></tr>" for x, y, n in rows)
            return f'<table><thead><tr><th>{title}: reference → output</th><th>×</th></tr></thead><tbody>{body}</tbody></table>'

        cards.append(f'<div class="card"><h3>{esc(m["label"])}</h3>'
                     f'{table(a["word_confusions"][:10], "Words")}{table(a["char_confusions"][:10], "Characters")}</div>')
    if not cards:
        return '<p class="muted">No results.</p>'
    return f'<details><summary>Show most frequent word and character errors per model</summary><div class="conf">{"".join(cards)}</div></details>'


def _prediction_card(m: dict, r: dict | None, ref_norm, norm: NormConfig) -> str:
    title = (f'<h3>{esc(m["label"])} <span class="model-sub" style="display:inline">'
             f'{esc(m["provider"])} · {esc(m["model"] or "—")}</span></h3>')
    if r is None:
        return f'<div class="card pred"><div class="pred-head">{title}<span class="badge warn">not run</span></div></div>'
    if r.get("error"):
        return (f'<div class="card pred"><div class="pred-head">{title}<span class="badge err">failed</span></div>'
                f'<div class="errbox">{esc(r["error"])}</div></div>')
    mt = r["metrics"]
    badges = [f'<span class="badge cer b{cer_bin(mt["cer"])}">CER {pct(mt["cer"], 1)}</span>']
    if r.get("stale"):
        badges.append('<span class="badge warn" title="Prompt, model settings or image changed since this output was produced. '
                      'Re-run with `ocr-eval run` to refresh.">stale</span>')
    if r.get("finish_reason") in ("length", "MAX_TOKENS", "max_tokens"):
        badges.append('<span class="badge warn">truncated (max tokens)</span>')
    if not (r.get("text") or "").strip():
        badges.append('<span class="badge err">empty output</span>')
    stats = [
        f'CER lenient <b>{pct(mt["cer_lenient"])}</b>',
        f'WER <b>{pct(mt["wer"])}</b>',
        f'edits <b>{mt["char_sub"]}</b> sub · <b>{mt["char_del"]}</b> del · <b>{mt["char_ins"]}</b> ins',
        f'punct. F1 <b>{pct(mt["punct"]["f1"])}</b>',
        f'¶ <b>{mt["para_hyp"]}</b>/{mt["para_ref"]} (F1 {pct(mt["para"]["f1"], 0)})',
    ]
    if mt["case_errors"] or mt["diacritic_errors"]:
        stats.append(f'case <b>{mt["case_errors"]}</b> · diacritic <b>{mt["diacritic_errors"]}</b>')
    if r.get("latency_s") is not None:
        stats.append(f'<b>{r["latency_s"]:.1f}s</b>')
    if r.get("input_tokens") or r.get("output_tokens"):
        stats.append(f'tokens <b>{num(r.get("input_tokens"))}</b> / <b>{num(r.get("output_tokens"))}</b>')

    hyp_norm = normalize(r["text"] or "", norm, is_output=True)
    word = render_diff(ref_norm.structured, hyp_norm.structured, "word")
    char = render_diff(ref_norm.structured, hyp_norm.structured, "char")
    return (f'<div class="card pred"><div class="pred-head">{title}{"".join(badges)}</div>'
            f'<div class="stats">{"".join(f"<span>{s}</span>" for s in stats)}</div>'
            f'<div class="diff dw" style="margin-top:10px">{word}</div>'
            f'<div class="diff dc" style="margin-top:10px">{char}</div>'
            f'<details style="margin-top:8px"><summary>Raw model output</summary><pre class="raw">{esc(r["text"] or "")}</pre></details>'
            '</div>')


def _sample_section(s: Sample, models: list[dict], results: dict, norm: NormConfig, img_src: str) -> str:
    ref_norm = normalize(s.reference, norm, is_output=False)
    cards = "".join(_prediction_card(m, results[m["label"]].get(s.id), ref_norm, norm) for m in models)
    return (f'<section class="sample" id="{_anchor(s.id)}">'
            f'<div class="side"><h3>{esc(s.id)}</h3>'
            f'<img class="thumb" src="{esc(img_src)}" alt="Handwritten page {esc(s.id)}" loading="lazy">'
            f'<p class="small">{len(ref_norm.flat):,} chars · {len(ref_norm.paragraphs)} paragraphs · click image to enlarge</p>'
            f'<details><summary>Reference text</summary><div class="card ref">{esc(s.reference)}</div></details></div>'
            f'<div>{cards}</div></section>')


def _setup(models: list[dict], prompts: dict[str, tuple[str, str]], norm: NormConfig) -> str:
    groups: dict[tuple[str, str], list[str]] = {}
    for m in models:
        groups.setdefault(prompts[m["label"]], []).append(m["label"])
    prompt_html = []
    for (system, user), labels in groups.items():
        prompt_html.append(f'<h3 style="margin-top:16px">Prompts used by: {esc(", ".join(labels))}</h3>'
                           f'<p class="small">System</p><pre class="prompt">{esc(system or "(none)")}</pre>'
                           f'<p class="small">User</p><pre class="prompt">{esc(user or "(none)")}</pre>')
    model_rows = "".join(
        f'<tr><td class="model-name">{esc(m["label"])}</td><td>{esc(m["provider"])}</td><td>{esc(m["model"] or "—")}</td>'
        f'<td style="text-align:left" class="mono">{esc(json.dumps(m["settings"], ensure_ascii=False))}</td></tr>'
        for m in models)
    char_map = ", ".join(f"{k} → {v}" for k, v in norm.char_map.items()) or "none"
    lenient_map = ", ".join(f"{k} → {v}" for k, v in norm.lenient_char_map.items()) or "none"
    return (
        '<div class="card scroll"><table><thead><tr><th>Label</th><th>Provider</th><th>Model</th>'
        f'<th style="text-align:left">Settings</th></tr></thead><tbody>{model_rows}</tbody></table></div>'
        + "".join(prompt_html)
        + '<h3 style="margin-top:16px">Normalization</h3><ul class="small">'
          '<li>Unicode NFC, zero-width characters removed; whitespace inside a paragraph collapsed (line wraps are not errors).</li>'
          '<li>Paragraphs: blocks separated by a blank line, or every line when the text has no blank lines.</li>'
          f'<li>Character map (both texts): <span class="mono">{esc(char_map)}</span></li>'
          f'<li>Lenient-only map: <span class="mono">{esc(lenient_map)}</span></li>'
          f'<li>Markdown stripped from model output: {"yes" if norm.strip_markdown else "no"}</li></ul>'
    )


def build_report(*, title: str, generated_at: str, data_dir: Path, samples: list[Sample], models: list[dict],
                 results: dict[str, dict[str, dict]], prompts: dict[str, tuple[str, str]], norm: NormConfig,
                 embed_images: bool, report_dir: Path) -> str:
    explain = (
        '<div class="card explain">'
        '<div><b>CER</b> - character error rate: edits needed to turn the output into the reference, divided by '
        'reference length. 0% is perfect; can exceed 100% when the model adds text.</div>'
        '<div><b>CER lenient</b> - same, ignoring case, punctuation and variants like ё/е. The gap to CER shows how much '
        'error comes from punctuation and case rather than reading letters.</div>'
        '<div><b>WER</b> - share of reference words that are wrong, missing or extra (punctuation ignored).</div>'
        '<div><b>Punct. F1 / ¶ F1</b> - how well punctuation marks and paragraph breaks match the reference, '
        'combining precision and recall.</div></div>'
    )
    sections = "".join(
        _sample_section(s, models, results, norm, _image_src(s, embed_images, report_dir)) for s in samples)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<header class="top"><div class="bar-inner">
  <strong>{esc(title)}</strong>
  <nav><a href="#leaderboard">Leaderboard</a><a href="#heatmap">By image</a><a href="#punctuation">Punctuation</a>
  <a href="#characters">Characters</a><a href="#images">Images</a><a href="#setup">Setup</a></nav>
  <span class="spacer"></span>
  <span class="seg" role="radiogroup" aria-label="Diff granularity">
    <input type="radio" name="diff" id="d-word" value="word"><label for="d-word">Word diff</label>
    <input type="radio" name="diff" id="d-char" value="char"><label for="d-char">Char diff</label>
  </span>
  <button id="theme" type="button">Theme: auto</button>
</div></header>
<main>
<h1>{esc(title)}</h1>
<p class="sub">{len(samples)} images · {len(models)} models · data: <span class="mono">{esc(data_dir)}</span> · generated {esc(generated_at)}</p>

<h2 id="leaderboard">Leaderboard</h2>
<p class="sub">Sorted by CER (micro-averaged over all images). Bold = best in column. Hover column headers for definitions.</p>
{_leaderboard(models)}
{explain}

<h2 id="heatmap">CER by image</h2>
<p class="sub"><span class="t-light">Darker</span><span class="t-dark">Brighter</span> = more errors (see legend below). Click a cell to jump to that image.</p>
{_heatmap(samples, models, results)}

<h2 id="punctuation">Punctuation and paragraphs</h2>
<p class="sub">Per-mark F1 over all images, from a character alignment of output and reference.</p>
{_punctuation(models)}

<h2 id="characters">Characters</h2>
<p class="sub">Which kinds of characters each model gets right, and the error mix.</p>
{_char_classes(models)}
<div style="margin-top:12px">{_confusions(models)}</div>

<h2 id="images">Images</h2>
<p class="sub">Diff of normalized text: <del>struck red</del> = in reference but missing or wrong in output,
<ins>underlined green</ins> = what the model wrote instead or added. ¶ marks paragraph breaks.</p>
{sections}

<h2 id="setup">Setup</h2>
{_setup(models, prompts, norm)}
</main>
<dialog id="viewer"><img id="viewer-img" alt=""></dialog>
<script>{JS}</script>
</body>
</html>
"""
