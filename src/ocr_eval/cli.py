"""ocr-eval command line: run models on a handwriting dataset, score them, write an HTML report."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

from .dataset import Sample, load_dataset
from .metrics import NormConfig, aggregate, evaluate_pair
from .providers import ModelSpec, Prediction, transcribe
from .report import build_report


# ---------------------------------------------------------------- config

class Config:
    def __init__(self, path: Path, raw: dict[str, Any]):
        self.path = path
        self.base = path.parent
        self.data_dir = self.resolve(raw.get("data_dir", "data"))
        self.output_dir = self.resolve(raw.get("output_dir", "results"))
        prompts = raw.get("prompts") or {}
        self.system_prompt = prompts.get("system")
        self.user_prompt = prompts.get("user")
        self.concurrency = int(raw.get("concurrency", 4))
        self.timeout = float(raw.get("timeout", 180))
        self.max_retries = int(raw.get("max_retries", 4))
        self.embed_images = bool(raw.get("embed_images", True))
        self.title = raw.get("title", "OCR Eval Report")
        self.norm = NormConfig.from_dict(raw.get("normalization"))
        self.models = [ModelSpec.from_dict(m) for m in raw.get("models") or []]
        labels = [m.label for m in self.models]
        dupes = {x for x in labels if labels.count(x) > 1}
        if dupes:
            raise ValueError(f"duplicate model labels: {sorted(dupes)}")

    def resolve(self, p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.base / p).resolve()

    def read_prompt(self, p: str | None) -> str:
        return self.resolve(p).read_text(encoding="utf-8").strip() if p else ""

    def prompts_for(self, spec: ModelSpec) -> tuple[str, str]:
        if spec.provider == "mistral-ocr":
            return "", ""  # dedicated OCR endpoint, doesn't take prompts
        return (self.read_prompt(spec.system_prompt or self.system_prompt),
                self.read_prompt(spec.user_prompt or self.user_prompt))


def load_config(path: Path) -> Config:
    if not path.exists():
        sys.exit(f"config not found: {path}")
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        return Config(path.resolve(), raw)
    except (ValueError, TypeError) as e:
        sys.exit(f"config error: {e}")


# ---------------------------------------------------------------- prediction cache

def safe_name(label: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", label).strip("_") or "model"


def cache_path(cfg: Config, spec: ModelSpec, sample: Sample) -> Path:
    return cfg.output_dir / "predictions" / safe_name(spec.label) / f"{sample.id}.json"


def fingerprint(spec: ModelSpec, system: str, user: str, sample: Sample) -> str:
    payload = json.dumps({"params": spec.request_params(), "system": system, "user": user,
                          "image": sample.image_sha256}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_cache(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- run

def select_models(cfg: Config, only: list[str] | None) -> list[ModelSpec]:
    models = cfg.models
    if only:
        unknown = set(only) - {m.label for m in models}
        if unknown:
            sys.exit(f"unknown model label(s): {sorted(unknown)}. Available: {[m.label for m in models]}")
        return [m for m in models if m.label in only]
    return [m for m in models if m.enabled]


def load_samples(cfg: Config, limit: int | None, ids: list[str] | None) -> list[Sample]:
    if not cfg.data_dir.is_dir():
        sys.exit(f"data_dir not found: {cfg.data_dir}")
    samples, missing = load_dataset(cfg.data_dir)
    for img in missing:
        print(f"warning: no ground-truth .txt for {img.relative_to(cfg.data_dir)} - skipped", file=sys.stderr)
    if ids:
        samples = [s for s in samples if s.id in ids or any(s.id.startswith(i) for i in ids)]
    if limit:
        samples = samples[:limit]
    if not samples:
        sys.exit(f"no image + .txt pairs found in {cfg.data_dir}")
    return samples


async def run_predictions(cfg: Config, specs: list[ModelSpec], samples: list[Sample], force: bool) -> None:
    total = len(specs) * len(samples)
    done = 0

    async def one(client: httpx.AsyncClient, spec: ModelSpec, sem: asyncio.Semaphore, sample: Sample,
                  system: str, user: str) -> None:
        nonlocal done
        path = cache_path(cfg, spec, sample)
        fp = fingerprint(spec, system, user, sample)
        cached = None if force else read_cache(path)
        if cached and cached.get("fingerprint") == fp and not cached.get("error"):
            done += 1
            print(f"[{done}/{total}] {spec.label} · {sample.id}: cached")
            return
        async with sem:
            pred: Prediction = await transcribe(client, spec, sample.id, sample.image_path, sample.mime, system, user,
                                                timeout=cfg.timeout, max_retries=cfg.max_retries)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"fingerprint": fp, "label": spec.label, "provider": spec.provider, "model": spec.model,
                  "sample_id": sample.id, "created_at": datetime.now().isoformat(timespec="seconds"), **asdict(pred)}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        if pred.error:
            print(f"[{done}/{total}] {spec.label} · {sample.id}: ERROR {pred.error[:200]}")
        else:
            cer = evaluate_pair(sample.reference, pred.text or "", cfg.norm)["cer"]
            lat = f"{pred.latency_s:.1f}s" if pred.latency_s is not None else ""
            print(f"[{done}/{total}] {spec.label} · {sample.id}: CER {cer * 100:.1f}% {lat}")

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as client:
        jobs = []
        for spec in specs:
            sem = asyncio.Semaphore(spec.concurrency or cfg.concurrency)
            system, user = cfg.prompts_for(spec)
            jobs += [one(client, spec, sem, s, system, user) for s in samples]
        await asyncio.gather(*jobs)


# ---------------------------------------------------------------- evaluate + report

def evaluate_all(cfg: Config, specs: list[ModelSpec], samples: list[Sample]) -> tuple[list[dict], dict]:
    results: dict[str, dict[str, dict]] = {}
    models: list[dict] = []
    for spec in specs:
        system, user = cfg.prompts_for(spec)
        per_sample: dict[str, dict] = {}
        for s in samples:
            rec = read_cache(cache_path(cfg, spec, s))
            if rec is None:
                continue
            rec["stale"] = rec.get("fingerprint") != fingerprint(spec, system, user, s)
            rec["metrics"] = None if rec.get("error") else evaluate_pair(s.reference, rec.get("text") or "", cfg.norm)
            per_sample[s.id] = rec
        results[spec.label] = per_sample

        ok = [r for r in per_sample.values() if r["metrics"]]
        latencies = [r["latency_s"] for r in ok if r.get("latency_s") is not None]
        tin = sum(r.get("input_tokens") or 0 for r in per_sample.values())
        tout = sum(r.get("output_tokens") or 0 for r in per_sample.values())
        cost = None
        if spec.price and (tin or tout):
            cost = (tin * spec.price.get("input", 0) + tout * spec.price.get("output", 0)) / 1e6
        settings = {k: v for k, v in spec.request_params().items()
                    if v not in (None, {}, "") and k not in ("provider", "model")}
        models.append({
            "label": spec.label, "provider": spec.provider, "model": spec.model, "settings": settings,
            "agg": aggregate([r["metrics"] for r in ok]),
            "n_ok": len(ok), "n_err": sum(1 for r in per_sample.values() if r.get("error")), "n_total": len(samples),
            "latency_avg": sum(latencies) / len(latencies) if latencies else None,
            "tokens_in": tin, "tokens_out": tout, "cost": cost,
        })
    models.sort(key=lambda m: (m["agg"].get("cer") is None, m["agg"].get("cer") or 0))
    return models, results


def write_outputs(cfg: Config, specs: list[ModelSpec], samples: list[Sample], models: list[dict],
                  results: dict) -> Path:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    (out / "results.json").write_text(json.dumps({
        "generated_at": generated_at,
        "data_dir": str(cfg.data_dir),
        "models": models,
        "samples": [{"id": s.id, "image": str(s.image_path), "reference": s.reference,
                     "predictions": {label: results[label].get(s.id) for label in results}} for s in samples],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "sample", "error", "cer", "cer_lenient", "wer", "punct_f1", "para_f1",
                    "para_hyp", "para_ref", "char_sub", "char_del", "char_ins", "case_errors", "diacritic_errors",
                    "latency_s", "input_tokens", "output_tokens"])
        for m in models:
            for s in samples:
                r = results[m["label"]].get(s.id)
                if r is None:
                    continue
                mt = r.get("metrics") or {}
                w.writerow([m["label"], s.id, r.get("error") or "", mt.get("cer"), mt.get("cer_lenient"), mt.get("wer"),
                            (mt.get("punct") or {}).get("f1"), (mt.get("para") or {}).get("f1"),
                            mt.get("para_hyp"), mt.get("para_ref"), mt.get("char_sub"), mt.get("char_del"),
                            mt.get("char_ins"), mt.get("case_errors"), mt.get("diacritic_errors"),
                            r.get("latency_s"), r.get("input_tokens"), r.get("output_tokens")])

    report = out / "report.html"
    report.write_text(build_report(
        title=cfg.title, generated_at=generated_at, data_dir=cfg.data_dir, samples=samples, models=models,
        results=results, prompts={s.label: cfg.prompts_for(s) for s in specs}, norm=cfg.norm,
        embed_images=cfg.embed_images, report_dir=out), encoding="utf-8")
    return report


def print_summary(models: list[dict]) -> None:
    def p(x: float | None) -> str:
        return "    —" if x is None else f"{x * 100:5.1f}%"

    print(f"\n{'model':<28} {'CER':>7} {'CERlen':>7} {'WER':>7} {'punctF1':>8} {'¶F1':>7}  images")
    for m in models:
        a = m["agg"]
        if not a.get("n"):
            print(f"{m['label']:<28} {'no results':>40}  0/{m['n_total']}"
                  + (f" ({m['n_err']} failed - see report)" if m["n_err"] else ""))
            continue
        print(f"{m['label']:<28} {p(a['cer']):>7} {p(a['cer_lenient']):>7} {p(a['wer']):>7} "
              f"{p(a['punct']['f1']):>8} {p(a['para']['f1']):>7}  {m['n_ok']}/{m['n_total']}"
              + (f" ({m['n_err']} failed)" if m["n_err"] else ""))


# ---------------------------------------------------------------- commands

def cmd_check(cfg: Config, args: argparse.Namespace) -> None:
    samples = load_samples(cfg, args.limit, args.image)
    print(f"data_dir: {cfg.data_dir}")
    for s in samples:
        print(f"  {s.id:<40} {s.image_path.suffix}  {len(s.reference):>6} chars")
    print(f"{len(samples)} samples\n\nmodels:")
    for spec in cfg.models:
        missing = spec.missing_key()
        status = "disabled" if not spec.enabled else (f"missing {missing}" if missing else "ready")
        print(f"  {spec.label:<28} {spec.provider:<12} {spec.model:<28} {status}")


def cmd_run(cfg: Config, args: argparse.Namespace) -> None:
    samples = load_samples(cfg, args.limit, args.image)
    specs = select_models(cfg, args.only)
    runnable = []
    for spec in specs:
        missing = spec.missing_key()
        if missing:
            print(f"skipping {spec.label}: set {missing} in .env or the environment", file=sys.stderr)
        else:
            runnable.append(spec)
    if runnable:
        print(f"running {len(runnable)} model(s) on {len(samples)} image(s)...")
        asyncio.run(run_predictions(cfg, runnable, samples, args.force))
    else:
        print("no runnable models - building report from cached predictions only", file=sys.stderr)
    finish(cfg, specs, samples)


def cmd_report(cfg: Config, args: argparse.Namespace) -> None:
    finish(cfg, select_models(cfg, args.only), load_samples(cfg, args.limit, args.image))


def finish(cfg: Config, specs: list[ModelSpec], samples: list[Sample]) -> None:
    models, results = evaluate_all(cfg, specs, samples)
    report = write_outputs(cfg, specs, samples, models, results)
    print_summary(models)
    print(f"\nreport:  {report}\nresults: {cfg.output_dir / 'results.json'}, {cfg.output_dir / 'results.csv'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ocr-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in [("run", "call the models (cached), evaluate, write the report"),
                        ("report", "re-evaluate cached predictions and rebuild the report (no API calls)"),
                        ("check", "list dataset pairs and which models have API keys")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
        p.add_argument("--limit", type=int, help="use only the first N images")
        p.add_argument("--image", action="append", help="only images whose id starts with this (repeatable)")
        if name != "check":
            p.add_argument("--only", action="append", help="only this model label (repeatable)")
        if name == "run":
            p.add_argument("--force", action="store_true", help="ignore cached predictions and call the API again")
    args = parser.parse_args(argv)

    load_dotenv(Path.cwd() / ".env")
    cfg = load_config(args.config)
    load_dotenv(cfg.base / ".env")
    {"run": cmd_run, "report": cmd_report, "check": cmd_check}[args.command](cfg, args)


if __name__ == "__main__":
    main()
