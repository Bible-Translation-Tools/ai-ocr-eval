"""Thin REST clients for vision-capable model APIs (no vendor SDKs, so request formats stay explicit)."""

from __future__ import annotations

import asyncio
import base64
import os
import random
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import httpx

# Environment variables checked (in order) for each provider's API key.
KEY_ENV: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "mistral-ocr": ["MISTRAL_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}
DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "mistral-ocr": "https://api.mistral.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "anthropic": "https://api.anthropic.com/v1",
}
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class ProviderError(Exception):
    pass


@dataclass
class ModelSpec:
    label: str
    provider: str
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    image_detail: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict[str, Any])
    system_prompt: str | None = None
    user_prompt: str | None = None
    concurrency: int | None = None
    price: dict[str, float] | None = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"model {d.get('label')!r}: unknown field(s) {sorted(unknown)}")
        spec = cls(**d)
        if spec.provider not in KEY_ENV:
            raise ValueError(f"model {spec.label!r}: unknown provider {spec.provider!r} (use one of {sorted(KEY_ENV)})")
        if spec.provider != "mistral-ocr" and not spec.model:
            raise ValueError(f"model {spec.label!r}: 'model' is required")
        return spec

    def key_env_names(self) -> list[str]:
        if self.api_key_env:
            return [self.api_key_env]
        if self.base_url:
            # Custom endpoint: never fall back to the vendor key, it would be sent to a third party.
            return []
        return KEY_ENV[self.provider]

    def api_key(self) -> str | None:
        for name in self.key_env_names():
            if os.environ.get(name):
                return os.environ[name]
        return None

    def missing_key(self) -> str | None:
        """Name of the env var that must be set, or None if the model can run."""
        names = self.key_env_names()
        if names and not self.api_key():
            return " or ".join(names)
        return None

    def request_params(self) -> dict[str, Any]:
        """Everything that changes the API output (used for the cache fingerprint)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "image_detail": self.image_detail,
            "base_url": self.base_url,
            "extra_body": self.extra_body,
        }


@dataclass
class Prediction:
    text: str | None = None
    error: str | None = None
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    attempts: int = 0


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        out[k] = deep_merge(_obj(out[k]), _obj(v)) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _chat_messages(system: str, user_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-style chat messages: optional system message + one user message."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    return messages


def _build_request(spec: ModelSpec, key: str | None, system: str, user: str,
                   b64: str, mime: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    p = spec.provider
    base = (spec.base_url or DEFAULT_BASE_URL[p]).rstrip("/")
    data_url = f"data:{mime};base64,{b64}"
    headers: dict[str, str] = {}
    body: dict[str, Any]

    if p == "openai":
        if key:
            headers["Authorization"] = f"Bearer {key}"
        image: dict[str, Any] = {"url": data_url}
        if spec.image_detail:
            image["detail"] = spec.image_detail
        body = {"model": spec.model, "messages": _chat_messages(system, [
            {"type": "image_url", "image_url": image},
            {"type": "text", "text": user},
        ])}
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        if spec.max_tokens:
            body["max_completion_tokens"] = spec.max_tokens
        url = f"{base}/chat/completions"

    elif p == "mistral":
        headers["Authorization"] = f"Bearer {key}"
        body = {"model": spec.model, "messages": _chat_messages(system, [
            {"type": "image_url", "image_url": data_url},
            {"type": "text", "text": user},
        ])}
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        if spec.max_tokens:
            body["max_tokens"] = spec.max_tokens
        url = f"{base}/chat/completions"

    elif p == "mistral-ocr":
        headers["Authorization"] = f"Bearer {key}"
        body = {"model": spec.model or "mistral-ocr-latest",
                "document": {"type": "image_url", "image_url": data_url}}
        url = f"{base}/ocr"

    elif p == "gemini":
        headers["x-goog-api-key"] = key or ""
        body = {"contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": mime, "data": b64}},
            {"text": user},
        ]}]}
        if system:
            body["system_instruction"] = {"parts": [{"text": system}]}
        gen: dict[str, Any] = {}
        if spec.temperature is not None:
            gen["temperature"] = spec.temperature
        if spec.max_tokens:
            gen["maxOutputTokens"] = spec.max_tokens
        if gen:
            body["generationConfig"] = gen
        url = f"{base}/models/{spec.model}:generateContent"

    elif p == "anthropic":
        headers["x-api-key"] = key or ""
        headers["anthropic-version"] = "2023-06-01"
        body = {"model": spec.model, "max_tokens": spec.max_tokens or 8192, "messages": [{
            "role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": user},
            ]}]}
        if system:
            body["system"] = system
        if spec.temperature is not None:
            body["temperature"] = spec.temperature
        url = f"{base}/messages"

    else:
        raise ValueError(p)

    return url, headers, deep_merge(body, spec.extra_body)


def _obj(x: Any) -> dict[str, Any]:
    """JSON object or {} (also when the API sends null)."""
    return x if isinstance(x, dict) else {}


def _list(x: Any) -> list[Any]:
    """JSON array or [] (also when the API sends null)."""
    return x if isinstance(x, list) else []


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    chunks = [_obj(c) for c in _list(content)]  # list of content chunks
    return "".join(str(c.get("text", "")) for c in chunks if c.get("type", "text") == "text")


def _parse_response(provider: str, data: dict[str, Any]) -> tuple[str, int | None, int | None, str | None]:
    """Returns (text, input_tokens, output_tokens, finish_reason)."""
    if provider in ("openai", "mistral"):
        choice = _obj(data["choices"][0])
        msg = _obj(choice["message"])
        if msg.get("refusal"):
            raise ProviderError(f"refusal: {msg['refusal']}")
        usage = _obj(data.get("usage"))
        return (_text_of(msg.get("content")), usage.get("prompt_tokens"),
                usage.get("completion_tokens"), choice.get("finish_reason"))

    if provider == "mistral-ocr":
        text = "\n\n".join(str(_obj(page).get("markdown", "")) for page in data["pages"])
        return text, None, None, None

    if provider == "gemini":
        candidates = _list(data.get("candidates"))
        if not candidates:
            raise ProviderError(f"no candidates (promptFeedback: {data.get('promptFeedback')})")
        cand = _obj(candidates[0])
        parts = [_obj(p) for p in _list(_obj(cand.get("content")).get("parts"))]
        text = "".join(str(p.get("text", "")) for p in parts if not p.get("thought"))
        usage = _obj(data.get("usageMetadata"))
        out_tokens: int = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
        return text, usage.get("promptTokenCount"), out_tokens or None, cand.get("finishReason")

    if provider == "anthropic":
        blocks = [_obj(b) for b in data["content"]]
        text = "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
        usage = _obj(data.get("usage"))
        return text, usage.get("input_tokens"), usage.get("output_tokens"), data.get("stop_reason")

    raise ValueError(provider)


def _retry_after(r: httpx.Response) -> float | None:
    try:
        return min(float(r.headers["retry-after"]), 120.0)
    except (KeyError, ValueError):
        return None


async def transcribe(client: httpx.AsyncClient, spec: ModelSpec, sample_id: str, image_path: Path,
                     mime: str, system: str, user: str, *, timeout: float, max_retries: int) -> Prediction:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    url, headers, body = _build_request(spec, spec.api_key(), system, user, b64, mime)
    last_error = ""
    attempt = 0
    for attempt in range(1, max_retries + 2):
        delay = min(60.0, 2.0 ** attempt) + random.random()
        start = time.perf_counter()
        try:
            r = await client.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.HTTPError as e:
            last_error = f"{type(e).__name__}: {e}"
        else:
            latency = time.perf_counter() - start
            if r.status_code < 400:
                try:
                    text, tin, tout, finish = _parse_response(spec.provider, r.json())
                except ProviderError as e:
                    return Prediction(error=str(e), latency_s=latency, attempts=attempt)
                except (KeyError, IndexError, TypeError, ValueError) as e:
                    return Prediction(error=f"unexpected response ({type(e).__name__}: {e}): {r.text[:500]}",
                                      latency_s=latency, attempts=attempt)
                return Prediction(text=text, latency_s=latency, input_tokens=tin, output_tokens=tout,
                                  finish_reason=finish, attempts=attempt)
            last_error = f"HTTP {r.status_code}: {r.text[:500]}"
            if r.status_code not in RETRY_STATUS:
                return Prediction(error=last_error, attempts=attempt)
            delay = _retry_after(r) or delay
        if attempt <= max_retries:
            await asyncio.sleep(delay)
    return Prediction(error=f"failed after {attempt} attempts: {last_error}", attempts=attempt)
