from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schemas import ProviderResponse, canonical_json


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    def generate_json(
        self, prompt: str, schema: dict[str, Any], seed: int | None = None
    ) -> ProviderResponse:
        raise NotImplementedError


class OpenAICompatibleProvider(LLMProvider):
    """Minimal OpenAI-compatible provider with content-addressed raw-response caching."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | str = "artifacts/cache/llm",
        timeout: float = 120.0,
    ) -> None:
        self.name = "openai-compatible"
        self.model = model
        self.base_url = (
            base_url or os.getenv("LAYA_LAB_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("LAYA_LAB_LLM_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    def generate_json(
        self, prompt: str, schema: dict[str, Any], seed: int | None = None
    ) -> ProviderResponse:
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return data only. Never return or suggest executable code.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "laya_lab_data", "strict": True, "schema": schema},
            },
        }
        if seed is not None:
            request_body["seed"] = seed
        key = hashlib.sha256(
            canonical_json({"url": self.base_url, "body": request_body}).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            raw = json.loads(path.read_text())
        else:
            if not self.api_key:
                raise RuntimeError("LAYA_LAB_LLM_API_KEY is required for uncached generation")
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(request_body).encode(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    raw = json.loads(response.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:1000]
                if exc.code not in (400, 422):
                    raise RuntimeError(f"provider returned HTTP {exc.code}: {body}") from exc
                # Some OpenAI-compatible providers support JSON mode but not strict json_schema.
                # Retry once with the schema embedded in the prompt; validation below remains strict.
                fallback = dict(request_body)
                fallback["response_format"] = {"type": "json_object"}
                fallback["messages"] = list(request_body["messages"])
                fallback["messages"][1] = {
                    "role": "user",
                    "content": prompt
                    + "\nReturn an object matching this JSON Schema exactly:\n"
                    + canonical_json(schema),
                }
                fallback_req = urllib.request.Request(
                    self.base_url + "/chat/completions",
                    data=json.dumps(fallback).encode(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                try:
                    with urllib.request.urlopen(fallback_req, timeout=self.timeout) as response:
                        raw = json.loads(response.read())
                except urllib.error.HTTPError as fallback_exc:
                    fallback_body = fallback_exc.read().decode(errors="replace")[:1000]
                    raise RuntimeError(
                        f"provider returned HTTP {fallback_exc.code}: {fallback_body}"
                    ) from fallback_exc
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        try:
            content_raw = raw["choices"][0]["message"]["content"]
            content = _decode_json_content(content_raw)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("provider response did not contain valid structured JSON") from exc
        return ProviderResponse(
            content=content,
            provider=self.name,
            model=self.model,
            seed=seed,
            prompt=prompt,
            cache_key=key,
        )


def _decode_json_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class StaticProvider(LLMProvider):
    """Test provider that still exercises strict response validation."""

    def __init__(self, responses: list[dict[str, Any]], model: str = "static-test") -> None:
        self.name, self.model, self.responses = "static", model, list(responses)

    def generate_json(
        self, prompt: str, schema: dict[str, Any], seed: int | None = None
    ) -> ProviderResponse:
        if not self.responses:
            raise RuntimeError("StaticProvider has no response left")
        content = self.responses.pop(0)
        key = hashlib.sha256((prompt + canonical_json(content)).encode()).hexdigest()
        return ProviderResponse(
            content=content,
            provider=self.name,
            model=self.model,
            seed=seed,
            prompt=prompt,
            cache_key=key,
        )
