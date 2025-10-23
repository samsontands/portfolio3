"""AI service helpers for the Streamlit application.

These helpers encapsulate calls to external AI providers so that the rest
of the codebase can depend on simple, well-typed functions.  They raise
:class:`AIServiceError` on failure which keeps the UI layer responsible for
how errors are displayed (e.g. via Streamlit alerts).  This design also makes
unit testing easier because dependencies such as API keys, HTTP sessions, or
OpenAI clients can be injected explicitly.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional
import re

import requests

try:  # Prefer the new OpenAI SDK
    from openai import OpenAI  # type: ignore

    _NEW_OPENAI_SDK = True
except Exception:  # pragma: no cover - fallback path for legacy installs
    OpenAI = None  # type: ignore
    import openai  # type: ignore

    _NEW_OPENAI_SDK = False


SecretsType = Optional[Mapping[str, Any]]


class AIServiceError(RuntimeError):
    """Raised when an AI provider returns an error response."""


def _get_secret_value(secrets: SecretsType, *candidates: str) -> Optional[str]:
    """Return the first truthy value for the provided keys."""
    if not secrets:
        return None
    for key in candidates:
        try:
            value = secrets[key]
        except (KeyError, TypeError):
            continue
        if value:
            return str(value)
    return None


def _ensure_openai_client(api_key: str) -> Any:
    """Create an OpenAI client for the active SDK implementation."""
    if _NEW_OPENAI_SDK:
        return OpenAI(api_key=api_key)  # type: ignore[call-arg]
    openai.api_key = api_key  # type: ignore[attr-defined]
    return openai


def generate_gpt_response(
    gpt_input: str,
    max_tokens: int,
    *,
    secrets: SecretsType = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> str:
    """Generate a completion from OpenAI's chat models.

    Parameters
    ----------
    gpt_input:
        Prompt content to send to the model.
    max_tokens:
        Maximum tokens to generate in the completion.
    secrets:
        Optional mapping providing API credentials (defaults to
        ``st.secrets`` when called from the Streamlit app).
    model:
        Target chat model name.
    temperature:
        Sampling temperature for the request.
    """
    api_key = _get_secret_value(secrets, "OPENAI_API_KEY", "openai_api_key")
    if not api_key:
        raise AIServiceError(
            "OpenAI API key not found. Add 'OPENAI_API_KEY' to your Streamlit secrets."
        )

    try:
        client = _ensure_openai_client(api_key)
        if _NEW_OPENAI_SDK:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": gpt_input}],
            )
            return response.choices[0].message.content.strip()

        response = client.ChatCompletion.create(  # type: ignore[attr-defined]
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": gpt_input}],
        )
        return response.choices[0].message["content"].strip()
    except Exception as exc:  # pragma: no cover - network failure paths
        raise AIServiceError(f"OpenAI request failed: {exc}") from exc


def get_groq_response(
    prompt: str,
    system_prompt: str,
    personal_info: str,
    *,
    secrets: SecretsType = None,
    model: str = "openai/gpt-oss-20b",
    max_tokens: int = 100,
    temperature: float = 0.2,
    session: Optional[requests.sessions.Session] = None,
) -> str:
    """Call the Groq chat completion endpoint and return the message text."""
    api_key = _get_secret_value(secrets, "GROQ_API_KEY", "groq_api_key")
    if not api_key:
        raise AIServiceError(
            "Groq API key not found. Add 'GROQ_API_KEY' to your Streamlit secrets."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{system_prompt} {personal_info}"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    http = session or requests
    try:
        response = http.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:  # pragma: no cover - network
        raise AIServiceError(f"Groq request failed: {exc}") from exc

    if not response.ok:
        message = _extract_error_message(response)
        raise AIServiceError(
            f"Groq API responded with error ({response.status_code}): {message}"
        )

    try:
        data = response.json()
    except ValueError as exc:  # pragma: no cover - unexpected payload
        raise AIServiceError("Groq API returned a non-JSON response") from exc

    if "choices" in data and data["choices"]:
        content = (
            data["choices"][0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if content:
            return content
        raise AIServiceError("Groq response did not include any content")

    if "error" in data:
        message = data["error"].get("message", "Unknown error")
        raise AIServiceError(f"Groq API error: {message}")

    raise AIServiceError("Groq API returned an unexpected response format")


def _extract_error_message(response: requests.Response) -> str:
    """Attempt to pull a helpful message from an error response."""
    try:
        error_payload = response.json()
    except ValueError:
        return response.text
    return (
        error_payload.get("error", {}).get("message")
        or error_payload.get("message")
        or response.text
        or "Unknown error"
    )


def extract_code(gpt_response: str) -> str:
    """Extract code or SQL content from fenced code blocks."""
    if "```" in gpt_response:
        match = re.search(r"```(.*?)```", gpt_response, re.DOTALL)
        if match:
            code = match.group(1)
            return re.sub(
                r"^\s*(python|py|sql)\s*\n",
                "",
                code,
                flags=re.IGNORECASE,
            )
    return gpt_response


__all__ = [
    "AIServiceError",
    "extract_code",
    "generate_gpt_response",
    "get_groq_response",
]
