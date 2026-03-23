"""
claude_service.py
=================
Integration layer for the Anthropic Claude API.

Responsibilities
----------------
- Async client lifecycle management
- Request construction and response normalisation
- Token-usage tracking
- Error classification and structured logging

This module is the single point of contact between the application and the
Anthropic SDK. All other services call ``analyze()`` and receive an
``AnalysisResponse`` — they never import the Anthropic SDK directly.
"""

from __future__ import annotations

import logging
from typing import Optional

import anthropic

from app.core.config import get_settings
from app.models.schemas import AnalysisResponse

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel used when the API returns no text blocks at all (should not
# happen in practice, but we guard against it to avoid returning empty strings
# without explanation).
_EMPTY_RESPONSE_SENTINEL = "[No text content returned by model]"


# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------

def _get_async_client() -> anthropic.AsyncAnthropic:
    """
    Instantiate a configured ``AsyncAnthropic`` client.

    We construct a fresh client per call rather than caching a module-level
    singleton. This trades a tiny amount of object-construction overhead for:

    - Guaranteed settings freshness (settings can be reloaded at runtime)
    - No shared state between concurrent requests (each request gets its own
      client with its own connection pool, preventing connection leaks)
    - Simpler test isolation (no need to monkey-patch module globals)

    The Anthropic SDK's ``AsyncAnthropic`` is safe for concurrent use, but
    sharing a single instance across the lifetime of a long-running process
    can lead to connection pool exhaustion under high load.
    """
    settings = get_settings()
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def _extract_text(content: list[anthropic.types.ContentBlock]) -> str:
    """
    Extract and concatenate all ``TextBlock`` entries from a Claude response.

    The Anthropic API can return a mix of content block types (text, tool_use,
    etc.). We collect all text blocks in order and join them with a single
    newline, which preserves paragraph structure when Claude emits multiple
    prose blocks.

    Parameters
    ----------
    content:
        The ``message.content`` list from an Anthropic API response.

    Returns
    -------
    str
        Concatenated text from all TextBlock entries, or the empty-response
        sentinel if no text blocks are present.
    """
    text_blocks: list[str] = [
        block.text
        for block in content
        if hasattr(block, "text") and isinstance(block.text, str) and block.text.strip()
    ]

    if not text_blocks:
        logger.warning("Claude response contained no text blocks | block_types=%s",
                       [type(b).__name__ for b in content])
        return _EMPTY_RESPONSE_SENTINEL

    return "\n".join(text_blocks)


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

async def analyze(
    prompt: str,
    system_prompt: str = "",
    *,
    override_model: Optional[str] = None,
    override_max_tokens: Optional[int] = None,
) -> AnalysisResponse:
    """
    Send a prompt to Claude and return a structured analysis response.

    This is the sole public entry point for all Claude API calls in the
    application. The function is intentionally kept simple: one user turn,
    one assistant turn. Multi-turn conversation management belongs in a
    higher-level layer if ever needed.

    Parameters
    ----------
    prompt:
        The user message to send. Should be a fully rendered, self-contained
        prompt string — no template processing happens here.
    system_prompt:
        Optional system-level instruction. When provided, it is passed as the
        ``system`` parameter to the Anthropic API. When omitted, the API uses
        its default behaviour (no system turn).
    override_model:
        Override the model specified in application settings. Useful for
        lightweight tasks where a cheaper/faster model is sufficient.
    override_max_tokens:
        Override the max_tokens limit from settings. Use when a particular
        analysis is known to require an unusually long or short response.

    Returns
    -------
    AnalysisResponse
        Contains the analysis text, the model ID that served the response, and
        the total token count (input + output).

    Raises
    ------
    anthropic.APIStatusError
        For 4xx/5xx responses from the Anthropic API (bad request, auth
        failure, rate limit, server error, etc.).
    anthropic.APIConnectionError
        For network-level failures reaching the Anthropic API.
    anthropic.APITimeoutError
        When the Anthropic API does not respond within the SDK's timeout.
    """
    settings = get_settings()
    model = override_model or settings.claude_model
    max_tokens = override_max_tokens or settings.claude_max_tokens

    logger.info(
        "Claude request | model=%s max_tokens=%d prompt_chars=%d system_chars=%d",
        model,
        max_tokens,
        len(prompt),
        len(system_prompt),
    )

    request_kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    if system_prompt:
        request_kwargs["system"] = system_prompt

    try:
        client = _get_async_client()
        message = await client.messages.create(**request_kwargs)

    except anthropic.APIStatusError as exc:
        logger.error(
            "Claude API status error | model=%s status=%d message=%s",
            model,
            exc.status_code,
            exc.message,
        )
        raise

    except anthropic.APIConnectionError as exc:
        logger.error("Claude API connection error | model=%s error=%s", model, exc)
        raise

    except anthropic.APITimeoutError as exc:
        logger.error("Claude API timeout | model=%s", model)
        raise

    # -----------------------------------------------------------------------
    # Response normalisation
    # -----------------------------------------------------------------------

    analysis_text = _extract_text(message.content)
    input_tokens: int = message.usage.input_tokens
    output_tokens: int = message.usage.output_tokens
    total_tokens: int = input_tokens + output_tokens

    logger.info(
        "Claude response | model=%s input_tokens=%d output_tokens=%d total=%d "
        "stop_reason=%s",
        message.model,
        input_tokens,
        output_tokens,
        total_tokens,
        message.stop_reason,
    )

    if message.stop_reason == "max_tokens":
        logger.warning(
            "Claude response was truncated at max_tokens=%d | model=%s",
            max_tokens,
            model,
        )

    return AnalysisResponse(
        analysis=analysis_text,
        model=message.model,
        tokens_used=total_tokens,
    )
