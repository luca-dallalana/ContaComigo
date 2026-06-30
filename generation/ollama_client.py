"""
Ollama client for streaming text generation.

Provides OllamaClient with two methods: generate (streaming or single-shot
generation) and health_check (verifies required models are installed).

This module has no imports from the retrieval or ingestion packages.
"""

import json
import logging
import os
import time
from collections.abc import Generator

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL: str = os.environ["OLLAMA_BASE_URL"]
EMBED_MODEL: str = os.environ["EMBED_MODEL"]
GENERATION_MODEL: str = os.environ["GENERATION_MODEL"]
EMBED_ENDPOINT: str = "/api/embed"
GENERATE_ENDPOINT: str = "/api/generate"
TAGS_ENDPOINT: str = "/api/tags"
EMBED_TIMEOUT: float = 30.0
GENERATE_TIMEOUT: float = 120.0
HEALTH_TIMEOUT: float = 5.0
MAX_RETRIES: int = 3
RETRY_DELAYS: list[int] = [1, 2, 4]
OLLAMA_ERROR_MESSAGE: str = "Ollama não está a correr. Inicia com: ollama serve"
TEMPERATURE: float = 0.1


from generation.errors import LLMConnectionError


class OllamaClient:
    """Client for Ollama embedding and text generation.

    Stateless: no instance attributes, no pooled connections. All methods
    read module-level constants loaded from the environment at import time.
    """

    def generate(
        self, prompt: str, stream: bool = True
    ) -> Generator[str, None, None]:
        """Generate a response from the configured language model.

        This is a generator function — nothing executes until the caller
        begins iterating. LLMConnectionError is therefore raised during
        iteration, not at call time. Callers must wrap the `for token in
        client.generate(...)` loop in try/except, not the call itself.

        Always uses temperature=0.1 to minimise hallucination on tax rules.

        Args:
            prompt: The fully-formatted prompt string.
            stream: If True, yields tokens as they arrive. If False, yields
                    the complete response once. Either way the return type is
                    a Generator so callers iterate consistently.

        Yields:
            Individual text tokens (streaming) or the full response (non-streaming).

        Raises:
            LLMConnectionError: on HTTP or connection errors.
        """
        url = f"{OLLAMA_BASE_URL}{GENERATE_ENDPOINT}"
        payload = {
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": TEMPERATURE},
        }

        if stream:
            try:
                with httpx.stream(
                    "POST", url, json=payload, timeout=GENERATE_TIMEOUT
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("done") is False:
                            yield data["response"]
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                raise LLMConnectionError(OLLAMA_ERROR_MESSAGE) from exc
        else:
            try:
                response = httpx.post(url, json=payload, timeout=GENERATE_TIMEOUT)
                response.raise_for_status()
                yield response.json()["response"]
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                raise LLMConnectionError(OLLAMA_ERROR_MESSAGE) from exc

    def health_check(self) -> bool:
        """Check that Ollama is running and both required models are installed.

        Logs a warning for each missing model with the install command.

        Returns:
            True if Ollama is running and both EMBED_MODEL and GENERATION_MODEL
            are available. False if Ollama is unreachable.

        Raises:
            Nothing — all errors are caught and logged.
        """
        url = f"{OLLAMA_BASE_URL}{TAGS_ENDPOINT}"
        try:
            response = httpx.get(url, timeout=HEALTH_TIMEOUT)
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.error("Ollama não está acessível: %s", exc)
            return False

        models = response.json().get("models", [])
        installed = {m["name"] for m in models}

        all_present = True
        for model in [EMBED_MODEL, GENERATION_MODEL]:
            if model not in installed:
                logger.warning(
                    "Modelo '%s' não está instalado. Instala com: ollama pull %s",
                    model,
                    model,
                )
                all_present = False

        return all_present
