"""
Groq cloud inference client for text generation.

Implements the same generate() / health_check() interface as OllamaClient
so callers can switch backends without code changes. Embeddings are not
provided — those always run locally via Ollama/nomic-embed-text.

Requires GROQ_API_KEY in the environment. GENERATION_MODEL should be set
to a Groq-hosted model tag, e.g. llama-3.3-70b-versatile.
"""

import json
import logging
import os
import time
from collections.abc import Generator

import httpx
from dotenv import load_dotenv

from generation.ollama_client import OllamaConnectionError

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GENERATION_MODEL: str = os.environ.get("GENERATION_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
GENERATE_ENDPOINT: str = "/chat/completions"
MODELS_ENDPOINT: str = "/models"
GENERATE_TIMEOUT: float = 120.0
HEALTH_TIMEOUT: float = 10.0
TEMPERATURE: float = 0.1
MAX_RETRIES: int = 6
RATE_LIMIT_FALLBACK_WAIT: float = 20.0


class GroqClient:
    """Generation-only client for the Groq cloud inference API.

    Uses the OpenAI-compatible chat completions endpoint. Streaming and
    non-streaming modes are supported, matching the OllamaClient interface.
    """

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

    def _wait_for_rate_limit(self, response: httpx.Response) -> None:
        wait = float(response.headers.get("retry-after", RATE_LIMIT_FALLBACK_WAIT))
        logger.warning("Groq rate limit hit. Waiting %.1fs before retry...", wait)
        time.sleep(wait)

    def generate(
        self, prompt: str, stream: bool = True
    ) -> Generator[str, None, None]:
        """Generate a response from the configured Groq-hosted model.

        Retries automatically on 429 rate limit responses, honouring the
        retry-after header. Temperature is fixed at 0.1 to minimise
        hallucination on tax rules.

        Args:
            prompt: Fully-formatted prompt string from build_prompt().
            stream: If True, yields tokens as they arrive via SSE.
                    If False, yields the complete response once.

        Yields:
            Text tokens (streaming) or the full response string (non-streaming).

        Raises:
            OllamaConnectionError: on HTTP or connection errors after all retries.
        """
        url = f"{GROQ_BASE_URL}{GENERATE_ENDPOINT}"
        payload = {
            "model": GENERATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "stream": stream,
        }

        for attempt in range(MAX_RETRIES):
            try:
                if stream:
                    with httpx.stream(
                        "POST", url, json=payload, headers=self._headers(), timeout=GENERATE_TIMEOUT
                    ) as response:
                        if response.status_code == 429:
                            self._wait_for_rate_limit(response)
                            continue
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            data_str = line[len("data: "):]
                            if data_str.strip() == "[DONE]":
                                return
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content")
                            if content:
                                yield content
                        return
                else:
                    response = httpx.post(
                        url, json=payload, headers=self._headers(), timeout=GENERATE_TIMEOUT
                    )
                    if response.status_code == 429:
                        self._wait_for_rate_limit(response)
                        continue
                    response.raise_for_status()
                    yield response.json()["choices"][0]["message"]["content"]
                    return
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                raise OllamaConnectionError(f"Groq API error: {exc}") from exc

        raise OllamaConnectionError("Groq rate limit exceeded after all retries.")

    def health_check(self) -> bool:
        """Check that the Groq API key is valid and the API is reachable.

        Returns:
            True if the API responds with 200. False otherwise.
        """
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY is not set.")
            return False
        url = f"{GROQ_BASE_URL}{MODELS_ENDPOINT}"
        try:
            response = httpx.get(url, headers=self._headers(), timeout=HEALTH_TIMEOUT)
            response.raise_for_status()
            return True
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.error("Groq API not reachable: %s", exc)
            return False
