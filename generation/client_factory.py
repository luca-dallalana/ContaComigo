"""
Factory for the generation client.

Reads INFERENCE_BACKEND from the environment (default: "ollama") and returns
the appropriate client instance. All clients expose the same generate() and
health_check() interface as OllamaClient.

  INFERENCE_BACKEND=ollama  →  OllamaClient  (local Ollama server)
  INFERENCE_BACKEND=groq    →  GroqClient    (Groq cloud API, requires GROQ_API_KEY)
"""

import os

from generation.groq_client import GroqClient
from generation.ollama_client import OllamaClient


def get_generation_client() -> OllamaClient | GroqClient:
    backend = os.environ.get("INFERENCE_BACKEND", "ollama").lower()
    if backend == "groq":
        return GroqClient()
    return OllamaClient()
