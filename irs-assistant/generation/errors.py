class LLMConnectionError(RuntimeError):
    """Raised when the LLM backend is unreachable or fails to respond."""


class LLMRateLimitError(RuntimeError):
    """Raised when the LLM backend returns a rate limit response.

    Attributes:
        retry_after: Seconds to wait before the limit resets, if known.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
