from typing import Any

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


RETRYABLE_GEMINI_STATUSES = {408, 429, 500, 502, 503, 504}


def _is_retryable_gemini_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    response = getattr(error, "response", None)
    status = (
        getattr(error, "code", None)
        or getattr(error, "status_code", None)
        or getattr(response, "status_code", None)
    )
    try:
        if int(status) in RETRYABLE_GEMINI_STATUSES:
            return True
    except (TypeError, ValueError):
        pass
    return type(error).__module__.startswith("httpx") and type(error).__name__ in {
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "WriteError",
        "WriteTimeout",
    }


@retry(
    retry=retry_if_exception(_is_retryable_gemini_error),
    wait=wait_random_exponential(multiplier=0.5, max=4),
    stop=stop_after_attempt(3),
    reraise=True,
)
def generate_content_with_retry(client: Any, **kwargs: Any) -> Any:
    return client.models.generate_content(**kwargs)
