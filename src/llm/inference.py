import json
import os
import time
from typing import Dict, Any, Generator, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from llm.provider import Provider
from util.logger import logger

@dataclass
class LLMConfig:
    """Configuration for LLM inference"""
    provider: str
    model: str
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = None
    streaming: bool = False
    max_retries: int = 3


def validate_config(config: LLMConfig) -> Provider:
    """Validate an LLM configuration and return the resolved provider.

    Raises:
        ValueError: if the provider is unsupported or any parameter is out of range.
    """
    try:
        provider = Provider(config.provider)
    except ValueError:
        supported = ", ".join(p.value for p in Provider)
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}. Supported providers: {supported}"
        )

    if not config.model or not str(config.model).strip():
        raise ValueError(f"Model name is required for provider {provider.value}.")

    if config.temperature is not None:
        low, high = provider.config.temperature_range
        if not low <= config.temperature <= high:
            raise ValueError(
                f"Temperature for {provider.value} must be within [{low}, {high}], "
                f"got {config.temperature}."
            )

    if config.max_tokens is not None and config.max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {config.max_tokens}.")

    if config.max_retries is None or config.max_retries < 1:
        raise ValueError(f"max_retries must be at least 1, got {config.max_retries}.")

    return provider


def get_model(config: LLMConfig):
    """Get a model instance based on configuration."""

    provider = validate_config(config)
    model_config = provider.config

    api_key = None
    if model_config.requires_api_key:
        api_key = os.getenv(model_config.env_key)
        if not api_key:
            logger.error(f"API Key Error: Please make sure {model_config.env_key} is set in your .env file.")
            raise ValueError(f"{provider.value} API key not found. Please set {model_config.env_key} in .env file.")

    kwargs = {
        "model": config.model,
        # 给每次 LLM 调用加 120s 硬超时，避免 API 卡住时进程死等
        "request_timeout": 120,
        **({"api_key": api_key} if model_config.requires_api_key else {}),
        **({"base_url": model_config.base_url} if model_config.base_url else {}),
        **({"temperature": config.temperature} if config.temperature is not None else {}),
        **({"max_tokens": config.max_tokens} if config.max_tokens is not None else {}),
        **({"streaming": config.streaming} if config.streaming else {}),
        **({"extra_body": model_config.extra_body} if model_config.extra_body else {}),
    }

    try:
        return model_config.model_class(**kwargs)
    except Exception as e:
        logger.error(f"{provider.value} Chat Error: {e}")
        raise ValueError(f"{provider.value} Chat Error: {e}")

def _describe_empty_response(response) -> str:
    """Summarize an LLM response that failed to produce structured output.

    Records the provider's stop reason and token breakdown, so a ``None`` result
    can be traced to either truncation (``finish_reason=length``, the reasoning
    budget is exhausted) or the model ignoring the forced tool call
    (``finish_reason=stop`` with no tool calls).
    """
    if response is None:
        return "no response returned"

    metadata = response.response_metadata or {}
    usage = response.usage_metadata or {}
    details = usage.get("output_token_details") or {}
    content = (response.content or "")[:200].replace("\n", " ")
    return (
        f"finish_reason={metadata.get('finish_reason')}, "
        f"tool_calls={len(response.tool_calls or [])}, "
        f"output_tokens={usage.get('output_tokens')}, "
        f"reasoning_tokens={details.get('reasoning')}, "
        f"content={content!r}"
    )


def agent_call(prompt: str, llm_config: Dict[str, Any], pydantic_model: BaseModel, agent: Optional[str] = None):
    """
    Makes an agent call with retry logic and structured output.
    
    Args:
        prompt: The prompt to send to the LLM
        llm_config: Configuration for the LLM
        output_model: The Pydantic model to use for structured output
        agent: Optional agent label used to attribute retry warnings
    Returns:
        An instance of output_model (with defaults if error occurs)
    """
    llm_cfg = LLMConfig(**llm_config)
    llm = get_model(llm_cfg)

    # Explicitly use function_calling method for structured output, keeping the
    # raw message so failures can be diagnosed (finish_reason, token usage).
    function_calling_llm = llm.with_structured_output(pydantic_model, method="function_calling", include_raw=True)

    # Fallback for providers (e.g. MiniMax) that intermittently ignore the forced
    # tool call and answer in prose: json_mode constrains the response body to
    # JSON instead. The schema is repeated in the prompt because the provider
    # does not always honor response_format on its own.
    json_mode_llm = llm.with_structured_output(pydantic_model, method="json_mode")
    json_prompt = (
        prompt
        + "\n\nRespond with a single JSON object only, no markdown fences, no commentary, "
        "matching this JSON schema:\n"
        + json.dumps(pydantic_model.model_json_schema())
    )

    label = f"Agent: {agent} | " if agent else ""

    for attempt in range(llm_cfg.max_retries):
        try:
            output = function_calling_llm.invoke(prompt)
            result = output.get("parsed")
            if result is not None:
                return result

            parsing_error = output.get("parsing_error")
            logger.warning(
                f"{label}function_calling returned None "
                f"({_describe_empty_response(output.get('raw'))}"
                + (f", parsing_error={parsing_error}" if parsing_error else "")
                + "); falling back to json_mode"
            )
            result = json_mode_llm.invoke(json_prompt)
            if result is None:
                raise ValueError("json_mode returned None")
            return result
        except Exception as e:
            logger.warning(f"{label}Attempt {attempt + 1}/{llm_cfg.max_retries} failed: {e}")
            if attempt == llm_cfg.max_retries - 1:
                logger.error(f"{label}All {llm_cfg.max_retries} attempts failed")
                return pydantic_model()

    return pydantic_model()


def _is_retryable(error: Exception) -> bool:
    """Return True for transient provider errors worth retrying (rate limit,
    server overload, gateway errors)."""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status in (408, 409, 425, 429, 500, 502, 503, 504, 529)


def generate_text(prompt: str, llm_config: Dict[str, Any]) -> str:
    """Generate a plain text completion with the configured LLM.

    Retries transient provider errors (429/5xx/529) with exponential backoff.

    Args:
        prompt: The prompt to send to the LLM
        llm_config: Configuration for the LLM
    Returns:
        The generated text content
    """
    config = LLMConfig(**{**llm_config, "streaming": False})
    llm = get_model(config)

    last_error = None
    for attempt in range(config.max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            last_error = e
            if not _is_retryable(e) or attempt == config.max_retries - 1:
                break
            delay = 2 ** attempt
            logger.warning(
                f"{config.provider} text generation attempt {attempt + 1}/{config.max_retries} "
                f"failed ({e}); retrying in {delay}s"
            )
            time.sleep(delay)

    logger.error(f"{config.provider} text generation failed: {last_error}")
    raise ValueError(f"{config.provider} text generation failed: {last_error}")


def stream_text(prompt: str, llm_config: Dict[str, Any]) -> Generator[str, None, None]:
    """Stream a text completion chunk by chunk with the configured LLM.

    Retries transient provider errors only when no chunk has been emitted yet,
    so callers never receive duplicated partial output.

    Args:
        prompt: The prompt to send to the LLM
        llm_config: Configuration for the LLM
    Yields:
        Text chunks in generation order
    """
    config = LLMConfig(**{**llm_config, "streaming": True})
    llm = get_model(config)

    last_error = None
    for attempt in range(config.max_retries):
        emitted = False
        try:
            for chunk in llm.stream(prompt):
                if chunk.content:
                    emitted = True
                    yield chunk.content
            return
        except Exception as e:
            last_error = e
            if emitted or not _is_retryable(e) or attempt == config.max_retries - 1:
                break
            delay = 2 ** attempt
            logger.warning(
                f"{config.provider} streaming attempt {attempt + 1}/{config.max_retries} "
                f"failed ({e}); retrying in {delay}s"
            )
            time.sleep(delay)

    logger.error(f"{config.provider} streaming failed: {last_error}")
    raise ValueError(f"{config.provider} streaming failed: {last_error}")
