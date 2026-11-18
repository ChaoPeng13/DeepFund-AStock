import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Type

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_fireworks import ChatFireworks
from langchain_core.language_models.chat_models import BaseChatModel

@dataclass
class ModelConfig:
    """Configuration for a model provider"""
    model_class: Type[BaseChatModel]
    env_key: Optional[str] = None
    base_url: Optional[str] = None
    requires_api_key: bool = True
    # Inclusive range accepted by the provider for the `temperature` parameter
    temperature_range: Tuple[float, float] = (0.0, 2.0)
    # Provider-specific request parameters forwarded as-is
    extra_body: Optional[dict] = None

class Provider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    ALIBABA = "Alibaba"
    ZHIPU = "ZhiPu"
    MINIMAX = "MiniMax"
    OLLAMA = "Ollama"
    FIREWORKS= "Fireworks"
    YIZHAN = "YiZhan"
    AIHUBMIX = "AiHubMix"

    @property
    def config(self) -> ModelConfig:
        """Get the configuration for this provider"""
        PROVIDER_CONFIGS = {
            Provider.OPENAI: ModelConfig(
                model_class=ChatOpenAI,
                env_key="OPENAI_API_KEY",
            ),
            Provider.ANTHROPIC: ModelConfig(
                model_class=ChatAnthropic,
                env_key="ANTHROPIC_API_KEY",
                temperature_range=(0.0, 1.0),
            ),
            Provider.DEEPSEEK: ModelConfig(
                model_class=ChatDeepSeek,
                env_key="DEEPSEEK_API_KEY",
            ),
            Provider.ALIBABA: ModelConfig(
                model_class=ChatOpenAI,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                env_key="QWEN_API_KEY",
            ),
            Provider.ZHIPU: ModelConfig(
                model_class=ChatOpenAI,
                base_url="https://open.bigmodel.cn/api/paas/v4",
                env_key="ZHIPU_API_KEY",
            ),
            Provider.MINIMAX: ModelConfig(
                model_class=ChatOpenAI,
                # Global keys: https://api.minimax.io/v1
                # Mainland China keys: https://api.minimaxi.com/v1
                base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.cn/v1"),
                env_key="MINIMAX_API_KEY",
                # Split thinking content out of `content` into reasoning fields
                extra_body={"reasoning_split": True},
            ),
            Provider.OLLAMA: ModelConfig(
                model_class=ChatOllama,
                requires_api_key=False,
            ),
            Provider.FIREWORKS: ModelConfig(
                model_class=ChatFireworks,
                env_key="FIREWORKS_API_KEY",
            ),
            Provider.YIZHAN: ModelConfig(
                model_class=ChatOpenAI,
                env_key="YIZHAN_API_KEY",
                base_url="https://vip.yi-zhan.top/v1",
            ),
            Provider.AIHUBMIX: ModelConfig(
                model_class=ChatOpenAI,
                env_key="AIHUBMIX_API_KEY",
                base_url="https://api.aihubmix.com/v1",
            ),
        }
        return PROVIDER_CONFIGS[self]