"""Model provider factory for chat LLMs.

Supports Azure OpenAI and OpenAI-compatible providers such as Qwen,
DeepSeek, Zhipu, and OpenAI by switching environment variables.

（Embedding 工厂已移除：全链路向量化统一走 DashScope text-embedding-v3，
见 services/text_embedding.py 与 services/vector_store.py。）
"""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

# 支持的对话模型厂商列表（均兼容 OpenAI 的 /v1/chat/completions 协议格式）
CHAT_PROVIDERS = {"openai", "qwen", "deepseek", "zhipu", "openai-compatible"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def get_model_provider() -> str:
    """Return configured provider, defaulting to Azure for backward compatibility."""
    return (_env("MODEL_PROVIDER", "azure") or "azure").strip().lower()


def create_chat_model(temperature: float = 0):
    """Create a chat model from environment configuration.

    Azure-compatible env vars:
        MODEL_PROVIDER=azure
        AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_VERSION

    OpenAI-compatible env vars:
        MODEL_PROVIDER=qwen|deepseek|zhipu|openai|openai-compatible
        LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    """
    provider = get_model_provider()

    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            api_version=_env("AZURE_OPENAI_VERSION"),
            temperature=temperature,
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            api_key=SecretStr(_env("AZURE_OPENAI_API_KEY", "") or ""),
        )

    if provider in CHAT_PROVIDERS:
        return ChatOpenAI(
            # 未配置 LLM_MODEL 时默认使用 qwen-plus
            model=_env("LLM_MODEL", "qwen-plus") or "qwen-plus",
            api_key=SecretStr(_env("LLM_API_KEY", "") or ""),
            base_url=_env("LLM_BASE_URL"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER={provider!r}. "
        "Use azure, qwen, deepseek, zhipu, openai, or openai-compatible."
    )
