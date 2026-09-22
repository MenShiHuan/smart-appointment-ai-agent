"""Model provider factory for chat LLMs and embeddings.

Supports Azure OpenAI and OpenAI-compatible providers such as Qwen,
DeepSeek, Zhipu, and OpenAI by switching environment variables.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from pydantic import SecretStr

load_dotenv()

# 支持的对话模型厂商列表（均兼容 OpenAI 的 /v1/chat/completions 协议格式）
CHAT_PROVIDERS = {"openai", "qwen", "deepseek", "zhipu", "openai-compatible"}
# 支持的 Embedding 向量模型厂商列表（ollama：本地部署，走 OpenAI 兼容 /v1/embeddings）
EMBEDDING_PROVIDERS = {"openai", "qwen", "zhipu", "openai-compatible", "ollama"}


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
            model=_env("LLM_MODEL", "qwen-plus") or "qwen-plus",
            api_key=SecretStr(_env("LLM_API_KEY", "") or ""),
            base_url=_env("LLM_BASE_URL"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER={provider!r}. "
        "Use azure, qwen, deepseek, zhipu, openai, or openai-compatible."
    )


def _check_ollama_health(base_url: str) -> None:
    """检查本地 Ollama 服务可用性；不可用时给出明确告警（不抛异常）。"""
    try:
        import requests
        tags_url = base_url.rstrip("/").replace("/v1", "") + "/api/tags"
        requests.get(tags_url, timeout=2)
    except Exception as e:
        print(
            f"[WARN] 本地 Ollama 服务不可达（{base_url}）：{e}。"
            "请确认已启动 Ollama 并已 `ollama pull bge-m3`；"
            "或将 EMBEDDING_PROVIDER 切换为其他 OpenAI 兼容服务。"
        )


def create_embedding_model():
    """Create an embedding model from environment configuration."""
    provider = (_env("EMBEDDING_PROVIDER") or get_model_provider()).strip().lower()

    if provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=_env("AZURE_OPENAI_DEPLOYMENT_EMBEDDING"),
            api_key=SecretStr(_env("AZURE_OPENAI_API_KEY", "") or ""),
            api_version=_env("AZURE_OPENAI_EMBEDDING_VERSION", "2023-05-15"),
            azure_endpoint=_env("AZURE_OPENAI_ENDPOINT_EMBEDDING"),
        )

    if provider in EMBEDDING_PROVIDERS:
        if provider == "ollama":
            # 本地 Ollama：启动健康检查，明确告警而不是静默失败
            _check_ollama_health(_env("EMBEDDING_BASE_URL") or "http://localhost:11434/v1")
        return OpenAIEmbeddings(
            model=_env("EMBEDDING_MODEL", "text-embedding-v3") or "text-embedding-v3",
            api_key=SecretStr(_env("EMBEDDING_API_KEY") or _env("LLM_API_KEY", "") or ""),
            base_url=_env("EMBEDDING_BASE_URL") or _env("LLM_BASE_URL"),
            # OpenAI-compatible providers like DashScope (Qwen) only accept raw
            # strings; disable token-id batching to send plain text.
            check_embedding_ctx_length=False,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER={provider!r}. "
        "Use azure, qwen, zhipu, openai, openai-compatible, or ollama."
    )