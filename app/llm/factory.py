"""
LLM 工厂：统一创建和管理 LLM 实例，支持按业务类型缓存。

业务类型：
    qa      → 回答生成（RAG / GENERAL / Fallback）
    rewrite → 查询改写（HyDE / Multi-Query）
    summary → 对话摘要

所有 API Key、Base URL、模型名来自全局配置。
禁止在节点中直接实例化 ChatOpenAI。
"""

from langchain_openai import ChatOpenAI

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# LLM 实例缓存：key=业务类型, value=ChatOpenAI 实例
_llm_cache: dict[str, ChatOpenAI] = {}


def _create_llm(
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """创建 ChatOpenAI 实例。

    Args:
        temperature: 生成温度。qa 类偏高，rewrite/summary 类偏低。
        max_tokens:  最大输出 token 数。

    Returns:
        ChatOpenAI 实例。
    """
    if not settings.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY 未配置，请在 .env 中设置")
    if not settings.LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL 未配置，请在 .env 中设置")

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_llm(business_type: str = "qa") -> ChatOpenAI:
    """获取指定业务类型的 LLM 实例（带缓存）。

    Args:
        business_type: "qa" | "rewrite" | "summary"

    Returns:
        ChatOpenAI 实例。
    """
    if business_type in _llm_cache:
        return _llm_cache[business_type]

    configs = {
        "qa":       {"temperature": 0.3, "max_tokens": 1024},
        "rewrite":  {"temperature": 0.0, "max_tokens": 512},
        "summary":  {"temperature": 0.0, "max_tokens": 512},
    }

    cfg = configs.get(business_type, configs["qa"])
    llm = _create_llm(**cfg)
    _llm_cache[business_type] = llm

    logger.info("LLM 实例创建: type=%s, model=%s", business_type, settings.LLM_MODEL)
    return llm
