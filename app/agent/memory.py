"""
会话记忆管理：MemorySaver 单例、thread_id 构造、滑动窗口、摘要触发判断。

第一版摘要保存在 LangGraph State 中，不连接 PostgreSQL。
"""

from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 全局 MemorySaver 单例
_memory_saver: MemorySaver | None = None


def get_memory_saver() -> MemorySaver:
    """获取全局 MemorySaver 单例。

    Returns:
        MemorySaver 实例。
    """
    global _memory_saver
    if _memory_saver is None:
        _memory_saver = MemorySaver()
        logger.info("MemorySaver 单例已创建")
    return _memory_saver


def build_thread_id(user_id: str, session_id: str) -> str:
    """根据 user_id + session_id 构造稳定的 thread_id。

    LangGraph 使用 thread_id 区分不同会话的 checkpoint。

    Args:
        user_id:    用户唯一标识。
        session_id: 会话唯一标识。

    Returns:
        组合后的 thread_id，格式: "{user_id}::{session_id}"。
    """
    return f"{user_id}::{session_id}"


def build_graph_config(user_id: str, session_id: str) -> dict:
    """构造 LangGraph 调用所需的 config 字典。

    Args:
        user_id:    用户唯一标识。
        session_id: 会话唯一标识。

    Returns:
        config 字典，包含 configurable.thread_id。
    """
    return {
        "configurable": {
            "thread_id": build_thread_id(user_id, session_id),
        }
    }


def should_trigger_summary(message_count: int) -> bool:
    """判断当前对话轮数是否达到摘要触发阈值。

    Args:
        message_count: 当前 messages 中的消息数量。

    Returns:
        是否应该触发摘要。
    """
    # 每对 Human + AI 消息算一轮，即 2 条消息
    rounds = message_count // 2
    return rounds >= settings.SUMMARY_TRIGGER_ROUNDS


def get_recent_messages(messages: list, window_size: int | None = None) -> list:
    """获取最近 N 轮的消息（滑动窗口）。
    Args:
        messages:    完整的消息列表。
        window_size: 窗口大小（消息条数），默认使用配置值。
    Returns:
        最近 window_size 条消息。
    """
    if window_size is None:
        window_size = settings.MEMORY_WINDOW_SIZE
    return messages[-window_size:] if len(messages) > window_size else messages
