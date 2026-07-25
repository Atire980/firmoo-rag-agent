"""通用文本辅助函数。"""

import hashlib
import re


def safe_extract_content(message) -> str:
    """安全地从 LangChain 消息对象中提取文本内容。

    Args:
        message: HumanMessage / AIMessage / 字符串。

    Returns:
        提取出的纯文本内容，失败时返回空字符串。
    """
    if isinstance(message, str):
        return message
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # 多模态消息的 content 可能是 list[dict]
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            return " ".join(parts)
    return ""


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白、统一换行。

    Args:
        text: 原始文本。

    Returns:
        清洗后的文本。
    """
    if not text:
        return ""
    # 将多个连续空白符（包括换行）合并为单个空格
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_hash(content: str, length: int = 16) -> str:
    """对内容生成稳定的短哈希，用于 chunk_id / parent_id。

    Args:
        content: 输入文本。
        length: 输出哈希的字符长度。

    Returns:
        十六进制哈希字符串。
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:length]


def format_sources(source_names: list[str]) -> str:
    """将来源列表格式化为追加到回答末尾的字符串。

    Args:
        source_names: 去重后的来源名称列表。

    Returns:
        格式化字符串，如 "\\n\\n---\\n📖 参考来源：\\n- xxx\\n- yyy"
    """
    if not source_names:
        return ""
    lines = ["", "---", "📖 参考来源："]
    for name in source_names:
        lines.append(f"- {name}")
    return "\n".join(lines)
