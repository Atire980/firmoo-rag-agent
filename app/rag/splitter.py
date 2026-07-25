"""
父子分块器：将每条文档拆分为父块（保留完整上下文）和子块（用于向量检索）。

- 父块大小默认 1200 字符，子块大小默认 350 字符。
- 每个子块通过 parent_id 关联到对应的父块。
- Milvus 记录中同时保存 child_content 和 parent_content。
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.utils.logger import get_logger
from app.utils.text import clean_text, stable_hash

logger = get_logger(__name__)


# ── 文本清洗专用分离器 ──────────────────────────────────────────
# 该分离器用于将原始文档按段落/句子边界切分为段落级片段，
# 作为父块的构建基础。
_segment_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", ". ", " ", ""],
    chunk_size=settings.PARENT_CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    length_function=len,
)


def _split_into_parent_chunks(content: str) -> list[str]:
    """将单条文档的正文拆分为多个父块文本。

    Args:
        content: 原始正文。

    Returns:
        父块文本列表。
    """
    cleaned = clean_text(content)
    if not cleaned:
        return []
    segments = _segment_splitter.split_text(cleaned)
    return [seg for seg in segments if seg.strip()]


def _build_child_chunks(parent_text: str) -> list[str]:
    """在单个父块内部，按子块大小进一步拆分。

    不跨父块边界拆分，保证子块检索后能映射回唯一父块。

    Args:
        parent_text: 单个父块的完整文本。

    Returns:
        属于该父块的子块文本列表。
    """
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", ". ", " ", ""],
        chunk_size=settings.CHILD_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
    )
    children = child_splitter.split_text(parent_text)
    return [c for c in children if c.strip()]


def split_document_to_chunks(doc) -> list[dict]:
    """将单个 LangChain Document 拆分为父子块，返回 Milvus 写入记录。

    每条子块记录的字段：
        chunk_id:      子块唯一 ID（hash）
        parent_id:     对应父块唯一 ID（hash）
        child_content: 子块文本
        parent_content: 父块完整文本
        source_name:   来源名称
        sku:           SKU
        category:      分类
        route_intent:  路由意图
        document_type: 文档类型（product / faq / policy / review）

    Args:
        doc: LangChain Document 实例。

    Returns:
        可直接写入 Milvus 的记录列表。
    """
    content = doc.page_content
    metadata = doc.metadata

    # 如果原文很短（<= 子块大小），不需要父子拆分，父块=子块=原文
    if len(content) <= settings.CHILD_CHUNK_SIZE:
        chunk_id = stable_hash(content + (metadata.get("doc_id") or ""))
        return [{
            "chunk_id": chunk_id,
            "parent_id": chunk_id,
            "child_content": content,
            "parent_content": content,
            "source_name": metadata.get("source_name", ""),
            "sku": metadata.get("sku", ""),
            "category": metadata.get("category", ""),
            "route_intent": metadata.get("route_intent", ""),
            "document_type": metadata.get("document_type", ""),
        }]

    # 1. 拆分为父块
    parent_chunks = _split_into_parent_chunks(content)
    if not parent_chunks:
        return []

    records: list[dict] = []

    # 2. 每个父块再拆分为子块
    for parent_text in parent_chunks:
        parent_id = stable_hash(parent_text + (metadata.get("doc_id") or ""))
        child_chunks = _build_child_chunks(parent_text)

        # 如果子块只有一个且等于父块，无需拆分子块
        if len(child_chunks) == 1 and child_chunks[0] == parent_text:
            records.append({
                "chunk_id": stable_hash(parent_text + ":child"),
                "parent_id": parent_id,
                "child_content": parent_text,
                "parent_content": parent_text,
                "source_name": metadata.get("source_name", ""),
                "sku": metadata.get("sku", ""),
                "category": metadata.get("category", ""),
                "route_intent": metadata.get("route_intent", ""),
                "document_type": metadata.get("document_type", ""),
            })
        else:
            for child_text in child_chunks:
                chunk_id = stable_hash(child_text + parent_id)
                records.append({
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "child_content": child_text,
                    "parent_content": parent_text,
                    "source_name": metadata.get("source_name", ""),
                    "sku": metadata.get("sku", ""),
                    "category": metadata.get("category", ""),
                    "route_intent": metadata.get("route_intent", ""),
                    "document_type": metadata.get("document_type", ""),
                })

    return records


def split_all_documents(documents: list) -> list[dict]:
    """对所有文档执行父子分块，合并为统一的记录列表。

    Args:
        documents: LangChain Document 列表。

    Returns:
        Milvus 写入记录列表。
    """
    all_records: list[dict] = []
    for doc in documents:
        records = split_document_to_chunks(doc)
        if records:
            all_records.extend(records)

    # 按 chunk_id 去重
    seen: set[str] = set()
    deduped: list[dict] = []
    for record in all_records:
        cid = record["chunk_id"]
        if cid not in seen:
            seen.add(cid)
            deduped.append(record)

    logger.info(
        "父子分块完成: 输入 %d 文档 → %d 条记录（去重后 %d 条）",
        len(documents), len(all_records), len(deduped),
    )
    return deduped
