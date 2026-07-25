"""
文档读取器：从 CSV / Markdown / TXT 文件中加载原始数据，转换为 LangChain Document。

支持的格式：
- rag_knowledge_base.csv（主知识库）
- *.md / *.txt（物流、退换货政策等纯文本）
"""

import csv
import os
from pathlib import Path

from langchain_core.documents import Document

from app.utils.logger import get_logger
from app.utils.text import clean_text

logger = get_logger(__name__)


def _read_csv_as_documents(file_path: str) -> list[Document]:
    """读取 CSV 文件，每行转换为一个 LangChain Document。

    CSV 预期列（rag_knowledge_base.csv）：
        doc_id, source_type, source_id, sku, category, title,
        content, route_intent, metadata_json

    Args:
        file_path: CSV 文件路径。

    Returns:
        Document 列表。CSV 格式错误时返回空列表。
    """
    documents: list[Document] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                doc_id = (row.get("doc_id") or "").strip()
                content = clean_text(row.get("content") or "")
                if not doc_id or not content:
                    continue

                metadata = {
                    "source": os.path.basename(file_path),
                    "source_name": (row.get("title") or "").strip(),
                    "sku": (row.get("sku") or "").strip(),
                    "category": (row.get("category") or "").strip(),
                    "route_intent": (row.get("route_intent") or "").strip(),
                    "document_type": (row.get("source_type") or "").strip(),
                    "doc_id": doc_id,
                    "source_id": (row.get("source_id") or "").strip(),
                }

                documents.append(Document(page_content=content, metadata=metadata))

        logger.info("从 %s 读取了 %d 条记录", file_path, len(documents))
    except Exception as e:
        logger.error("读取 CSV 文件失败: %s, 错误: %s", file_path, e)

    return documents


def _read_text_as_document(file_path: str) -> Document | None:
    """读取纯文本 / Markdown 文件，整体作为一个 Document。

    Args:
        file_path: 文本文件路径。

    Returns:
        单个 Document，或 None（读取失败时）。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            logger.warning("文件 %s 内容为空，跳过", file_path)
            return None

        filename = os.path.basename(file_path)
        metadata = {
            "source": filename,
            "source_name": filename,
            "sku": "",
            "category": "",
            "route_intent": "",
            "document_type": "policy",
            "doc_id": f"TEXT_{filename}",
            "source_id": "",
        }

        logger.info("从 %s 读取文本，长度 %d 字符", file_path, len(content))
        return Document(page_content=content, metadata=metadata)

    except Exception as e:
        logger.error("读取文本文件失败: %s, 错误: %s", file_path, e)
        return None


def load_documents(data_dir: str = "data/raw") -> list[Document]:
    """从数据目录加载所有支持的文档。

    读取顺序：
    1. rag_knowledge_base.csv → 逐行转为 Document
    2. policies/ 目录下的 .md / .txt → 逐文件转为 Document

    Args:
        data_dir: 原始数据目录路径。

    Returns:
        合并后的 Document 列表。
    """
    all_documents: list[Document] = []
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error("数据目录不存在: %s", data_path)
        return all_documents

    # 1. 加载主知识库 CSV
    kb_csv = data_path / "rag_knowledge_base.csv"
    if kb_csv.exists():
        all_documents.extend(_read_csv_as_documents(str(kb_csv)))
    else:
        logger.warning("未找到知识库 CSV: %s", kb_csv)

    # 2. 加载 policies/ 目录下所有 Markdown 和 TXT
    policies_dir = data_path / "policies"
    if policies_dir.is_dir():
        for ext in ("*.md", "*.txt"):
            for file_path in policies_dir.glob(ext):
                doc = _read_text_as_document(str(file_path))
                if doc is not None:
                    all_documents.append(doc)
    else:
        logger.info("policies/ 目录不存在，跳过纯文本加载")

    logger.info("共加载 %d 条原始文档", len(all_documents))
    return all_documents
