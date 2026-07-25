"""
Reranker（重排序器）：使用 BGE-Reranker 或兼容的 CrossEncoder 对候选结果重新打分。

- 单例加载模型，禁止每次请求重复加载。
- 输入 (query, candidate_document) 对，输出重排后的结果。
- 生成阶段使用 parent_content。
"""

import asyncio
from typing import Any

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 模型单例
_reranker_model: Any = None


def _load_reranker():
    """惰性加载 Reranker 模型。"""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    model_path = settings.RERANKER_MODEL_PATH
    if not model_path:
        raise RuntimeError(
            "RERANKER_MODEL_PATH 未配置，请在 .env 中设置 Reranker 模型本地路径"
        )

    logger.info("正在加载 Reranker 模型: %s", model_path)
    try:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(model_path)
        logger.info("Reranker 模型加载完成")
    except Exception as e:
        logger.error("Reranker 模型加载失败: %s", e)
        raise

    return _reranker_model


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    content_field: str = "child_content",
) -> list[dict]:
    """对候选文档重新打分排序。

    Args:
        query:         当前查询文本。
        candidates:    候选文档列表（来自 retriever）。
        top_k:         重排后保留的数量，默认使用配置中的 RERANK_TOP_K。
        content_field: 用于计算相似度的字段，"child_content" 或 "parent_content"。

    Returns:
        按 rerank 分数降序排列的文档列表，会新增 'rerank_score' 字段。
    """
    if top_k is None:
        top_k = settings.RERANK_TOP_K
    if not candidates:
        logger.info("无候选文档，跳过重排")
        return []

    model = _load_reranker()

    # 构造 (query, document_content) 对
    pairs = [(query, doc.get(content_field, "")) for doc in candidates]

    # 批量打分
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.error("Reranker 打分失败: %s", e)
        # 降级：使用原始 score 排序
        candidates.sort(key=lambda d: d.get("score", 0), reverse=True)
        return candidates[:top_k]

    # 附加 rerank 分数
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    for i, doc in enumerate(candidates):
        doc["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

    # 按 rerank 分数降序排列
    candidates.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)

    result = candidates[:top_k]

    logger.info(
        "重排完成: %d 条 → %d 条, top_score=%.4f",
        len(candidates), len(result),
        result[0].get("rerank_score", 0) if result else 0,
    )
    return result


async def async_rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    content_field: str = "child_content",
) -> list[dict]:
    """异步重排入口。"""
    return await asyncio.to_thread(rerank, query, candidates, top_k, content_field)
