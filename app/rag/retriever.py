"""
统一检索器：调用 BGE-M3 生成 Query embedding，借助 Milvus 混合检索返回候选文档。

对外只暴露检索接口，不负责生成答案。
"""

import asyncio

from app.config import settings
from app.rag.embeddings import encode_query
from app.rag.milvus_store import hybrid_search as milvus_hybrid_search
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve(
    query: str,
    top_k: int | None = None,
    sku_filter: str | None = None,
    category_filter: str | None = None,
    route_intent_filter: str | None = None,
    source_type_filter: list[str] | None = None,
) -> list[dict]:
    """同步检索入口。

    Args:
        query:               用户查询或改写后的查询文本。
        top_k:               返回数量。
        sku_filter:          按 SKU 过滤。
        category_filter:     按分类过滤。
        route_intent_filter: 按路由意图过滤。
        source_type_filter:  按文档类型过滤（如 ["product", "faq", "policy"]）。

    Returns:
        候选文档列表，每条包含 chunk_id、parent_content、score 等字段。
    """
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K

    # 构造 document_type 过滤表达式
    doc_type_expr: str | None = None
    if source_type_filter and len(source_type_filter) > 0:
        quoted = ", ".join(f'"{t}"' for t in source_type_filter)
        doc_type_expr = f"document_type in [{quoted}]"

    # 1. 编码 Query
    dense_vec, sparse_vec = encode_query(query)

    # 2. 混合检索
    candidates = milvus_hybrid_search(
        dense_vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        dense_weight=settings.DENSE_WEIGHT,
        sparse_weight=settings.SPARSE_WEIGHT,
        sku_filter=sku_filter,
        category_filter=category_filter,
        route_intent_filter=route_intent_filter,
        document_type_filter=doc_type_expr,
    )

    logger.info(
        "检索完成: query=%s → %d 条候选 (filter=%s)",
        query[:60], len(candidates), source_type_filter or "无",
    )
    return candidates


async def async_retrieve(
    query: str,
    top_k: int | None = None,
    **filters,
) -> list[dict]:
    """异步检索入口，供 LangGraph 异步节点调用。"""
    return await asyncio.to_thread(retrieve, query, top_k, **filters)


def retrieve_multi(
    queries: list[str],
    top_k: int | None = None,
) -> list[dict]:
    """多查询检索：对每个子查询分别检索后合并去重。

    用于 BROAD 路径的 Multi-Query 拆分。

    Args:
        queries: 子查询列表。
        top_k:   每个子查询的返回量。

    Returns:
        合并去重后的候选文档列表。
    """
    all_candidates: list[dict] = []
    for q in queries:
        candidates = retrieve(q, top_k=top_k)
        all_candidates.extend(candidates)

    # 按 chunk_id 去重，保留最高分
    seen: dict[str, dict] = {}
    for doc in all_candidates:
        cid = doc["chunk_id"]
        if cid not in seen or doc.get("score", 0) > seen[cid].get("score", 0):
            seen[cid] = doc

    merged = sorted(
        seen.values(),
        key=lambda d: d.get("score", 0),
        reverse=True,
    )

    logger.info(
        "Multi-Query 检索: %d 子查询 → %d 候选 → %d 去重",
        len(queries), len(all_candidates), len(merged),
    )
    return merged
