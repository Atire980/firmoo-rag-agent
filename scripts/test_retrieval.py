"""
单独测试 RAG 检索核心链路：

    query → hybrid retrieval → rerank → 打印 Top-K、来源与分数

使用方式：
    python -m scripts.test_retrieval

可通过环境变量 TEST_QUERY 指定测试查询。
"""

import os

from app.config import settings
from app.rag.retriever import retrieve
from app.rag.reranker import rerank
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 默认测试查询列表
DEFAULT_QUERIES = [
    "TR90 镜框有什么优点？",
    "它还有其他颜色吗？",
    "怎么选择适合自己的眼镜？",
    "退换货政策是什么？",
]


def test_single_query(query: str) -> None:
    """测试单条查询的完整检索链路。"""
    print(f"\n{'=' * 60}")
    print(f"查询: {query}")
    print(f"{'=' * 60}")

    # Step 1: Hybrid Retrieval
    candidates = retrieve(query, top_k=settings.RETRIEVAL_TOP_K)
    if not candidates:
        print("⚠ 无检索结果")
        return

    print(f"\n[检索] 返回 {len(candidates)} 条候选 (Top-K={settings.RETRIEVAL_TOP_K})")
    for i, doc in enumerate(candidates[:3]):
        print(f"  [{i+1}] score={doc.get('score', 0):.4f} | {doc.get('source_name', '')[:50]}")

    # Step 2: Rerank
    reranked = rerank(query, candidates, top_k=settings.RERANK_TOP_K)
    if not reranked:
        print("⚠ 重排后无结果")
        return

    print(f"\n[重排] Top-{settings.RERANK_TOP_K} 结果:")
    for i, doc in enumerate(reranked):
        print(f"  [{i+1}] rerank_score={doc.get('rerank_score', 0):.4f}")
        print(f"       来源: {doc.get('source_name', '')}")
        print(f"       SKU:  {doc.get('sku', '')}")
        print(f"       类型: {doc.get('document_type', '')}")
        content_preview = doc.get("parent_content", "")[:120].replace("\n", " ")
        print(f"       内容: {content_preview}...")


def main():
    """测试入口。"""
    test_query = os.getenv("TEST_QUERY", "")
    if test_query:
        test_single_query(test_query)
    else:
        for q in DEFAULT_QUERIES:
            test_single_query(q)


if __name__ == "__main__":
    main()
