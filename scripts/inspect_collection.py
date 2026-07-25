"""
查看 Milvus Collection 统计信息和少量样例数据。

使用方式：
    python -m scripts.inspect_collection
"""

import os
import sys

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """查看 Collection 信息。"""
    from pymilvus import MilvusClient

    client = MilvusClient(
        uri=settings.MILVUS_URI,
        token=settings.MILVUS_TOKEN or None,
    )

    collection_name = settings.MILVUS_COLLECTION

    # 检查是否存在
    has = client.has_collection(collection_name)
    print(f"Collection '{collection_name}' 存在: {has}")

    if not has:
        print("请先运行 python -m scripts.build_index 构建知识库")
        return

    # 统计信息
    stats = client.get_collection_stats(collection_name)
    print(f"\nCollection 统计:\n{stats}")

    # 样例数据
    print("\n--- 样例数据 (前 3 条) ---")
    results = client.query(
        collection_name=collection_name,
        filter="chunk_id != ''",
        output_fields=["chunk_id", "parent_id", "source_name", "sku", "category",
                       "route_intent", "document_type"],
        limit=3,
    )
    for i, row in enumerate(results):
        print(f"\n[{i+1}] chunk_id: {row.get('chunk_id')}")
        print(f"    parent_id:    {row.get('parent_id')}")
        print(f"    source_name:  {row.get('source_name')}")
        print(f"    sku:          {row.get('sku')}")
        print(f"    category:     {row.get('category')}")
        print(f"    route_intent: {row.get('route_intent')}")
        print(f"    doc_type:     {row.get('document_type')}")


if __name__ == "__main__":
    main()
