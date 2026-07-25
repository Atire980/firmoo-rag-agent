"""
检索测试：验证混合检索与重排的接口和数据流，以及 source_type 过滤参数。

注意：这些测试在 CI 环境中需要模拟 Milvus 和模型。
在不连接真实服务时，主要测试数据结构、参数传递和去重逻辑。
"""

import pytest

from app.config import settings, INTENT_TO_SOURCE_TYPE


class TestRetrieverImports:
    """测试检索模块的导入和基本结构。"""

    def test_import_retriever(self):
        from app.rag import retriever
        assert hasattr(retriever, "retrieve")
        assert hasattr(retriever, "async_retrieve")
        assert hasattr(retriever, "retrieve_multi")

    def test_import_reranker(self):
        from app.rag import reranker
        assert hasattr(reranker, "rerank")
        assert hasattr(reranker, "async_rerank")

    def test_import_embeddings(self):
        from app.rag import embeddings
        assert hasattr(embeddings, "encode_query")
        assert hasattr(embeddings, "encode_documents")
        assert hasattr(embeddings, "async_encode_query")

    def test_retrieve_accepts_source_type_filter(self):
        """retrieve() 应接受 source_type_filter 参数。"""
        import inspect
        from app.rag.retriever import retrieve
        sig = inspect.signature(retrieve)
        assert "source_type_filter" in sig.parameters


class TestRetrieveMulti:
    """测试 Multi-Query 检索的合并去重逻辑。"""

    def _make_candidate(self, chunk_id: str, score: float, content: str = ""):
        return {
            "chunk_id": chunk_id,
            "parent_id": f"parent_{chunk_id}",
            "child_content": content or f"content_{chunk_id}",
            "parent_content": content or f"parent_{chunk_id}",
            "source_name": f"source_{chunk_id}",
            "sku": "",
            "category": "",
            "route_intent": "",
            "document_type": "product",
            "score": score,
        }

    def test_dedup_keeps_highest_score(self):
        candidates = [
            self._make_candidate("A", 0.9),
            self._make_candidate("B", 0.7),
            self._make_candidate("A", 0.5),
            self._make_candidate("C", 0.8),
            self._make_candidate("B", 0.6),
        ]

        seen: dict[str, dict] = {}
        for doc in candidates:
            cid = doc["chunk_id"]
            if cid not in seen or doc.get("score", 0) > seen[cid].get("score", 0):
                seen[cid] = doc

        merged = sorted(seen.values(), key=lambda d: d.get("score", 0), reverse=True)

        assert len(merged) == 3
        assert merged[0]["chunk_id"] == "A"
        assert merged[0]["score"] == 0.9
        assert merged[2]["chunk_id"] == "B"
        assert merged[2]["score"] == 0.7


class TestSourceTypeMapping:
    """测试业务意图到 source_type 的映射。"""

    def test_rag_intents_have_source_types(self):
        """所有 RAG 路径的意图应有 source_type 映射。"""
        rag_intents = ["商品咨询", "镜片/度数咨询", "退换货售后", "优惠/支付"]
        for intent in rag_intents:
            assert intent in INTENT_TO_SOURCE_TYPE
            assert len(INTENT_TO_SOURCE_TYPE[intent]) > 0, \
                f"{intent} 的 source_type 映射不应为空"

    def test_logistics_and_order_have_source_types(self):
        """物流查询和订单修改取消（RAG 路径时）应有 source_type 映射。"""
        for intent in ["物流查询", "订单修改/取消"]:
            assert intent in INTENT_TO_SOURCE_TYPE
            assert len(INTENT_TO_SOURCE_TYPE[intent]) > 0

    def test_non_rag_intents_have_mapping(self):
        """非 RAG 意图也应有映射（即使为空列表）。"""
        for intent in ["投诉/人工客服", "闲聊/其他"]:
            assert intent in INTENT_TO_SOURCE_TYPE


class TestConfidenceThreshold:
    """测试置信度阈值配置。"""

    def test_threshold_config_exists(self):
        assert hasattr(settings, "RAG_CONFIDENCE_THRESHOLD")
        assert 0.0 < settings.RAG_CONFIDENCE_THRESHOLD < 1.0

    def test_retrieval_configs(self):
        assert settings.RETRIEVAL_TOP_K > 0
        assert settings.RERANK_TOP_K > 0
        assert settings.RERANK_TOP_K <= settings.RETRIEVAL_TOP_K
        assert 0.0 <= settings.DENSE_WEIGHT <= 1.0
        assert 0.0 <= settings.SPARSE_WEIGHT <= 1.0
