"""
Agent 节点测试：验证两层分类、路由和新节点。

测试目标：
    - classify_query_node 返回 business_intent + route_action + retrieval_strategy
    - 闲聊/其他 → GENERAL → generate_general
    - 投诉/人工客服 → HANDOFF → generate_handoff
    - 订单号查询 → UNSUPPORTED → generate_unsupported
    - RAG+PRECISE → retrieve
    - RAG+VAGUE → hyde_generate → retrieve
    - RAG+BROAD → multi_query_rewrite → retrieve
    - 高置信度 → generate_rag
    - 低置信度 → generate_fallback
"""

import pytest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.state import EcommerceState


class TestPrepareQueryNode:
    """测试 prepare_query_node。"""

    @pytest.mark.asyncio
    async def test_extracts_last_human_message(self):
        from app.agent.nodes import prepare_query_node

        state: EcommerceState = {
            "messages": [
                HumanMessage(content="你好"),
                AIMessage(content="你好！有什么可以帮你的？"),
                HumanMessage(content="TR90 镜框有什么优点？"),
            ],
        }

        result = await prepare_query_node(state)

        assert result["original_query"] == "TR90 镜框有什么优点？"
        assert result["business_intent"] == ""

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        from app.agent.nodes import prepare_query_node

        state: EcommerceState = {"messages": []}
        result = await prepare_query_node(state)

        assert result["original_query"] == ""


class TestClassifyQueryNode:
    """测试两层分类节点。"""

    @pytest.mark.asyncio
    async def test_classifies_general(self):
        """'你好' → 闲聊/其他 → GENERAL。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "你好"}
        result = await classify_query_node(state)

        assert result["business_intent"] == "闲聊/其他"
        assert result["route_action"] == "GENERAL"

    @pytest.mark.asyncio
    async def test_classifies_handoff(self):
        """'我要投诉' → 投诉/人工客服 → HANDOFF。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "我要投诉你们"}
        result = await classify_query_node(state)

        assert result["business_intent"] == "投诉/人工客服"
        assert result["route_action"] == "HANDOFF"

    @pytest.mark.asyncio
    async def test_classifies_unsupported_order(self):
        """'帮我取消订单OD001234' → 订单修改/取消 + is_specific → UNSUPPORTED。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "帮我取消订单OD001234"}
        result = await classify_query_node(state)

        assert result["business_intent"] == "订单修改/取消"
        assert result["route_action"] == "UNSUPPORTED"

    @pytest.mark.asyncio
    async def test_classifies_unsupported_logistics(self):
        """'OD001864多久能到' → 物流查询+订单号 → UNSUPPORTED。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "OD001864多久能到？"}
        result = await classify_query_node(state)

        assert result["business_intent"] == "物流查询"
        assert result["route_action"] == "UNSUPPORTED"

    @pytest.mark.asyncio
    async def test_classifies_rag_precise(self):
        """明确的商品查询 → 商品咨询 → RAG+PRECISE。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "TR90 镜框有什么优点？"}
        result = await classify_query_node(state)

        assert result["business_intent"] == "商品咨询"
        assert result["route_action"] == "RAG"
        assert result["retrieval_strategy"] == "PRECISE"

    @pytest.mark.asyncio
    async def test_classifies_rag_vague(self):
        """含指代词的查询 → RAG+VAGUE。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "它还有其他颜色吗？"}
        result = await classify_query_node(state)

        assert result["route_action"] == "RAG"
        assert result["retrieval_strategy"] == "VAGUE"

    @pytest.mark.asyncio
    async def test_classifies_rag_broad(self):
        """范围过大的查询 → RAG+BROAD。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "怎么选择适合自己的眼镜？"}
        result = await classify_query_node(state)

        assert result["route_action"] == "RAG"
        assert result["retrieval_strategy"] == "BROAD"

    @pytest.mark.asyncio
    async def test_general_policy_logistics_is_rag(self):
        """通用物流政策查询 → RAG（不是 UNSUPPORTED）。"""
        from app.agent.nodes import classify_query_node

        state: EcommerceState = {"original_query": "物流一般几天到？"}
        result = await classify_query_node(state)

        assert result["business_intent"] == "物流查询"
        assert result["route_action"] == "RAG"


class TestHydeAndMultiQuery:
    """测试 HyDE 和 Multi-Query 节点。"""

    @pytest.mark.asyncio
    async def test_hyde_generates_document(self):
        from app.agent.nodes import hyde_generate_node

        state: EcommerceState = {
            "original_query": "它还有别的颜色吗？",
            "messages": [
                HumanMessage(content="TR90 镜框有什么颜色？"),
                AIMessage(content="TR90 镜框有黑色、银色可选。"),
                HumanMessage(content="它还有别的颜色吗？"),
            ],
        }

        with patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(
                content="用户可能在询问 TR90 材质镜框的颜色选项。"
            )
            mock_get_llm.return_value = mock_llm

            result = await hyde_generate_node(state)

            assert "hyde_document" in result
            assert len(result["hyde_document"]) > 0

    @pytest.mark.asyncio
    async def test_multi_query_splits(self):
        from app.agent.nodes import multi_query_rewrite_node

        state: EcommerceState = {"original_query": "怎么选择适合自己的眼镜？"}

        with patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(
                content="如何根据脸型选择镜框\n不同镜框材质对比\n镜片折射率怎么选"
            )
            mock_get_llm.return_value = mock_llm

            result = await multi_query_rewrite_node(state)
            assert len(result["rewritten_queries"]) >= 2

    @pytest.mark.asyncio
    async def test_multi_query_fallback(self):
        from app.agent.nodes import multi_query_rewrite_node

        state: EcommerceState = {"original_query": "选择眼镜的建议"}

        with patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="")
            mock_get_llm.return_value = mock_llm

            result = await multi_query_rewrite_node(state)
            assert result["rewritten_queries"] == ["选择眼镜的建议"]


class TestRetrieveNode:
    """测试检索节点。"""

    @pytest.mark.asyncio
    async def test_high_confidence_rag(self):
        from app.agent.nodes import retrieve_node

        state: EcommerceState = {
            "retrieval_strategy": "PRECISE",
            "original_query": "TR90 镜框材质",
            "business_intent": "商品咨询",
        }

        mock_candidates = [
            {"chunk_id": "c1", "score": 0.95, "rerank_score": 0.9, "parent_content": "test",
             "source_name": "test", "parent_id": "p1", "child_content": "test",
             "sku": "", "category": "", "route_intent": "", "document_type": "product"}
        ]

        with patch("app.agent.nodes.retrieve") as mock_retrieve, \
             patch("app.agent.nodes.rerank") as mock_rerank:
            mock_retrieve.return_value = mock_candidates
            mock_rerank.return_value = mock_candidates

            result = await retrieve_node(state)
            assert result["is_high_confidence"] is True

    @pytest.mark.asyncio
    async def test_no_results_fallback(self):
        from app.agent.nodes import retrieve_node

        state: EcommerceState = {
            "retrieval_strategy": "PRECISE",
            "original_query": "不存在的商品",
            "business_intent": "商品咨询",
        }

        with patch("app.agent.nodes.retrieve", return_value=[]):
            result = await retrieve_node(state)
            assert result["is_high_confidence"] is False
            assert result["confidence"] == 0.0


class TestGenerateNodes:
    """测试生成节点。"""

    @pytest.mark.asyncio
    async def test_generate_rag(self):
        from app.agent.nodes import generate_rag_node

        state: EcommerceState = {
            "ranked_chunks": [
                {"chunk_id": "c1", "source_name": "TR90镜框知识",
                 "parent_content": "TR90材质轻便有弹性。", "sku": "FM001",
                 "parent_id": "p1", "child_content": "TR90材质",
                 "category": "", "route_intent": "", "document_type": "product",
                 "rerank_score": 0.9},
            ],
            "original_query": "TR90镜框",
            "existing_summary": "",
            "business_intent": "商品咨询",
            "messages": [HumanMessage(content="TR90镜框")],
        }

        with patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="TR90材质轻便有弹性。")
            mock_get_llm.return_value = mock_llm

            result = await generate_rag_node(state)
            assert result["answer_mode"] == "rag"
            assert len(result["answer"]) > 0

    @pytest.mark.asyncio
    async def test_generate_general(self):
        from app.agent.nodes import generate_general_node

        state: EcommerceState = {
            "original_query": "你好",
            "existing_summary": "",
            "messages": [HumanMessage(content="你好")],
        }

        with patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="你好！")
            mock_get_llm.return_value = mock_llm

            result = await generate_general_node(state)
            assert result["answer_mode"] == "general"

    @pytest.mark.asyncio
    async def test_generate_handoff(self):
        from app.agent.nodes import generate_handoff_node

        state: EcommerceState = {
            "original_query": "我要投诉",
            "existing_summary": "",
        }

        with patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="非常抱歉...")
            mock_get_llm.return_value = mock_llm

            result = await generate_handoff_node(state)
            assert result["answer_mode"] == "handoff"
            assert len(result["answer"]) > 0

    @pytest.mark.asyncio
    async def test_generate_unsupported(self):
        from app.agent.nodes import generate_unsupported_node

        state: EcommerceState = {
            "original_query": "OD001864多久能到？",
            "existing_summary": "",
        }

        with patch("app.agent.nodes.retrieve") as mock_retrieve, \
             patch("app.agent.nodes.rerank") as mock_rerank, \
             patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_retrieve.return_value = []
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="抱歉，无法查询具体订单...")
            mock_get_llm.return_value = mock_llm

            result = await generate_unsupported_node(state)
            assert result["answer_mode"] == "unsupported"
            assert len(result["answer"]) > 0

    @pytest.mark.asyncio
    async def test_generate_fallback(self):
        from app.agent.nodes import generate_fallback_node

        state: EcommerceState = {
            "original_query": "随机问题",
            "existing_summary": "",
            "business_intent": "商品咨询",
            "messages": [HumanMessage(content="随机问题")],
        }

        with patch("app.agent.nodes.get_llm") as mock_get_llm:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = AIMessage(content="抱歉，暂未找到。")
            mock_get_llm.return_value = mock_llm

            result = await generate_fallback_node(state)
            assert result["answer_mode"] == "fallback"


class TestRoutingFunctions:
    """测试路由函数（纯函数）。"""

    def test_route_general(self):
        from app.agent.graph import route_after_classify

        state: EcommerceState = {"route_action": "GENERAL"}
        assert route_after_classify(state) == "generate_general"

    def test_route_handoff(self):
        from app.agent.graph import route_after_classify

        state: EcommerceState = {"route_action": "HANDOFF"}
        assert route_after_classify(state) == "generate_handoff"

    def test_route_unsupported(self):
        from app.agent.graph import route_after_classify

        state: EcommerceState = {"route_action": "UNSUPPORTED"}
        assert route_after_classify(state) == "generate_unsupported"

    def test_route_rag_precise(self):
        from app.agent.graph import route_after_classify

        state: EcommerceState = {
            "route_action": "RAG",
            "retrieval_strategy": "PRECISE",
        }
        assert route_after_classify(state) == "retrieve"

    def test_route_rag_vague(self):
        from app.agent.graph import route_after_classify

        state: EcommerceState = {
            "route_action": "RAG",
            "retrieval_strategy": "VAGUE",
        }
        assert route_after_classify(state) == "hyde_generate"

    def test_route_rag_broad(self):
        from app.agent.graph import route_after_classify

        state: EcommerceState = {
            "route_action": "RAG",
            "retrieval_strategy": "BROAD",
        }
        assert route_after_classify(state) == "multi_query_rewrite"

    def test_route_high_confidence(self):
        from app.agent.graph import route_after_retrieve

        state: EcommerceState = {"is_high_confidence": True}
        assert route_after_retrieve(state) == "generate_rag"

    def test_route_low_confidence(self):
        from app.agent.graph import route_after_retrieve

        state: EcommerceState = {"is_high_confidence": False}
        assert route_after_retrieve(state) == "generate_fallback"
