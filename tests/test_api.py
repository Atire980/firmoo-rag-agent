"""
API 端到端测试：验证 FastAPI 接口的请求/响应结构。

使用 HTTPX + ASGI 传输进行测试，不启动真实服务器。
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.chat import ChatRequest, ChatResponse


class TestHealthEndpoint:
    """测试 /health 端点。"""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestChatEndpointStructure:
    """测试 /api/chat 的请求验证和响应结构。"""

    @pytest.mark.asyncio
    async def test_chat_request_validation(self):
        request = ChatRequest(
            user_id="user_001",
            session_id="session_001",
            message="TR90 镜框有什么优点？",
        )
        assert request.user_id == "user_001"
        assert request.session_id == "session_001"
        assert request.message == "TR90 镜框有什么优点？"

    @pytest.mark.asyncio
    async def test_chat_request_requires_fields(self):
        with pytest.raises(Exception):
            ChatRequest()  # type: ignore

    @pytest.mark.asyncio
    async def test_chat_response_structure(self):
        response = ChatResponse(
            answer="TR90 镜框采用...",
            sources=["rag_knowledge_base.csv"],
            confidence=0.85,
            answer_mode="rag",
            business_intent="商品咨询",
        )

        assert response.answer != ""
        assert isinstance(response.sources, list)
        assert 0.0 <= response.confidence <= 1.0
        assert response.answer_mode in ("general", "rag", "fallback", "handoff", "unsupported")
        assert response.business_intent == "商品咨询"

    @pytest.mark.asyncio
    async def test_chat_response_defaults(self):
        response = ChatResponse()
        assert response.answer == ""
        assert response.sources == []
        assert response.confidence == 0.0
        assert response.answer_mode == "general"
        assert response.business_intent == ""


class TestChatEndpointIntegration:
    """集成测试。"""

    @pytest.mark.asyncio
    async def test_chat_endpoint_accepts_request(self):
        request_body = {
            "user_id": "user_001",
            "session_id": "session_001",
            "message": "你好",
        }

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.post("/api/chat", json=request_body)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "answer_mode" in data
        assert "business_intent" in data

    @pytest.mark.asyncio
    async def test_chat_endpoint_rejects_invalid_body(self):
        invalid_body = {"user_id": "user_001"}

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test",
        ) as client:
            response = await client.post("/api/chat", json=invalid_body)

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_multi_turn_session_same_thread_id(self):
        from app.agent.memory import build_thread_id

        tid1 = build_thread_id("user_001", "session_001")
        tid2 = build_thread_id("user_001", "session_001")
        tid3 = build_thread_id("user_001", "session_002")

        assert tid1 == tid2
        assert tid1 != tid3

    @pytest.mark.asyncio
    async def test_different_thread_ids_isolated(self):
        from app.agent.memory import build_thread_id

        ids = {
            build_thread_id("u1", "s1"),
            build_thread_id("u1", "s2"),
            build_thread_id("u2", "s1"),
        }

        assert len(ids) == 3
