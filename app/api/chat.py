"""
聊天 API 路由：定义 /api/chat 接口。

- 把请求转换成 LangGraph 初始 State。
- 使用 thread_id 调用编译后的图。
- 返回 answer、sources、confidence、answer_mode。
- API 层不直接调用 Milvus、Embedding 或 LLM。
"""

from fastapi import APIRouter

from app.agent.graph import get_ecommerce_graph
from app.agent.memory import build_graph_config
from app.agent.state import EcommerceState
from app.schemas.chat import ChatRequest, ChatResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """聊天接口：接收用户消息，通过 LangGraph Agent 生成回复。

    Args:
        request: ChatRequest (user_id, session_id, message)。

    Returns:
        ChatResponse (answer, sources, confidence, answer_mode)。
    """
    logger.info(
        "收到请求: user=%s, session=%s, message=%s",
        request.user_id, request.session_id, request.message[:80],
    )

    # 构造初始 State
    from langchain_core.messages import HumanMessage

    initial_state: EcommerceState = {
        "messages": [HumanMessage(content=request.message)],
        "user_id": request.user_id,
        "session_id": request.session_id,
    }

    # 构造 config（含 thread_id）
    config = build_graph_config(request.user_id, request.session_id)

    # 调用 LangGraph
    graph = get_ecommerce_graph()

    try:
        final_state = await graph.ainvoke(initial_state, config)
    except Exception as e:
        logger.error("Agent 执行失败: %s", e)
        return ChatResponse(
            answer="抱歉，系统处理您的请求时出现了问题，请稍后重试或联系人工客服。",
            sources=[],
            confidence=0.0,
            answer_mode="fallback",
        )

    # 提取最终输出
    answer = final_state.get("answer", "")
    sources = final_state.get("sources", [])
    confidence = final_state.get("confidence", 0.0)
    answer_mode = final_state.get("answer_mode", "general")
    business_intent = final_state.get("business_intent", "")

    logger.info(
        "响应: mode=%s, intent=%s, confidence=%.3f, answer_len=%d, sources=%d",
        answer_mode, business_intent, confidence, len(answer), len(sources),
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        answer_mode=answer_mode,
        business_intent=business_intent,
    )
