"""
LangGraph EcommerceState 定义。

messages 使用 add_messages Reducer 累加，其他列表字段不使用消息 Reducer。
"""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class EcommerceState(TypedDict, total=False):
    """眼镜电商 RAG Agent 的全局 State。

    所有字段均为可选（total=False），各节点只返回需要更新的字段。

    分类体系（两层）：
        第一层：business_intent   → 八种业务意图标签
        第二层：retrieval_strategy → PRECISE / VAGUE / BROAD（仅 RAG 路径使用）
        路由：  route_action       → GENERAL / HANDOFF / UNSUPPORTED / RAG
    """

    # ── 对话消息（累加） ──
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 会话标识 ──
    user_id: str
    session_id: str

    # ── 查询处理 ──
    original_query: str

    # ── 第一层：业务意图分类（八种标签之一） ──
    business_intent: str            # 商品咨询 / 镜片度数 / 物流查询 / 退换货售后
                                    # / 订单修改取消 / 投诉人工 / 闲聊其他 / 优惠支付
    business_intent_confidence: float  # 分类器置信度

    # ── 路由动作（由 business_intent + 规则推导） ──
    route_action: str               # GENERAL / HANDOFF / UNSUPPORTED / RAG

    # ── 第二层：检索策略（仅 route_action=RAG 时有效） ──
    retrieval_strategy: str         # PRECISE / VAGUE / BROAD

    # ── 查询改写 ──
    hyde_document: str              # VAGUE 路径：HyDE 生成的假设文档
    rewritten_queries: list[str]    # BROAD 路径：Multi-Query 拆分的子查询

    # ── 检索结果 ──
    ranked_chunks: list[dict]       # 重排后的候选文档
    confidence: float               # 检索置信度
    is_high_confidence: bool        # 是否高置信度

    # ── 对话摘要 ──
    existing_summary: str           # 当前累积的对话摘要
    should_summarize: bool          # 是否触发摘要

    # ── 最终输出 ──
    answer: str
    sources: list[str]
    answer_mode: str                # general / rag / fallback / handoff / unsupported
    structured_output: dict
