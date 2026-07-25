"""Chat API 的请求 / 响应 Pydantic 模型。"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。"""
    user_id: str = Field(..., description="用户唯一标识")
    session_id: str = Field(..., description="会话唯一标识")
    message: str = Field(..., description="用户当前输入消息")


class SourceItem(BaseModel):
    """单条来源信息。"""
    source_name: str = Field(default="", description="来源名称，如 rag_knowledge_base.csv")
    content_snippet: str = Field(default="", description="引用内容片段")


class ChatResponse(BaseModel):
    """聊天响应体。"""
    answer: str = Field(default="", description="Agent 最终回答")
    sources: list[str] = Field(default_factory=list, description="引用的来源标识列表")
    confidence: float = Field(default=0.0, description="检索置信度分数，0.0 ~ 1.0")
    answer_mode: str = Field(
        default="general",
        description="回答模式：general / rag / fallback / handoff / unsupported",
    )
    business_intent: str = Field(
        default="",
        description="业务意图：商品咨询/镜片度数/物流查询/退换货售后/订单修改取消/投诉人工/闲聊其他/优惠支付",
    )
