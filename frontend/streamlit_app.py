"""
Firmoo 智能眼镜顾问 — Streamlit 聊天前端。

基于 LangGraph、混合检索和 Reranker 的眼镜电商智能客服。

启动方式：
    python -m streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from frontend.api_client import (
    API_BASE_URL,
    REQUEST_TIMEOUT,
    ApiClientError,
    health_check,
    send_chat_message,
)
from frontend.styles import GLOBAL_CSS

# ═══════════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Firmoo 智能眼镜顾问",
    page_icon="👓",
    layout="wide",
)

# 注入自定义 CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════

ANSWER_MODE_LABELS: dict[str, str] = {
    "rag": "知识库回答",
    "general": "通用回答",
    "handoff": "人工客服",
    "unsupported": "暂不支持",
    "fallback": "保守回答",
}

QUICK_QUESTIONS: list[str] = [
    "TR90 镜框有什么优点？",
    "600 度近视适合什么镜片？",
    "普通非定制眼镜可以退货吗？",
    "国际配送一般需要多久？",
    "优惠券应该怎么使用？",
    "我想转人工客服",
]

MAX_INPUT_LENGTH: int = 2000


# ═══════════════════════════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════════════════════════

def init_session_state() -> None:
    """首次访问时初始化所有 session_state 键。"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = "web-" + uuid.uuid4().hex[:8]

    if "session_id" not in st.session_state:
        st.session_state.session_id = "session-" + uuid.uuid4().hex[:12]

    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, Any]] = []

    if "last_response" not in st.session_state:
        st.session_state.last_response: dict[str, Any] | None = None

    if "service_status" not in st.session_state:
        st.session_state.service_status: str = "unknown"  # online | offline | unknown

    if "checking_health" not in st.session_state:
        st.session_state.checking_health: bool = False


init_session_state()


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def do_health_check() -> None:
    """调用健康检查并更新 service_status。"""
    st.session_state.checking_health = True
    try:
        result = health_check()
        if result.get("status") == "ok":
            st.session_state.service_status = "online"
        else:
            st.session_state.service_status = "offline"
    except ApiClientError:
        st.session_state.service_status = "offline"
    finally:
        st.session_state.checking_health = False


def start_new_session() -> None:
    """开始新会话：清空消息并重新生成 session_id。"""
    st.session_state.session_id = "session-" + uuid.uuid4().hex[:12]
    st.session_state.messages = []
    st.session_state.last_response = None


def deduplicate_sources(sources: list[str]) -> list[str]:
    """去除空值和重复来源，保持原顺序。"""
    seen: set[str] = set()
    result: list[str] = []
    for s in sources:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def format_confidence(confidence: float) -> str:
    """把 0~1 的置信度格式化为百分比字符串。"""
    return f"{confidence * 100:.2f}%"


def get_confidence_class(confidence: float) -> str:
    """根据置信度返回 CSS class 名称。"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-mid"
    return "confidence-low"


# ═══════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """渲染侧边栏：服务状态、会话信息、系统说明。"""
    with st.sidebar:
        st.markdown("## 👓 Firmoo RAG Agent")

        st.divider()

        # ---- 服务状态 ----
        st.markdown("### 🔌 服务状态")
        status = st.session_state.service_status

        if status == "online":
            dot_html = '<span class="status-dot online"></span>'
            st.markdown(f"{dot_html} **在线** — 后端正常", unsafe_allow_html=True)
        elif status == "offline":
            dot_html = '<span class="status-dot offline"></span>'
            st.markdown(f"{dot_html} **离线** — 无法连接", unsafe_allow_html=True)
        else:
            dot_html = '<span class="status-dot unknown"></span>'
            st.markdown(f"{dot_html} **未知** — 尚未检查", unsafe_allow_html=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("🔄 检查连接", use_container_width=True):
                do_health_check()
                st.rerun()
        with col_b:
            if st.button("🆕 新会话", use_container_width=True):
                start_new_session()
                st.rerun()

        st.divider()

        # ---- 后端地址 ----
        st.markdown("### 📡 后端地址")
        st.code(API_BASE_URL, language=None)

        # ---- 当前会话 ----
        st.markdown("### 🆔 当前会话")
        sid = st.session_state.session_id
        st.caption(f"{sid[:20]}...")
        st.caption(f"用户: {st.session_state.user_id}")

        st.divider()

        # ---- 系统能力 ----
        st.markdown("### 🛠 系统能力")
        capabilities: list[str] = [
            "💎 商品材质、颜色、款式",
            "🔍 镜片、度数、折射率",
            "🚚 物流及配送政策",
            "🔄 退换货及售后",
            "🎫 优惠券与支付",
            "👤 人工客服引导",
        ]
        for cap in capabilities:
            st.markdown(f"- {cap}")

        st.divider()

        # ---- 注意事项 ----
        st.markdown(
            """<div class="note-box">
            <strong>⚠️ 注意事项</strong><br>
            当前系统不连接真实订单数据库，无法查询、取消或修改个人订单。
            请勿输入密码、银行卡号等敏感信息。
            </div>""",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# 欢迎区域
# ═══════════════════════════════════════════════════════════════

def render_welcome() -> None:
    """首屏欢迎信息 + 快捷问题按钮。"""
    st.markdown(
        """
        <div class="welcome-card">
            <h2>👓 您好，我是 Firmoo 智能眼镜顾问</h2>
            <p>
                基于 <strong>LangGraph</strong>、混合检索和 <strong>Reranker</strong> 的眼镜电商智能客服<br>
                您可以向我咨询镜框、镜片、度数、配送、退换货和优惠支付等问题
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 快捷问题按钮
    st.markdown("##### 💬 试试这些问题：")

    cols = st.columns(3)
    for i, question in enumerate(QUICK_QUESTIONS):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                # 仅标记待发送消息，由底部逻辑统一处理
                st.session_state._pending_message = question
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# 聊天消息渲染
# ═══════════════════════════════════════════════════════════════

def render_message(msg: dict[str, Any]) -> None:
    """渲染单条聊天消息（用户或助手）。"""
    role: str = msg.get("role", "user")
    content: str = msg.get("content", "")
    meta: dict[str, Any] | None = msg.get("metadata")

    with st.chat_message(role):
        st.markdown(content)

        # 仅助手消息展示元信息
        if role == "assistant" and meta:
            render_message_meta(meta)


def render_message_meta(meta: dict[str, Any]) -> None:
    """在助手消息下方展示业务意图、回答模式、置信度等。"""
    parts: list[str] = []

    # 业务意图
    intent: str = meta.get("business_intent", "")
    if intent:
        parts.append(f"📌 业务意图：{intent}")

    # 回答模式
    mode: str = meta.get("answer_mode", "")
    if mode:
        label = ANSWER_MODE_LABELS.get(mode, mode)
        parts.append(f"📋 回答模式：{label}")

    # 置信度
    confidence: float = float(meta.get("confidence", 0))
    if confidence > 0:
        pct = format_confidence(confidence)
        css_cls = get_confidence_class(confidence)
        parts.append(
            f'<span class="meta-tag {css_cls}">📊 置信度：{pct}</span>'
        )

    # 来源数量
    sources: list[str] = meta.get("sources", [])
    if sources:
        parts.append(f"📚 参考来源：{len(sources)} 条")

    if parts:
        html = '<div class="meta-row">' + "".join(
            f'<span class="meta-tag">{p}</span>' for p in parts
        ) + "</div>"
        st.markdown(html, unsafe_allow_html=True)

    # 来源展开
    sources = deduplicate_sources(sources)
    if sources:
        with st.expander("📖 查看参考来源"):
            for i, src in enumerate(sources, 1):
                st.markdown(f"{i}. `{src}`")


# ═══════════════════════════════════════════════════════════════
# 消息发送逻辑
# ═══════════════════════════════════════════════════════════════

def handle_user_message(user_input: str) -> None:
    """处理用户消息：调用后端并更新消息列表。"""
    user_input = user_input.strip()

    # 空消息不做任何事
    if not user_input:
        return

    # 长度限制
    if len(user_input) > MAX_INPUT_LENGTH:
        st.warning(f"消息过长（{len(user_input)}/{MAX_INPUT_LENGTH} 字符），已截断。")
        user_input = user_input[:MAX_INPUT_LENGTH]

    # 添加用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "metadata": None,
    })

    # 调用后端
    with st.spinner("正在分析问题并检索知识库……"):
        try:
            data = send_chat_message(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                message=user_input,
            )

            answer = data.get("answer", "")
            if not answer:
                answer = "⚠️ 抱歉，系统未能生成有效回复，请稍后重试或联系人工客服。"

            meta = {
                "business_intent": data.get("business_intent", ""),
                "answer_mode": data.get("answer_mode", "general"),
                "confidence": float(data.get("confidence", 0)),
                "sources": data.get("sources", []),
            }

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "metadata": meta,
            })
            st.session_state.last_response = data
            st.session_state.service_status = "online"

        except ApiClientError as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"❌ **{e}**\n\n"
                    "请确认后端已启动：\n\n"
                    "```bash\n"
                    "python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000\n"
                    "```"
                ),
                "metadata": None,
            })
            st.session_state.service_status = "offline"


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """Streamlit 应用主入口。"""

    # ---- 页面标题 ----
    st.title("👓 Firmoo 智能眼镜顾问")
    st.caption("基于 LangGraph、混合检索和 Reranker 的眼镜电商智能客服")

    # ---- 首次加载时健康检查 ----
    if st.session_state.service_status == "unknown":
        do_health_check()

    # ---- 侧边栏 ----
    render_sidebar()

    # ---- 主区域 ----
    messages: list[dict[str, Any]] = st.session_state.messages

    if not messages:
        render_welcome()

    # 渲染历史消息
    for msg in messages:
        render_message(msg)

    # ---- 聊天输入 ----
    # 需要用一个标记来区分 "快捷按钮触发" 和 "用户手动输入"
    # 快捷按钮会设置 _pending_message，这里先检查它
    pending: str | None = st.session_state.pop("_pending_message", None)

    user_input = st.chat_input(
        "请输入您的问题……",
        max_chars=MAX_INPUT_LENGTH,
    )

    # 最终要处理的消息：优先快捷按钮 > 手动输入
    final_input = pending or user_input

    if final_input:
        # 防止重复提交：如果最后一条用户消息就是这条，跳过
        last_msg = messages[-1] if messages else None
        if last_msg and last_msg["role"] == "user" and last_msg["content"] == final_input:
            return

        handle_user_message(final_input)
        st.rerun()


if __name__ == "__main__":
    main()
