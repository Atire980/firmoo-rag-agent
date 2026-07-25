"""
LangGraph StateGraph 组装：注册全部节点，添加固定边与条件边，使用 MemorySaver 编译。

路由函数保持纯函数，不调用 LLM。

流程：
    START → prepare_query → classify_query
      ├── GENERAL     → generate_general ────────┐
      ├── HANDOFF     → generate_handoff ────────┤
      ├── UNSUPPORTED → generate_unsupported ────┤
      ├── RAG+PRECISE → retrieve ────────────────┤
      ├── RAG+VAGUE   → hyde_generate → retrieve │
      └── RAG+BROAD   → multi_query_rewrite → retrieve
                                            retrieve
                                               ↓
                                       route_by_confidence
                                      ├── HIGH → generate_rag ──┐
                                      └── LOW  → generate_fallback
                              generate_general ──────────────────┘
                              generate_handoff ──────────────────┘
                              generate_unsupported ──────────────┘
                                               ↓
                                          save_memory → END
"""

from langgraph.graph import END, StateGraph

from app.agent.memory import get_memory_saver
from app.agent.nodes import (
    prepare_query_node,
    classify_query_node,
    hyde_generate_node,
    multi_query_rewrite_node,
    retrieve_node,
    generate_rag_node,
    generate_general_node,
    generate_handoff_node,
    generate_unsupported_node,
    generate_fallback_node,
    save_memory_node,
)
from app.agent.state import EcommerceState
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 编译后的图单例
_compiled_graph = None


# ═══════════════════════════════════════════════════════════════
# 路由函数（纯函数，不调用 LLM）
# ═══════════════════════════════════════════════════════════════

def route_after_classify(state: EcommerceState) -> str:
    """根据 route_action + retrieval_strategy 路由到下一个节点。

    Returns:
        下一个节点名称:
        - "generate_general"     → GENERAL
        - "generate_handoff"     → HANDOFF
        - "generate_unsupported" → UNSUPPORTED
        - "retrieve"             → RAG + PRECISE
        - "hyde_generate"        → RAG + VAGUE
        - "multi_query_rewrite"  → RAG + BROAD
    """
    route_action = state.get("route_action", "RAG")

    # 非 RAG 路由
    if route_action == "GENERAL":
        logger.info("路由: GENERAL → generate_general")
        return "generate_general"

    if route_action == "HANDOFF":
        logger.info("路由: HANDOFF → generate_handoff")
        return "generate_handoff"

    if route_action == "UNSUPPORTED":
        logger.info("路由: UNSUPPORTED → generate_unsupported")
        return "generate_unsupported"

    # RAG 路由：根据检索策略分发
    strategy = state.get("retrieval_strategy", "PRECISE")

    strategy_map = {
        "PRECISE": "retrieve",
        "VAGUE":   "hyde_generate",
        "BROAD":   "multi_query_rewrite",
    }

    next_node = strategy_map.get(strategy, "retrieve")
    logger.info("路由: RAG+%s → %s", strategy, next_node)
    return next_node


def route_after_retrieve(state: EcommerceState) -> str:
    """根据检索置信度路由到生成或兜底节点。

    Returns:
        "generate_rag" 或 "generate_fallback"。
    """
    is_high = state.get("is_high_confidence", False)
    next_node = "generate_rag" if is_high else "generate_fallback"
    logger.info("路由(检索后): high=%s → %s", is_high, next_node)
    return next_node


# ═══════════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """构建并编译 Ecommerce RAG Agent 的 StateGraph。

    Returns:
        编译后的 CompiledGraph 实例。
    """
    workflow = StateGraph(EcommerceState)

    # ── 注册节点 ──
    workflow.add_node("prepare_query", prepare_query_node)
    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("hyde_generate", hyde_generate_node)
    workflow.add_node("multi_query_rewrite", multi_query_rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_rag", generate_rag_node)
    workflow.add_node("generate_general", generate_general_node)
    workflow.add_node("generate_handoff", generate_handoff_node)
    workflow.add_node("generate_unsupported", generate_unsupported_node)
    workflow.add_node("generate_fallback", generate_fallback_node)
    workflow.add_node("save_memory", save_memory_node)

    # ── 边 ──

    # 入口
    workflow.set_entry_point("prepare_query")

    # prepare → classify（固定边）
    workflow.add_edge("prepare_query", "classify_query")

    # classify → 六条分支（条件边）
    workflow.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "generate_general": "generate_general",
            "generate_handoff": "generate_handoff",
            "generate_unsupported": "generate_unsupported",
            "retrieve": "retrieve",
            "hyde_generate": "hyde_generate",
            "multi_query_rewrite": "multi_query_rewrite",
        },
    )

    # VAGUE 改写后 → retrieve
    workflow.add_edge("hyde_generate", "retrieve")

    # BROAD 改写后 → retrieve
    workflow.add_edge("multi_query_rewrite", "retrieve")

    # retrieve → 置信度路由
    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "generate_rag": "generate_rag",
            "generate_fallback": "generate_fallback",
        },
    )

    # 所有生成节点 → save_memory → END
    workflow.add_edge("generate_rag", "save_memory")
    workflow.add_edge("generate_general", "save_memory")
    workflow.add_edge("generate_handoff", "save_memory")
    workflow.add_edge("generate_unsupported", "save_memory")
    workflow.add_edge("generate_fallback", "save_memory")

    workflow.add_edge("save_memory", END)

    # ── 编译 ──
    memory = get_memory_saver()
    compiled = workflow.compile(checkpointer=memory)

    logger.info("LangGraph 图编译完成")
    return compiled


def get_ecommerce_graph():
    """获取编译后的图单例。"""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
