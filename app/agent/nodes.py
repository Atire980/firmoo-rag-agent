"""
LangGraph 节点实现：每个节点函数接收 EcommerceState，返回部分 State 更新字典。

节点只返回 State 更新，不原地修改 State。
不在节点中重复实现 Milvus、Embedding 或模型加载。

两层分类：
    第一层：business_intent   → 八种业务意图（MiniLM 八分类）
    第二层：retrieval_strategy → PRECISE / VAGUE / BROAD（仅 RAG 路径）

路由：
    GENERAL     → generate_general
    HANDOFF     → generate_handoff
    UNSUPPORTED → generate_unsupported
    RAG         → retrieve（先经 PRECISE/VAGUE/BROAD 策略）
"""

import asyncio
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.memory import should_trigger_summary, get_recent_messages
from app.agent.prompts import (
    RAG_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    FALLBACK_SYSTEM_PROMPT,
    HANDOFF_SYSTEM_PROMPT,
    UNSUPPORTED_SYSTEM_PROMPT,
    HYDE_PROMPT,
    MULTI_QUERY_PROMPT,
    SUMMARY_PROMPT,
    format_context_from_chunks,
)
from app.agent.state import EcommerceState
from app.classifier.rules import apply_rules
from app.config import settings, INTENT_TO_ROUTE, INTENT_TO_SOURCE_TYPE
from app.llm.factory import get_llm
from app.rag.retriever import retrieve, retrieve_multi
from app.rag.reranker import rerank
from app.utils.logger import get_logger
from app.utils.text import safe_extract_content, format_sources

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _derive_route_action(
    business_intent: str,
    is_specific_order: bool,
) -> str:
    """根据业务意图和具体订单标识推导路由动作。

    Args:
        business_intent:   八种业务意图标签。
        is_specific_order: 是否涉及具体订单号/运单号。

    Returns:
        "GENERAL" / "HANDOFF" / "UNSUPPORTED" / "RAG"
    """
    # 闲聊/其他 → GENERAL
    if business_intent == "闲聊/其他":
        return "GENERAL"

    # 投诉/人工客服 → HANDOFF
    if business_intent == "投诉/人工客服":
        return "HANDOFF"

    # 物流查询 + 具体单号 → UNSUPPORTED
    if business_intent == "物流查询" and is_specific_order:
        logger.info("物流查询 → 检测到具体运单号/订单号，路由到 UNSUPPORTED")
        return "UNSUPPORTED"

    # 订单修改/取消 + 具体订单操作 → UNSUPPORTED
    if business_intent == "订单修改/取消" and is_specific_order:
        logger.info("订单修改/取消 → 检测到具体订单操作请求，路由到 UNSUPPORTED")
        return "UNSUPPORTED"

    # 默认进入 RAG
    return "RAG"


def _determine_retrieval_strategy(
    query: str,
    rule_strategy: str,
) -> str:
    """确定检索策略（第二层分类）。

    优先使用规则的 VAGUE/BROAD 提示，覆盖部分指代词和开放性问题。
    默认 PRECISE。

    Args:
        query:         用户原始查询。
        rule_strategy: 规则返回的检索策略提示。

    Returns:
        "PRECISE" / "VAGUE" / "BROAD"
    """
    if rule_strategy in ("VAGUE", "BROAD"):
        return rule_strategy

    # 规则未给出策略时的追加判断
    # 含指代词 → VAGUE 候选
    if re.search(
        r"(它|这个|那个|这款|那款|刚才|之前|上面|前面)"
        r".*(怎么样|如何|呢|吗|什么|多少|有哪些|有没有|还有别的)",
        query,
    ):
        return "VAGUE"

    # 开放性问题模式 → BROAD 候选
    if re.search(
        r"(都有哪些|怎么选|如何选择|怎样挑选|有什么区别|有什么不同|全面介绍|对比)",
        query,
    ):
        return "BROAD"

    return "PRECISE"


# ═══════════════════════════════════════════════════════════════
# 1. prepare_query_node
# ═══════════════════════════════════════════════════════════════

async def prepare_query_node(state: EcommerceState) -> dict[str, Any]:
    """从最后一条 HumanMessage 提取 original_query，初始化本轮临时字段。"""
    messages = state.get("messages", [])
    query = ""

    # 找到最后一条用户消息
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = safe_extract_content(msg)
            break

    if not query:
        logger.warning("未找到用户消息，使用空查询")
        query = ""

    logger.info("prepare_query: %s", query[:80])

    return {
        "original_query": query,
        # 初始化本轮临时字段
        "business_intent": "",
        "business_intent_confidence": 0.0,
        "route_action": "",
        "retrieval_strategy": "",
        "hyde_document": "",
        "rewritten_queries": [],
        "ranked_chunks": [],
        "confidence": 0.0,
        "is_high_confidence": False,
        "answer": "",
        "sources": [],
        "answer_mode": "",
    }


# ═══════════════════════════════════════════════════════════════
# 2. classify_query_node
# ═══════════════════════════════════════════════════════════════

async def classify_query_node(state: EcommerceState) -> dict[str, Any]:
    """两层分类节点。

    第一层：确定 business_intent（规则优先 → 八分类模型补充）。
    第二层：RAG 路径下确定 retrieval_strategy。
    推导 route_action 用于图路由。

    Returns:
        {
            business_intent, business_intent_confidence,
            route_action, retrieval_strategy
        }
    """
    query = state.get("original_query", "")
    if not query:
        return {
            "business_intent": "闲聊/其他",
            "business_intent_confidence": 1.0,
            "route_action": "GENERAL",
            "retrieval_strategy": "",
        }

    # ── Step 1: 规则先行 ──
    rule_result = apply_rules(query)
    rule_intent = rule_result["business_intent"]
    is_specific_order = rule_result["is_specific_order"]
    rule_strategy = rule_result["retrieval_strategy"]

    logger.info(
        "规则结果: intent=%s, specific_order=%s, strategy=%s",
        rule_intent or "(未确定)", is_specific_order, rule_strategy or "(未确定)",
    )

    # ── Step 2: 确定 business_intent ──
    business_intent = ""
    confidence = 0.0

    if rule_intent:
        # 规则命中 → 直接使用
        business_intent = rule_intent
        confidence = 0.85  # 规则置信度略低于模型但足够高
        logger.info("规则命中: business_intent=%s", business_intent)
    else:
        # 规则未命中 → 调用八分类模型
        logger.info("规则未命中，调用八分类模型...")
        try:
            from app.classifier.intent_classifier import async_classify
            business_intent, confidence = await async_classify(query)
            logger.info(
                "模型分类: business_intent=%s, confidence=%.3f",
                business_intent, confidence,
            )
        except Exception as e:
            logger.error("八分类模型失败: %s，默认 商品咨询", e)
            business_intent = "商品咨询"
            confidence = 0.5

    # ── Step 3: 推导 route_action ──
    route_action = _derive_route_action(business_intent, is_specific_order)

    # ── Step 4: 确定 retrieval_strategy（仅 RAG 路径） ──
    retrieval_strategy = ""
    if route_action == "RAG":
        retrieval_strategy = _determine_retrieval_strategy(query, rule_strategy)

    logger.info(
        "分类完成: intent=%s → route=%s → strategy=%s (confidence=%.3f)",
        business_intent, route_action, retrieval_strategy or "N/A", confidence,
    )

    return {
        "business_intent": business_intent,
        "business_intent_confidence": confidence,
        "route_action": route_action,
        "retrieval_strategy": retrieval_strategy,
    }


# ═══════════════════════════════════════════════════════════════
# 3. hyde_generate_node (VAGUE 路径)
# ═══════════════════════════════════════════════════════════════

async def hyde_generate_node(state: EcommerceState) -> dict[str, Any]:
    """VAGUE 路径：结合历史对话生成 HyDE 假设文档。

    HyDE 文档只用于后续检索，不是最终答案。
    """
    query = state.get("original_query", "")
    messages = state.get("messages", [])

    # 提取最近历史
    history_parts = []
    recent = get_recent_messages(messages, window_size=8)
    for msg in recent[:-1]:  # 排除最后一条（当前 query）
        role = "用户" if isinstance(msg, HumanMessage) else "客服"
        content = safe_extract_content(msg)
        if content:
            history_parts.append(f"{role}: {content}")

    history_text = "\n".join(history_parts) if history_parts else "（无历史对话）"

    prompt = HYDE_PROMPT.format(history=history_text, query=query)

    try:
        llm = get_llm("rewrite")
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        hyde_doc = safe_extract_content(response).strip()
        logger.info("HyDE 生成: %s", hyde_doc[:100])
        return {"hyde_document": hyde_doc}
    except Exception as e:
        logger.error("HyDE 生成失败: %s，回退为原查询", e)
        return {"hyde_document": query}


# ═══════════════════════════════════════════════════════════════
# 4. multi_query_rewrite_node (BROAD 路径)
# ═══════════════════════════════════════════════════════════════

async def multi_query_rewrite_node(state: EcommerceState) -> dict[str, Any]:
    """BROAD 路径：将范围过大的问题拆分为 3～5 个具体子问题。"""
    query = state.get("original_query", "")

    prompt = MULTI_QUERY_PROMPT.format(query=query)

    try:
        llm = get_llm("rewrite")
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = safe_extract_content(response).strip()

        # 解析：每行一个子问题，去除编号前缀
        queries = []
        for line in content.split("\n"):
            line = re.sub(r"^[\d]+[\.\)、\s]+", "", line).strip()
            if line and len(line) > 2:
                queries.append(line)

        if len(queries) < 2:
            logger.warning("Multi-Query 拆分结果不足 2 个，回退原查询")
            queries = [query]
        else:
            queries = queries[:5]  # 最多 5 个

        logger.info("Multi-Query 拆分: %d 个子查询 → %s", len(queries), queries)
        return {"rewritten_queries": queries}

    except Exception as e:
        logger.error("Multi-Query 拆分失败: %s，回退原查询", e)
        return {"rewritten_queries": [query]}


# ═══════════════════════════════════════════════════════════════
# 5. retrieve_node
# ═══════════════════════════════════════════════════════════════

async def retrieve_node(state: EcommerceState) -> dict[str, Any]:
    """统一检索节点：根据检索策略选择不同检索方式，并传入 source_type 过滤。

    - PRECISE: 原问题直接检索
    - VAGUE:   用 HyDE 文档检索
    - BROAD:   Multi-Query 拆分后并行检索并合并去重

    使用 business_intent → source_type 映射作为辅助过滤条件。
    """
    strategy = state.get("retrieval_strategy", "PRECISE")
    original_query = state.get("original_query", "")
    business_intent = state.get("business_intent", "")

    # ── 确定 source_type 过滤 ──
    source_types = INTENT_TO_SOURCE_TYPE.get(business_intent, [])
    # v1 不强制过滤：source_types 为空时不传过滤条件，搜索全库
    # 后续可根据需求改为传入 source_types

    # ── 确定检索文本 ──
    if strategy == "VAGUE":
        search_query = state.get("hyde_document", original_query)
        logger.info("VAGUE 检索: 使用 HyDE 文档")
    else:
        search_query = original_query

    # ── 执行检索 ──
    try:
        if strategy == "BROAD":
            queries = state.get("rewritten_queries", [original_query])
            candidates = await asyncio.to_thread(
                retrieve_multi, queries, settings.RETRIEVAL_TOP_K
            )
            logger.info("BROAD 检索: Multi-Query (%d 个子查询)", len(queries))
        else:
            candidates = await asyncio.to_thread(
                retrieve, search_query, settings.RETRIEVAL_TOP_K
            )
            logger.info("%s 检索: 使用单查询", strategy)
    except Exception as e:
        logger.error("检索失败: %s", e)
        return {
            "ranked_chunks": [],
            "confidence": 0.0,
            "is_high_confidence": False,
        }

    if not candidates:
        logger.warning("检索无结果")
        return {
            "ranked_chunks": [],
            "confidence": 0.0,
            "is_high_confidence": False,
        }

    # ── Rerank ──
    try:
        rerank_query = original_query  # 始终用原始用户问题做重排
        ranked = await asyncio.to_thread(
            rerank, rerank_query, candidates, settings.RERANK_TOP_K
        )
    except Exception as e:
        logger.error("重排失败: %s，使用原始检索结果", e)
        ranked = sorted(candidates, key=lambda d: d.get("score", 0), reverse=True)
        ranked = ranked[:settings.RERANK_TOP_K]

    # ── 计算置信度 ──
    if ranked:
        top_score = ranked[0].get("rerank_score", ranked[0].get("score", 0))
        confidence = min(top_score, 1.0)
        if confidence > 1.0:
            confidence = 1.0 / (1.0 + abs(confidence - 1.0))
    else:
        confidence = 0.0

    is_high = confidence >= settings.RAG_CONFIDENCE_THRESHOLD

    logger.info(
        "检索完成: strategy=%s, 候选=%d, 重排=%d, confidence=%.3f, high=%s",
        strategy, len(candidates), len(ranked), confidence, is_high,
    )

    return {
        "ranked_chunks": ranked,
        "confidence": confidence,
        "is_high_confidence": is_high,
    }


# ═══════════════════════════════════════════════════════════════
# 6. generate_rag_node
# ═══════════════════════════════════════════════════════════════

async def generate_rag_node(state: EcommerceState) -> dict[str, Any]:
    """高置信度 RAG 回答：使用精排后的父块内容生成回答。"""
    ranked_chunks = state.get("ranked_chunks", [])
    original_query = state.get("original_query", "")
    existing_summary = state.get("existing_summary", "") or "（无历史摘要）"
    messages = state.get("messages", [])
    business_intent = state.get("business_intent", "商品咨询")

    context = format_context_from_chunks(ranked_chunks)

    system_prompt = RAG_SYSTEM_PROMPT.format(
        summary=existing_summary,
        context=context,
        business_intent=business_intent,
    )

    # 收集来源
    sources: list[str] = []
    for chunk in ranked_chunks:
        name = chunk.get("source_name", "")
        if name and name not in sources:
            sources.append(name)

    try:
        llm = get_llm("qa")

        llm_messages = [SystemMessage(content=system_prompt)]

        recent = get_recent_messages(messages, window_size=8)
        for msg in recent[:-1]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                llm_messages.append(msg)

        llm_messages.append(HumanMessage(content=original_query))

        response = await llm.ainvoke(llm_messages)
        answer = safe_extract_content(response).strip()

        source_text = format_sources(sources)
        if source_text and source_text not in answer:
            answer += source_text

        logger.info("RAG 回答生成: %d 字符", len(answer))

        return {
            "answer": answer,
            "sources": sources,
            "answer_mode": "rag",
        }

    except Exception as e:
        logger.error("RAG 回答生成失败: %s", e)
        return {
            "answer": "抱歉，生成回答时出现了问题，请稍后重试或联系人工客服。",
            "sources": sources,
            "answer_mode": "rag",
        }


# ═══════════════════════════════════════════════════════════════
# 7. generate_general_node
# ═══════════════════════════════════════════════════════════════

async def generate_general_node(state: EcommerceState) -> dict[str, Any]:
    """GENERAL 回答：问候、闲聊与通用问题，不查询知识库。"""
    original_query = state.get("original_query", "")
    existing_summary = state.get("existing_summary", "") or "（无历史摘要）"
    messages = state.get("messages", [])

    system_prompt = GENERAL_SYSTEM_PROMPT.format(summary=existing_summary)

    try:
        llm = get_llm("qa")

        llm_messages = [SystemMessage(content=system_prompt)]

        recent = get_recent_messages(messages, window_size=8)
        for msg in recent[:-1]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                llm_messages.append(msg)

        llm_messages.append(HumanMessage(content=original_query))

        response = await llm.ainvoke(llm_messages)
        answer = safe_extract_content(response).strip()

        logger.info("GENERAL 回答生成: %d 字符", len(answer))

        return {
            "answer": answer,
            "sources": [],
            "answer_mode": "general",
        }

    except Exception as e:
        logger.error("GENERAL 回答生成失败: %s", e)
        return {
            "answer": "你好！我是 Firmoo 眼镜客服助手，有什么可以帮你的吗？",
            "sources": [],
            "answer_mode": "general",
        }


# ═══════════════════════════════════════════════════════════════
# 8. generate_handoff_node
# ═══════════════════════════════════════════════════════════════

async def generate_handoff_node(state: EcommerceState) -> dict[str, Any]:
    """HANDOFF 回答：投诉、不满或要求转人工客服。"""
    original_query = state.get("original_query", "")
    existing_summary = state.get("existing_summary", "") or "（无历史摘要）"

    system_prompt = HANDOFF_SYSTEM_PROMPT.format(
        summary=existing_summary,
        query=original_query,
    )

    try:
        llm = get_llm("qa")
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=original_query),
        ])
        answer = safe_extract_content(response).strip()

        logger.info("HANDOFF 回答生成: %d 字符", len(answer))

        return {
            "answer": answer,
            "sources": [],
            "answer_mode": "handoff",
        }

    except Exception as e:
        logger.error("HANDOFF 回答生成失败: %s", e)
        return {
            "answer": (
                "非常抱歉给您带来不便，我已经记录了您的反馈。\n\n"
                "为了更快地解决您的问题，建议您联系人工客服，"
                "他们会为您优先处理。您可以在官网或 App 中找到在线客服入口。"
            ),
            "sources": [],
            "answer_mode": "handoff",
        }


# ═══════════════════════════════════════════════════════════════
# 9. generate_unsupported_node
# ═══════════════════════════════════════════════════════════════

async def generate_unsupported_node(state: EcommerceState) -> dict[str, Any]:
    """UNSUPPORTED 回答：具体订单/运单操作，需要订单系统支持。

    尝试先做一次轻量 RAG 检索，获取通用政策供参考。
    """
    original_query = state.get("original_query", "")
    existing_summary = state.get("existing_summary", "") or "（无历史摘要）"

    # 尝试检索通用政策知识（source_type=policy）
    try:
        candidates = await asyncio.to_thread(
            retrieve, original_query, settings.RETRIEVAL_TOP_K,
        )
        if candidates:
            ranked = await asyncio.to_thread(
                rerank, original_query, candidates, 3,  # 只取 top-3
            )
            policy_context = "\n\n".join(
                f"- {c.get('parent_content', c.get('child_content', ''))[:200]}"
                for c in ranked[:3]
            )
        else:
            policy_context = "（暂无通用政策资料）"
    except Exception:
        policy_context = "（检索政策资料时出现问题）"

    system_prompt = UNSUPPORTED_SYSTEM_PROMPT.format(
        summary=existing_summary,
        query=original_query,
    ) + f"\n{policy_context}"

    try:
        llm = get_llm("qa")
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=original_query),
        ])
        answer = safe_extract_content(response).strip()

        logger.info("UNSUPPORTED 回答生成: %d 字符", len(answer))

        return {
            "answer": answer,
            "sources": [],
            "answer_mode": "unsupported",
        }

    except Exception as e:
        logger.error("UNSUPPORTED 回答生成失败: %s", e)
        return {
            "answer": (
                "抱歉，我暂时无法直接查询或操作具体订单，这需要访问订单系统。\n\n"
                "您可以：\n"
                "1. 通过官网「我的订单」页面查看订单状态和物流信息\n"
                "2. 联系人工客服，他们会帮您处理具体的订单操作\n"
                "3. 如需了解退换货政策，我可以为您查询通用规则"
            ),
            "sources": [],
            "answer_mode": "unsupported",
        }


# ═══════════════════════════════════════════════════════════════
# 10. generate_fallback_node
# ═══════════════════════════════════════════════════════════════

async def generate_fallback_node(state: EcommerceState) -> dict[str, Any]:
    """低置信度兜底：不编造商品、价格、库存、物流与售后信息。"""
    original_query = state.get("original_query", "")
    existing_summary = state.get("existing_summary", "") or "（无历史摘要）"
    messages = state.get("messages", [])
    business_intent = state.get("business_intent", "商品咨询")

    system_prompt = FALLBACK_SYSTEM_PROMPT.format(
        summary=existing_summary,
        business_intent=business_intent,
    )

    try:
        llm = get_llm("qa")

        llm_messages = [SystemMessage(content=system_prompt)]

        recent = get_recent_messages(messages, window_size=8)
        for msg in recent[:-1]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                llm_messages.append(msg)

        llm_messages.append(HumanMessage(content=original_query))

        response = await llm.ainvoke(llm_messages)
        answer = safe_extract_content(response).strip()

        logger.info("Fallback 回答生成: %d 字符", len(answer))

        return {
            "answer": answer,
            "sources": [],
            "answer_mode": "fallback",
        }

    except Exception as e:
        logger.error("Fallback 回答生成失败: %s", e)
        return {
            "answer": (
                "抱歉，我暂时没有找到足够可靠的资料来回答您的问题。\n\n"
                "建议您：\n"
                "1. 提供更具体的商品名称或 SKU 编号\n"
                "2. 描述更详细的疑问\n"
                "3. 联系人工客服获取进一步帮助"
            ),
            "sources": [],
            "answer_mode": "fallback",
        }


# ═══════════════════════════════════════════════════════════════
# 11. save_memory_node
# ═══════════════════════════════════════════════════════════════

async def save_memory_node(state: EcommerceState) -> dict[str, Any]:
    """保存记忆节点：判断是否触发摘要，更新 existing_summary。"""
    messages = state.get("messages", [])
    existing_summary = state.get("existing_summary", "") or ""

    message_count = len(messages)

    need_summary = should_trigger_summary(message_count)
    if not need_summary:
        return {"should_summarize": False}

    logger.info("触发对话摘要 (消息数: %d)", message_count)

    history_parts = []
    for msg in messages[-settings.SUMMARY_TRIGGER_ROUNDS * 2:]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "用户" if isinstance(msg, HumanMessage) else "客服"
            content = safe_extract_content(msg)
            if content:
                history_parts.append(f"{role}: {content}")

    history_text = "\n".join(history_parts)

    prompt = SUMMARY_PROMPT.format(history=history_text)

    try:
        llm = get_llm("summary")
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        new_summary = safe_extract_content(response).strip()

        if existing_summary:
            combined = f"{existing_summary}\n{new_summary}"
        else:
            combined = new_summary

        if len(combined) > 500:
            combined = combined[:500]

        logger.info("摘要更新: %d 字符", len(combined))

        return {
            "existing_summary": combined,
            "should_summarize": False,
        }

    except Exception as e:
        logger.error("摘要生成失败: %s", e)
        return {"should_summarize": False}
