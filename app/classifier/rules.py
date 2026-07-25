"""
轻量关键词规则：作为八分类的快速通道，模型无法确定时回退到规则。

规则返回 dict：
    {
        "business_intent": str | "",     # 八种标签之一，"" 表示规则无法确定
        "is_specific_order": bool,       # 是否涉及具体订单/运单号操作
        "retrieval_strategy": str | "",   # 检索策略提示："VAGUE" / "BROAD" / ""
    }

两层分类关系：
    第一层：business_intent → 八种业务意图
    第二层：retrieval_strategy → 仅 RAG 路径使用

规则不直接决定 route_action；route_action 由 classify_query_node 综合
business_intent + is_specific_order 推导。
"""

import re

# ═══════════════════════════════════════════════════════════════
# 闲聊/其他 → GENERAL 路由
# ═══════════════════════════════════════════════════════════════

GENERAL_PATTERNS = [
    (r"^(你好|您好|hi|hello|hey)[\s!！。.,，]*$", 0.95),
    (r"^(谢谢|感谢|多谢|thanks|thank you|thx)[\s!！。.,，]*$", 0.95),
    (r"^(再见|拜拜|bye|goodbye|88)[\s!！。.,，]*$", 0.95),
    (r"^(早上好|下午好|晚上好|早安|晚安|good morning|good afternoon|good evening)[\s!！。.,，]*$", 0.95),
    (r"^(你是谁|你叫什么|你是什么|你能做什么|介绍一下自己)", 0.90),
    (r"(今天天气|讲个笑话|聊天|无聊|随便看看|今天有什么推荐)", 0.80),
]

# ═══════════════════════════════════════════════════════════════
# 投诉/人工客服 → HANDOFF 路由
# ═══════════════════════════════════════════════════════════════

COMPLAINT_KEYWORDS = [
    "投诉",
    "人工客服",
    "转人工",
    "找人工",
    "不满意",
    "升级处理",
    "客服不回复",
    "客服一直不回复",
    "一直不回复",
    "客服不理",
    "没人回复",
    "投诉你们",
    "我要投诉",
]


# ═══════════════════════════════════════════════════════════════
# 八种业务意图的关键词映射
# ═══════════════════════════════════════════════════════════════

INTENT_KEYWORD_MAP: dict[str, list[str]] = {
    "商品咨询": [
        # 镜框/眼镜本体
        "镜框", "镜架", "眼镜", "镜腿", "鼻梁", "鼻托", "螺丝",
        "材质", "TR90", "板材", "金属", "塑料", "醋酸纤维", "钛",
        "颜色", "黑色", "银色", "金色", "玳瑁", "透明", "粉色",
        "silver", "black", "gold", "transparent", "pink",
        "款式", "全框", "半框", "无框", "圆形", "方形", "矩形", "猫眼", "飞行员",
        "重量", "尺码", "大小", "总宽", "镜片宽度", "镜腿长度",
        # 脸型/佩戴
        "脸型", "圆脸", "方脸", "长脸", "瓜子脸", "椭圆脸",
        "夹头", "夹脸", "压鼻梁", "下滑", "显脸大", "显脸小",
        "适合男生", "适合女生", "适合日常", "推荐",
        # SKU 模式
        "FM", "FR",
    ],
    "镜片/度数咨询": [
        "近视", "远视", "散光", "老花", "度数", "瞳距", "PD",
        "蓝光", "防蓝光", "防紫外线", "UV", "渐进", "双光", "单光",
        "折射率", "高折射", "镀膜", "变色", "偏光",
        "可以配吗", "怎么填写", "怎么选镜片", "什么镜片",
    ],
    "物流查询": [
        "物流", "快递", "配送", "发货", "到货", "运输",
        "几天到", "多久到", "多久能到", "什么时候到", "什么时候发",
        "物流单号", "运单号", "快递单号", "查物流",
        "加急配送", "加急发货", "什么时候发货",
    ],
    "退换货售后": [
        "退货", "换货", "退款", "退换", "保修", "售后", "维修",
        "退运险", "退货运费", "运费险",
        "超过七天", "免费重做", "镜片有划痕", "镜腿松",
        "不舒服可以退", "戴了两天", "可以退吗", "可以换吗",
    ],
    "订单修改/取消": [
        "订单", "下单", "取消订单", "修改订单", "取消",
        "改地址", "修改地址", "地址填错", "地址错了",
        "颜色选错", "度数填错", "填错度数",
        "进入加工", "还能取消", "可以取消",
    ],
    "优惠/支付": [
        "优惠券", "折扣", "打折", "优惠码", "促销", "活动",
        "支付", "付款", "PayPal", "信用卡", "支付失败",
        "可以优惠", "有优惠", "有折扣", "怎么付款", "支持什么支付",
        "价格", "多少钱", "怎么卖",
    ],

}
ORDER_MODIFY_PRIORITY_KEYWORDS = [
    "颜色选错",
    "颜色错了",
    "度数填错",
    "填错度数",
    "地址填错",
    "地址错了",
    "修改订单",
    "取消订单",
    "改订单",
]



AFTER_SALES_PRIORITY_KEYWORDS = [
    "退换货",
    "退货",
    "退款",
    "换货",
    "申请退货",
    "申请退款",
    "申请换货",
    "可以退",
    "能退吗",
    "售后",
]
# ═══════════════════════════════════════════════════════════════
# 具体订单/运单号识别（用于区分 UNSUPPORTED）
# ═══════════════════════════════════════════════════════════════

SPECIFIC_ORDER_PATTERNS = [
    # 有明确订单号
    r"\bOD\d{4,}\b",
    # 指向用户自己的订单
    r"我的订单",
    r"我的快递",
    r"我的物流",
    # 明确要求系统执行操作
    r"帮我取消",
    r"帮我退",
    r"帮我改",
    r"帮我修改",
    r"我要取消",
    r"我要退",
    r"我要改",
    r"我要修改",
    # 明确指向某一订单
    r"取消(?:这个|该|我的).*订单",
    r"修改(?:这个|该|我的).*订单",
    r"退(?:这个|该|我的).*订单",
    # 具体地址操作
    r"修改收货地址",
    r"改收货地址",
    r"改配送地址",
    r"修改.*地址",
    r"更改.*地址",
    # 具体物流查询
    r"查.*物流单号",
    r"查.*运单号",
    r"查.*快递单号",
    r"物流单号.*查",
    r"运单号.*查",
    r"我的.*到哪里",
    r"我的.*物流",
    r"我的.*快递",
    r"查我的物流",
    r"查我的快递",
    r"(?:查|查询)?(?:一下)?(?:我的)?(?:快递|物流).*(?:到哪|到哪里|到什么位置)",
]
# 预编译
_compiled_specific = [
    re.compile(
        pattern,
        re.IGNORECASE | re.ASCII,
    )
    for pattern in SPECIFIC_ORDER_PATTERNS
]


# ═══════════════════════════════════════════════════════════════
# 检索策略提示规则（VAGUE / BROAD）
# ═══════════════════════════════════════════════════════════════

VAGUE_PATTERNS = [
    r"(它|这个|那个|这款|那款|刚才|之前|上面|前面|刚刚|上次)"
    r".*(怎么样|如何|呢|吗|什么|多少|有哪些|有没有|还有其他|还有别的)",
    r"(没懂|不明白|再说一遍|解释一下|什么意思|听不懂|能再说)",
    r"^(什么|怎么|为什么|哪个|哪款|有没有|还有).{0,10}$",
    r"^(它|这个|那个|这款|那款).{0,15}$",
]

BROAD_PATTERNS = [
    r"(全面介绍|都有哪些|怎么选|如何选择|怎样挑选"
    r"|有什么区别|有什么不同|对比|比较|推荐)",
    r"(所有|全部|各种|各类|不同.{0,5}的).*(眼镜|镜框|镜片|款式|材质|颜色)",
]

_compiled_vague = [re.compile(p) for p in VAGUE_PATTERNS]
_compiled_broad = [re.compile(p) for p in BROAD_PATTERNS]


# ═══════════════════════════════════════════════════════════════
# 主规则函数
# ═══════════════════════════════════════════════════════════════

def apply_rules(query: str) -> dict:
    """对用户查询应用轻量规则，返回第一层和第二层的初步判断。

    Args:
        query: 用户输入文本。

    Returns:
        {
            "business_intent": str | "",
            "is_specific_order": bool,
            "retrieval_strategy": str | "",
        }
    """
    query = query.strip()
    if not query:
        return {
            "business_intent": "闲聊/其他",
            "is_specific_order": False,
            "retrieval_strategy": "",
        }

    query_lower = query.lower()

    # ── 1. 检查 GENERAL（闲聊/其他） ──
    for pattern, _ in GENERAL_PATTERNS:
        if re.search(pattern, query_lower):
            return {
                "business_intent": "闲聊/其他",
                "is_specific_order": False,
                "retrieval_strategy": "",
            }
    # ── 2. 检查投诉/人工客服 ──
    for kw in COMPLAINT_KEYWORDS:
        if kw.lower() in query_lower:
            return {
                "business_intent": "投诉/人工客服",
                "is_specific_order": False,
                "retrieval_strategy": "",
            }

    # 超短文本（<= 3 字符）→ 闲聊/其他
    if len(query) <= 3:
        return {
            "business_intent": "闲聊/其他",
            "is_specific_order": False,
            "retrieval_strategy": "",
        }

    # ── 3. 检查具体订单号/运单号 ──
    is_specific_order = False
    for compiled in _compiled_specific:
        if compiled.search(query_lower):
            is_specific_order = True
            break
    # 明确的订单修改/取消语义优先于普通商品关键词
    for kw in ORDER_MODIFY_PRIORITY_KEYWORDS:
        if kw in query_lower:
            return {
                "business_intent": "订单修改/取消",
                "is_specific_order": is_specific_order,
                "retrieval_strategy": "",
            }

    # 明确的退货、退款、换货表达优先于普通商品关键词
    for kw in AFTER_SALES_PRIORITY_KEYWORDS:
        if kw in query_lower:
            return {
                "business_intent": "退换货售后",
                "is_specific_order": is_specific_order,
                "retrieval_strategy": "",
            }

    # ── 4. 检查八种业务意图关键词 ──
    intent_scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORD_MAP.items():
        score = 0

        for kw in keywords:
            # SKU 单独通过正则识别
            if kw in ("FM", "FR"):
                if re.search(
                        rf"\b{kw}\d+",
                        query,
                        re.IGNORECASE,
                ):
                    score += 1
                continue

            # 中英文关键词全部统一为小写比较
            if kw.lower() in query_lower:
                score += 1

        intent_scores[intent] = score

    if intent_scores:
        # 取得分最高的意图
        best_intent = max(intent_scores, key=lambda k: intent_scores[k])
        if intent_scores[best_intent] > 0:
            return {
                "business_intent": best_intent,
                "is_specific_order": is_specific_order,
                "retrieval_strategy": "",
            }

    # ── 5. 检查检索策略提示（VAGUE / BROAD） ──
    for compiled in _compiled_broad:
        if compiled.search(query_lower):
            return {
                "business_intent": "",
                "is_specific_order": is_specific_order,
                "retrieval_strategy": "BROAD",
            }

    for compiled in _compiled_vague:
        if compiled.search(query_lower):
            return {
                "business_intent": "",
                "is_specific_order": is_specific_order,
                "retrieval_strategy": "VAGUE",
            }

    # ── 6. 规则无法确定 ──
    return {
        "business_intent": "",
        "is_specific_order": is_specific_order,
        "retrieval_strategy": "",
    }
