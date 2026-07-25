"""
意图分类测试：覆盖八分类规则 + 订单号识别 + 检索策略提示。

测试目标：
    你好                         → 闲聊/其他
    TR90 镜框有什么优点？        → 商品咨询
    我有600度近视可以配吗？      → 镜片/度数咨询
    物流到哪里了？               → 物流查询
    戴了两天不舒服可以退吗？     → 退换货售后
    颜色选错了能换吗？           → 订单修改/取消
    客服一直不回复？             → 投诉/人工客服
    优惠券怎么用？               → 优惠/支付
    它还有其他颜色吗？           → VAGUE 策略提示
    怎么选择适合自己的眼镜？      → BROAD 策略提示
    我的订单OD001864多久能到？    → 订单修改/取消(原为物流查询)+ is_specific_order
    帮我取消订单OD001234         → is_specific_order
"""

import pytest

from app.classifier.rules import apply_rules


class TestBusinessIntentRules:
    """测试八种业务意图的规则识别。"""

    @pytest.mark.parametrize("query, expected_intent", [
        # 闲聊/其他
        ("你好", "闲聊/其他"),
        ("您好", "闲聊/其他"),
        ("hi", "闲聊/其他"),
        ("谢谢", "闲聊/其他"),
        ("再见", "闲聊/其他"),
        ("你是谁", "闲聊/其他"),
        ("早上好", "闲聊/其他"),
        # 商品咨询
        ("TR90 镜框有什么优点？", "商品咨询"),
        ("这个镜框是什么材质？", "商品咨询"),
        ("圆脸适合什么镜框？", "商品咨询"),
        ("有没有Silver色的？", "商品咨询"),
        ("钛材质的镜框重不重？", "商品咨询"),
        # 镜片/度数咨询
        ("我有600度近视可以配吗？", "镜片/度数咨询"),
        ("渐进镜片怎么选择？", "镜片/度数咨询"),
        ("PD 瞳距不知道怎么办？", "镜片/度数咨询"),
        ("这款可以配蓝光镜片吗？", "镜片/度数咨询"),
        ("折射率选多少合适？", "镜片/度数咨询"),
        # 物流查询
        ("物流到哪里了？", "物流查询"),
        ("可以加急配送吗？", "物流查询"),
        ("物流单号在哪里看？", "物流查询"),
        # 退换货售后
        ("退货政策是怎样的？", "退换货售后"),
        ("戴了两天不舒服可以退吗？", "退换货售后"),
        ("镜片有划痕可以换吗？", "退换货售后"),
        ("超过七天还能退吗？", "退换货售后"),
        # 订单修改/取消
        ("颜色选错了能换吗？", "订单修改/取消"),
        ("订单进入加工后还能取消吗？", "订单修改/取消"),
        ("地址填错了可以改吗？", "订单修改/取消"),
        # 投诉/人工客服
        ("客服一直不回复？", "投诉/人工客服"),
        ("请帮我升级处理", "投诉/人工客服"),
        ("我要投诉", "投诉/人工客服"),
        ("转人工", "投诉/人工客服"),
        # 优惠/支付
        ("优惠券怎么用？", "优惠/支付"),
        ("为什么支付失败？", "优惠/支付"),
        ("可以用 PayPal 吗？", "优惠/支付"),
    ])
    def test_rule_intent_matches(self, query, expected_intent):
        """规则应正确识别八种业务意图。"""
        result = apply_rules(query)
        assert result["business_intent"] == expected_intent, \
            f"查询 '{query}' 期望意图 {expected_intent}，实际 {result['business_intent']}"

    def test_empty_query_defaults_general(self):
        """空查询默认为 闲聊/其他。"""
        result = apply_rules("")
        assert result["business_intent"] == "闲聊/其他"

    def test_short_query_defaults_general(self):
        """超短文本（<=3 字符）归为 闲聊/其他。"""
        result = apply_rules("嗯")
        assert result["business_intent"] == "闲聊/其他"


class TestSpecificOrderDetection:
    """测试具体订单号/运单号识别（用于区分 UNSUPPORTED）。"""

    @pytest.mark.parametrize("query, expected_specific", [
        # 具体订单号
        ("订单OD001864多久能到？", True),
        ("帮我取消订单OD001234", True),
        ("OD999999 物流状态", True),
        # 具体操作请求
        ("帮我取消订单", True),
        ("我要退货退款", True),
        ("修改收货地址", True),
        ("我要改配送地址", True),
        ("查我的物流", True),
        ("查快递到哪里了", True),
        # 非具体操作（通用政策）
        ("退换货政策是什么？", False),
        ("怎么取消订单？", False),
        ("物流一般几天到？", False),
        ("可以加急配送吗？", False),
        ("订单修改的流程是什么？", False),
    ])
    def test_specific_order_detection(self, query, expected_specific):
        """应正确区分具体操作请求和通用政策查询。"""
        result = apply_rules(query)
        assert result["is_specific_order"] == expected_specific, \
            f"查询 '{query}' 期望 is_specific_order={expected_specific}，" \
            f"实际 {result['is_specific_order']}"


class TestRetrievalStrategyHints:
    """测试 VAGUE / BROAD 检索策略提示。"""

    @pytest.mark.parametrize("query", [
        "它还有其他颜色吗？",
        "这个会不会显脸大？",
        "刚才那款有其他尺寸吗？",
        "没懂，能再说一遍吗？",
    ])
    def test_vague_strategy_hint(self, query):
        """含指代词的查询应提示 VAGUE 策略。"""
        result = apply_rules(query)
        assert result["retrieval_strategy"] in ("VAGUE", ""), \
            f"查询 '{query}' 期望 VAGUE 或空策略提示"

    @pytest.mark.parametrize("query", [
        "怎么选择适合自己的眼镜？",
        "都有哪些镜框材质？",
        "不同材质的镜框有什么区别？",
    ])
    def test_broad_strategy_hint(self, query):
        """范围过大的问题应提示 BROAD 策略。"""
        result = apply_rules(query)
        assert result["retrieval_strategy"] in ("BROAD", ""), \
            f"查询 '{query}' 期望 BROAD 或空策略提示"


class TestConfigLabels:
    """测试配置中的八分类标签定义。"""

    def test_eight_labels_defined(self):
        """BUSINESS_INTENT_LABELS 应包含恰好八种标签。"""
        from app.config import BUSINESS_INTENT_LABELS
        assert len(BUSINESS_INTENT_LABELS) == 8
        expected = [
            "商品咨询", "镜片/度数咨询", "物流查询", "退换货售后",
            "订单修改/取消", "投诉/人工客服", "闲聊/其他", "优惠/支付",
        ]
        assert BUSINESS_INTENT_LABELS == expected

    def test_intent_to_route_coverage(self):
        """INTENT_TO_ROUTE 应覆盖所有标签。"""
        from app.config import INTENT_TO_ROUTE, BUSINESS_INTENT_LABELS
        for label in BUSINESS_INTENT_LABELS:
            if label in ("物流查询", "订单修改/取消"):
                # 这两个需要规则二次判断，不在固定映射中
                assert label not in INTENT_TO_ROUTE
            else:
                assert label in INTENT_TO_ROUTE, \
                    f"INTENT_TO_ROUTE 缺少 {label}"

    def test_intent_to_source_type_coverage(self):
        """INTENT_TO_SOURCE_TYPE 应覆盖所有八种标签。"""
        from app.config import INTENT_TO_SOURCE_TYPE, BUSINESS_INTENT_LABELS
        for label in BUSINESS_INTENT_LABELS:
            assert label in INTENT_TO_SOURCE_TYPE, \
                f"INTENT_TO_SOURCE_TYPE 缺少 {label}"
