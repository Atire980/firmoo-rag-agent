"""
集中管理所有配置项，使用 pydantic-settings 从 .env 文件加载。

所有阈值、路径、连接信息统一从这里获取，禁止在其他文件中硬编码。
"""

from pydantic_settings import BaseSettings


# ═══════════════════════════════════════════════════════════════
# 八种业务意图标签（与 intent_train.csv 的 label 列一致）
# ═══════════════════════════════════════════════════════════════

BUSINESS_INTENT_LABELS: list[str] = [
    "商品咨询",
    "镜片/度数咨询",
    "物流查询",
    "退换货售后",
    "订单修改/取消",
    "投诉/人工客服",
    "闲聊/其他",
    "优惠/支付",
]

# label → index
BUSINESS_INTENT_TO_ID: dict[str, int] = {
    label: idx for idx, label in enumerate(BUSINESS_INTENT_LABELS)
}

# ═══════════════════════════════════════════════════════════════
# 业务意图 → 路由动作 映射
# ═══════════════════════════════════════════════════════════════

# 固定映射：这些意图一定走对应路由
INTENT_TO_ROUTE: dict[str, str] = {
    "商品咨询":       "RAG",
    "镜片/度数咨询":   "RAG",
    "退换货售后":     "RAG",
    "优惠/支付":      "RAG",
    "投诉/人工客服":   "HANDOFF",
    "闲聊/其他":      "GENERAL",
    # 物流查询 → 需要规则再判断（通用政策 RAG，具体单号 UNSUPPORTED）
    # 订单修改/取消 → 需要规则再判断（通用规则 RAG，具体操作 UNSUPPORTED）
}

# 业务意图 → 检索时优先过滤的 source_type
INTENT_TO_SOURCE_TYPE: dict[str, list[str]] = {
    "商品咨询":       ["product", "faq", "review"],
    "镜片/度数咨询":   ["product", "faq"],
    "物流查询":       ["policy", "faq"],
    "退换货售后":     ["policy", "faq"],
    "订单修改/取消":   ["policy", "faq"],
    "优惠/支付":      ["policy", "faq"],
    "投诉/人工客服":   [],
    "闲聊/其他":      [],
}


class Settings(BaseSettings):
    """应用全局配置，所有值均可通过环境变量或 .env 文件覆盖。"""

    # --- 应用基础 ---
    APP_ENV: str = "development"
    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = 8000

    # --- LLM ---
    # TODO: 请填入你的 LLM API 信息
    LLM_API_KEY: str =""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = ""

    # --- Milvus ---
    # TODO: 请填入你的 Milvus 连接信息
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_TOKEN: str = ""
    MILVUS_COLLECTION: str = "ecommerce_knowledge"

    # --- 模型路径 ---
    # TODO: 请填入你电脑上的模型路径，以下为占位符
    BGE_M3_MODEL_PATH: str = ""
    RERANKER_MODEL_PATH: str = ""
    # 八分类模型路径（由 scripts/train_classifier.py 训练产出）
    CLASSIFIER_MODEL_PATH: str = ""

    # --- 混合检索权重 ---
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4
    RETRIEVAL_TOP_K: int = 12
    RERANK_TOP_K: int = 4
    RAG_CONFIDENCE_THRESHOLD: float = 0.5

    # --- 分块参数 ---
    PARENT_CHUNK_SIZE: int = 1200
    CHILD_CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 60

    # --- 记忆参数 ---
    MEMORY_WINDOW_SIZE: int = 10
    SUMMARY_TRIGGER_ROUNDS: int = 10

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


# 全局单例，禁止在其他文件中重复实例化 Settings()
settings = Settings()
