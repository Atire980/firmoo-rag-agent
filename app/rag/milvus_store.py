"""
Milvus 向量存储：管理 Collection 的创建、索引、写入与混合检索。

Collection Schema 字段：
    chunk_id        VARCHAR (PK)    子块唯一 ID
    parent_id       VARCHAR         父块 ID
    child_content   VARCHAR         子块文本（检索用）
    parent_content  VARCHAR         父块文本（生成用）
    dense_vector    FLOAT_VECTOR    BGE-M3 dense embedding
    sparse_vector   SPARSE_FLOAT_VECTOR  BGE-M3 sparse embedding
    source_name     VARCHAR         来源名称
    sku             VARCHAR         SKU
    category        VARCHAR         分类
    route_intent    VARCHAR         路由意图
    document_type   VARCHAR         文档类型
"""

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    WeightedRanker,
    AnnSearchRequest,
    connections,
    utility,
)

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# BGE-M3 dense 向量维度
BGE_M3_DENSE_DIM = 1024

# 单例 MilvusClient（用于建库和写入）
_client: MilvusClient | None = None


def _get_client() -> MilvusClient:
    """获取 MilvusClient 单例。"""
    global _client
    if _client is None:
        _client = MilvusClient(
            uri=settings.MILVUS_URI,
            token=settings.MILVUS_TOKEN or None,
        )
    return _client


def _collection_name() -> str:
    return settings.MILVUS_COLLECTION

def connect_milvus() -> None:
    """建立 pymilvus ORM 连接。

    Collection、utility 和 hybrid_search 使用 ORM 连接；
    每个新 Python 进程都需要重新连接一次。
    """
    connections.connect(
        alias="default",
        uri=settings.MILVUS_URI,
        token=settings.MILVUS_TOKEN or None,
    )


# ──────────────────────── Schema 定义 ────────────────────────────

def build_schema() -> CollectionSchema:
    """构建 Milvus Collection Schema。

    Returns:
        CollectionSchema 实例。
    """
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="child_content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="parent_content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=BGE_M3_DENSE_DIM),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="sku", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="route_intent", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="document_type", dtype=DataType.VARCHAR, max_length=32),
    ]
    return CollectionSchema(fields, description="眼镜电商 RAG 知识库")


def create_collection(drop_existing: bool = False) -> Collection:
    """创建 Milvus Collection。

    Args:
        drop_existing: 是否先删除同名 Collection。

    Returns:
        创建好的 Collection 对象。
    """
    collection_name = _collection_name()

    # 连接（pymilvus ORM 方式）
    connections.connect(
        alias="default",
        uri=settings.MILVUS_URI,
        token=settings.MILVUS_TOKEN or None,
    )

    if drop_existing and utility.has_collection(collection_name):
        logger.info("删除已存在的 Collection: %s", collection_name)
        utility.drop_collection(collection_name)

    if utility.has_collection(collection_name):
        logger.info("Collection %s 已存在，跳过创建", collection_name)
        return Collection(collection_name)

    schema = build_schema()
    collection = Collection(collection_name, schema)
    logger.info("Collection %s 创建完成", collection_name)
    return collection


def create_indexes() -> None:
    """为 dense_vector 和 sparse_vector 创建索引。"""
    collection_name = _collection_name()
    collection = Collection(collection_name)

    # dense 索引 (IVF_FLAT)
    dense_index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",  # Inner Product（BGE-M3 已归一化）
        "params": {"nlist": 128},
    }
    try:
        collection.create_index("dense_vector", dense_index_params)
        logger.info("dense_vector 索引创建完成")
    except Exception as e:
        logger.warning("dense_vector 索引创建: %s", e)

    # sparse 索引 (SPARSE_INVERTED_INDEX)
    sparse_index_params = {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
    }
    try:
        collection.create_index("sparse_vector", sparse_index_params)
        logger.info("sparse_vector 索引创建完成")
    except Exception as e:
        logger.warning("sparse_vector 索引创建: %s", e)

    collection.load()
    logger.info("Collection %s 加载完成", collection_name)


# ──────────────────────── 数据写入 ──────────────────────────────

def insert_records(records: list[dict], dense_vectors: list[list[float]], sparse_vectors: list[dict]) -> None:
    """将分块记录批量写入 Milvus。

    Args:
        records:   分块记录列表（来自 splitter）。
        dense_vectors:  对应的 dense 向量。
        sparse_vectors: 对应的 sparse 向量。
    """
    if not records:
        logger.warning("没有记录需要写入")
        return

    client = _get_client()
    collection_name = _collection_name()

    data = []
    for i, record in enumerate(records):
        data.append({
            "chunk_id": record["chunk_id"],
            "parent_id": record["parent_id"],
            "child_content": record["child_content"],
            "parent_content": record["parent_content"],
            "dense_vector": dense_vectors[i],
            "sparse_vector": sparse_vectors[i],
            "source_name": record.get("source_name", ""),
            "sku": record.get("sku", ""),
            "category": record.get("category", ""),
            "route_intent": record.get("route_intent", ""),
            "document_type": record.get("document_type", ""),
        })

    # 分批写入
    batch_size = 200
    total = len(data)
    for start in range(0, total, batch_size):
        batch = data[start:start + batch_size]
        client.insert(collection_name=collection_name, data=batch)
        logger.debug("写入 %d/%d", min(start + batch_size, total), total)

    logger.info("共写入 %d 条记录到 %s", total, collection_name)


# ──────────────────────── 混合检索 ──────────────────────────────

def hybrid_search(
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    top_k: int | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    sku_filter: str | None = None,
    category_filter: str | None = None,
    route_intent_filter: str | None = None,
    document_type_filter: str | None = None,
) -> list[dict]:
    """执行 BGE-M3 dense + sparse 混合检索。

    Args:
        dense_vector:  Query 的 dense embedding。
        sparse_vector: Query 的 sparse embedding。
        top_k:         返回数量，默认使用配置中的 RETRIEVAL_TOP_K。
        dense_weight:  dense 路径权重，默认使用配置。
        sparse_weight: sparse 路径权重，默认使用配置。
        sku_filter:    可选的 SKU 过滤（精确匹配）。
        category_filter: 可选分类过滤（精确匹配）。
        route_intent_filter: 可选路由意图过滤（精确匹配）。
        document_type_filter: 可选文档类型过滤，支持 "in" 语法，
            例如 'document_type in ["product","faq"]'。

    Returns:
        检索结果列表，每条结果包含所有 Milvus 字段。
    """
    if top_k is None:
        top_k = settings.RETRIEVAL_TOP_K
    if dense_weight is None:
        dense_weight = settings.DENSE_WEIGHT
    if sparse_weight is None:
        sparse_weight = settings.SPARSE_WEIGHT

    collection_name = _collection_name()
    # 每个新 Python 进程在执行 ORM 检索前都要建立连接
    connect_milvus()

    # 构造过滤表达式
    filter_parts = []

    # 构造过滤表达式
    filter_parts = []
    if sku_filter:
        filter_parts.append(f'sku == "{sku_filter}"')
    if category_filter:
        filter_parts.append(f'category == "{category_filter}"')
    if route_intent_filter:
        filter_parts.append(f'route_intent == "{route_intent_filter}"')
    if document_type_filter:
        filter_parts.append(document_type_filter)
    filter_expr = " && ".join(filter_parts) if filter_parts else None

    # dense 检索请求
    dense_search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_search_params,
        limit=top_k * 2,  # 多取一些供加权合并
        expr=filter_expr,
    )

    # sparse 检索请求
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_search_params,
        limit=top_k * 2,
        expr=filter_expr,
    )

    # 加权混合
    ranker = WeightedRanker(dense_weight, sparse_weight)

    # 使用 pymilvus ORM 的 hybrid_search
    collection = Collection(collection_name)
    try:
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=ranker,
            limit=top_k,
            output_fields=[
                "chunk_id", "parent_id", "child_content", "parent_content",
                "source_name", "sku", "category", "route_intent", "document_type",
            ],
        )
    except Exception as e:
        logger.error("混合检索失败: %s", e)
        return []

    # 转换输出格式
    # 转换输出格式
    output: list[dict] = []

    for hits in results:
        for hit in hits:
            output.append({
                "chunk_id": (
                        hit.entity.get("chunk_id") or ""
                ),
                "parent_id": (
                        hit.entity.get("parent_id") or ""
                ),
                "child_content": (
                        hit.entity.get("child_content") or ""
                ),
                "parent_content": (
                        hit.entity.get("parent_content") or ""
                ),
                "source_name": (
                        hit.entity.get("source_name") or ""
                ),
                "sku": (
                        hit.entity.get("sku") or ""
                ),
                "category": (
                        hit.entity.get("category") or ""
                ),
                "route_intent": (
                        hit.entity.get("route_intent") or ""
                ),
                "document_type": (
                        hit.entity.get("document_type") or ""
                ),
                "score": float(hit.score),
            })

    return output
