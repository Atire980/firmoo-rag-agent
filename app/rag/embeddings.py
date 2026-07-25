"""
BGE-M3 Embedding 单例：提供文档批量编码和查询编码，同时输出 dense + sparse 向量。

使用 FlagEmbedding 库的 BGEM3FlagModel，不在其他文件中重复加载模型。
"""

import asyncio
from typing import Any

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 模型单例，惰性加载
_bge_model: Any = None


def _load_model():
    """惰性加载 BGE-M3 模型（仅在首次调用时加载）。"""
    global _bge_model
    if _bge_model is not None:
        return _bge_model

    model_path = settings.BGE_M3_MODEL_PATH
    if not model_path:
        raise RuntimeError(
            "BGE_M3_MODEL_PATH 未配置，请在 .env 中设置本地模型路径"
        )

    logger.info("正在加载 BGE-M3 模型: %s", model_path)
    try:
        from FlagEmbedding import BGEM3FlagModel
        _bge_model = BGEM3FlagModel(
            model_path,
            use_fp16=True,
        )
        logger.info("BGE-M3 模型加载完成 (dense dim=%d)", _bge_model.model.config.hidden_size)
    except Exception as e:
        logger.error("BGE-M3 模型加载失败: %s", e)
        raise

    return _bge_model


def get_model():
    """获取 BGE-M3 单例模型。"""
    return _load_model()


def encode_documents(texts: list[str]) -> tuple[list[list[float]], list[dict]]:
    """批量编码文档文本，返回 (dense_vectors, sparse_vectors)。

    Args:
        texts: 待编码的文本列表。

    Returns:
        dense_vectors: 每个文本的 dense 向量 (list[float])。
        sparse_vectors: 每个文本的 sparse 向量 (dict[int, float])。
    """
    model = get_model()
    # BGE-M3 的 encode 方法同时返回 dense 和 sparse（lexical weights）
    output = model.encode(
        texts,
        batch_size=32,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_list: list[list[float]] = []
    sparse_list: list[dict] = []

    for i in range(len(texts)):
        dense_vec = output["dense_vecs"][i].tolist()
        sparse_dict = output["lexical_weights"][i]
        # 确保 sparse_dict 的 key 是 int 类型
        sparse_dict_int = {int(k): float(v) for k, v in sparse_dict.items()}
        dense_list.append(dense_vec)
        sparse_list.append(sparse_dict_int)

    return dense_list, sparse_list


def encode_query(query: str) -> tuple[list[float], dict[int, float]]:
    """编码单条查询文本，返回 (dense_vector, sparse_vector)。

    Args:
        query: 查询文本。

    Returns:
        dense_vector: dense 向量 (list[float])。
        sparse_vector: sparse 向量 (dict[int, float])。
    """
    dense_list, sparse_list = encode_documents([query])
    return dense_list[0], sparse_list[0]


async def async_encode_query(query: str) -> tuple[list[float], dict[int, float]]:
    """异步包装，在 executor 中执行同步编码。

    用于 LangGraph 异步节点中调用，避免阻塞事件循环。

    Args:
        query: 查询文本。

    Returns:
        (dense_vector, sparse_vector)
    """
    return await asyncio.to_thread(encode_query, query)


async def async_encode_documents(
    texts: list[str],
) -> tuple[list[list[float]], list[dict]]:
    """异步批量编码文档。"""
    return await asyncio.to_thread(encode_documents, texts)
