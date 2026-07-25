"""
建库流程：读取原始数据 → 清洗 → 父子分块 → BGE-M3 批量编码 → 创建 Milvus Collection → 批量写入。

使用方式：
    python -m scripts.build_index

或在代码中调用：
    from app.rag.ingest import run_ingest
    run_ingest()
"""

from app.config import settings
from app.rag.document_loader import load_documents
from app.rag.splitter import split_all_documents
from app.rag.embeddings import encode_documents
from app.rag.milvus_store import create_collection, create_indexes, insert_records
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_ingest(data_dir: str = "data/raw", drop_existing: bool = False) -> None:
    """执行完整建库流程。

    Args:
        data_dir:      原始数据目录。
        drop_existing: 是否删除已有 Collection 后重建。
    """
    logger.info("=" * 50)
    logger.info("开始建库流程")
    logger.info("=" * 50)

    # 1. 加载文档
    logger.info("[1/5] 加载原始文档...")
    documents = load_documents(data_dir)
    if not documents:
        logger.error("未加载到任何文档，建库终止")
        return
    logger.info("共加载 %d 条文档", len(documents))

    # 2. 父子分块
    logger.info("[2/5] 父子分块...")
    records = split_all_documents(documents)
    if not records:
        logger.error("分块后无记录，建库终止")
        return
    logger.info("分块完成，共 %d 条记录", len(records))

    # 3. BGE-M3 批量编码（使用 child_content）
    logger.info("[3/5] BGE-M3 批量编码...")
    child_contents = [r["child_content"] for r in records]
    dense_vectors, sparse_vectors = encode_documents(child_contents)
    logger.info("编码完成，dense=%d, sparse=%d", len(dense_vectors), len(sparse_vectors))

    # 4. 创建 Milvus Collection
    logger.info("[4/5] 创建 Milvus Collection...")
    create_collection(drop_existing=drop_existing)
    create_indexes()

    # 5. 批量写入
    logger.info("[5/5] 写入 Milvus...")
    insert_records(records, dense_vectors, sparse_vectors)

    logger.info("=" * 50)
    logger.info("建库完成！共写入 %d 条向量记录到 %s", len(records), settings.MILVUS_COLLECTION)
    logger.info("=" * 50)
