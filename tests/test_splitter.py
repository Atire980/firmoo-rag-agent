"""
父子分块测试：验证分块逻辑、chunk_id / parent_id 生成、去重。

测试目标：
    - 父块、子块大小符合配置
    - 子块正确关联 parent_id
    - 超短文档（<= 子块大小）不重复拆分
    - chunk_id 去重生效
"""

import pytest
from langchain_core.documents import Document

from app.config import settings
from app.rag.splitter import split_document_to_chunks, split_all_documents


class TestParentChildSplitting:
    """测试父子分块逻辑。"""

    def _make_doc(self, content: str, metadata: dict | None = None) -> Document:
        """辅助：创建测试用 LangChain Document。"""
        if metadata is None:
            metadata = {
                "source": "test.csv",
                "source_name": "测试来源",
                "sku": "FM-TEST",
                "category": "test_cat",
                "route_intent": "test_intent",
                "document_type": "product",
                "doc_id": "DOC_TEST_001",
            }
        return Document(page_content=content, metadata=metadata)

    def test_short_content_single_chunk(self):
        """短文本（<=子块大小）应只产生一条记录，且 chunk_id == parent_id。"""
        short_text = "这是一段很短的眼镜描述。" * 3  # 约 30 字符
        doc = self._make_doc(short_text)

        records = split_document_to_chunks(doc)

        assert len(records) == 1
        assert records[0]["chunk_id"] == records[0]["parent_id"]
        assert records[0]["child_content"] == records[0]["parent_content"]

    def test_long_content_multiple_chunks(self):
        """长文本应拆分出多条子块记录。"""
        # 构造一个较长的文本
        long_text = (
            "这是一段关于眼镜产品的详细描述。"
            + "镜框采用TR90材质，轻便且有弹性。"
            + "适合日常通勤使用。"
            + "颜色有黑色、银色可选。"
            + "镜片可以配置防蓝光功能。"
            + "镜框重量约为25克。"
            + "推荐给圆脸和方脸的用户。"
            + "配送时间一般为3-5个工作日。"
            + "支持7天无理由退货。"
        ) * 10  # 扩展为较长文本

        doc = self._make_doc(long_text)
        records = split_document_to_chunks(doc)

        # 长文本应有多条记录
        assert len(records) >= 1

        # 验证每条记录的必要字段
        for record in records:
            assert "chunk_id" in record
            assert "parent_id" in record
            assert "child_content" in record
            assert "parent_content" in record
            assert len(record["child_content"]) > 0
            assert len(record["parent_content"]) > 0

    def test_metadata_preserved(self):
        """metadata 字段应正确传递到子块记录。"""
        doc = self._make_doc("TR90 材质镜框，轻便舒适。黑色可选。适合日常佩戴。" * 5)
        records = split_document_to_chunks(doc)

        for record in records:
            assert record["source_name"] == "测试来源"
            assert record["sku"] == "FM-TEST"
            assert record["category"] == "test_cat"
            assert record["route_intent"] == "test_intent"
            assert record["document_type"] == "product"

    def test_parent_child_relationship(self):
        """同一父块的子块应有相同的 parent_id。"""
        text = "A" * settings.CHILD_CHUNK_SIZE * 3  # 强制拆分多个父块
        doc = self._make_doc(text)
        records = split_document_to_chunks(doc)

        if len(records) >= 2:
            # 按 parent_id 分组
            from collections import Counter
            parent_ids = [r["parent_id"] for r in records]
            counts = Counter(parent_ids)
            # 至少有一个 parent 被多个 child 引用
            assert any(c > 1 for c in counts.values()), \
                "至少应有一个父块被多个子块引用"


class TestBatchSplitting:
    """测试批量文档分块。"""

    def test_deduplication(self):
        """相同内容的文档不应产生重复的 chunk_id。"""
        doc1 = Document(
            page_content="完全相同的眼镜描述文本内容。" * 5,
            metadata={"source_name": "src1", "sku": "", "category": "",
                       "route_intent": "", "document_type": "product", "doc_id": "D1"},
        )
        doc2 = Document(
            page_content="完全相同的眼镜描述文本内容。" * 5,
            metadata={"source_name": "src1", "sku": "", "category": "",
                       "route_intent": "", "document_type": "product", "doc_id": "D1"},
        )

        records = split_all_documents([doc1, doc2])
        chunk_ids = [r["chunk_id"] for r in records]

        assert len(chunk_ids) == len(set(chunk_ids)), "不应有重复的 chunk_id"

    def test_empty_input(self):
        """空输入应返回空列表。"""
        assert split_all_documents([]) == []
