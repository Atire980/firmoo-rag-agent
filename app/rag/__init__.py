from collections import Counter

from app.rag.document_loader import load_documents


def main() -> None:
    # 调用 document_loader.py 中的总加载函数
    documents = load_documents("data/raw")

    print("=" * 60)
    print("文档总数：", len(documents))

    # 防止 documents 为空时访问 documents[0] 报错
    if not documents:
        print("没有成功加载任何文档。")
        return

    first_document = documents[0]

    print("第一条数据类型：", type(first_document))
    print("第一条正文：")
    print(first_document.page_content[:300])

    print("\n第一条 metadata：")
    print(first_document.metadata)

    # 统计不同知识类型的数量
    document_type_counts = Counter(
        document.metadata.get("document_type", "")
        for document in documents
    )

    print("\n各类知识数量：")
    print(document_type_counts)

    # 基础断言
    assert len(documents) > 0, "文档列表不能为空"
    assert first_document.page_content, "正文不能为空"
    assert "doc_id" in first_document.metadata, "metadata 缺少 doc_id"
    assert "document_type" in first_document.metadata, (
        "metadata 缺少 document_type"
    )

    print("\n文档加载测试通过！")


if __name__ == "__main__":
    main()