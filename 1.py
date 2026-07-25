import asyncio

from langchain_core.messages import HumanMessage

from app.agent.graph import get_ecommerce_graph


async def main() -> None:
    graph = get_ecommerce_graph()

    # 两轮必须使用完全相同的 thread_id
    config = {
        "configurable": {
            "thread_id": "memory-test-001"
        }
    }

    first_query = "我正在看 TR90 镜框，它有什么优点？"
    second_query = "那它适合长时间佩戴吗？"

    print("=" * 70)
    print("第一轮：", first_query)

    first_result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=first_query)
            ]
        },
        config=config,
    )

    print("第一轮回答：")
    print(first_result.get("answer"))

    print("\n" + "=" * 70)
    print("第二轮：", second_query)

    # 这里仍然使用同一个 config
    second_result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=second_query)
            ]
        },
        config=config,
    )

    print("第二轮回答：")
    print(second_result.get("answer"))

    # 读取当前 thread_id 保存的完整 State
    snapshot = await graph.aget_state(config)

    messages = (
        snapshot.values.get("messages", [])
        if snapshot and snapshot.values
        else []
    )

    print("\n" + "=" * 70)
    print("MemorySaver 中的消息数量：", len(messages))
    print("保存的对话历史：")

    contents: list[str] = []

    for index, message in enumerate(messages, start=1):
        content = str(getattr(message, "content", ""))
        contents.append(content)

        print(
            f"{index}. "
            f"{type(message).__name__}: "
            f"{content[:200]}"
        )

    # 至少确认两轮用户问题都进入了同一个会话历史
    assert any(
        first_query in content
        for content in contents
    ), "第一轮问题没有写入记忆"

    assert any(
        second_query in content
        for content in contents
    ), "第二轮问题没有写入记忆"

    print("\n多轮记忆测试通过！")


if __name__ == "__main__":
    asyncio.run(main())