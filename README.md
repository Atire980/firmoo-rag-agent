# 眼镜跨境电商 LangGraph RAG 客服 Agent

基于 LangGraph 的智能客服工作流系统，面向跨境眼镜电商场景，提供商品咨询、镜片知识、尺码建议、物流售后等多轮对话服务。

## 核心能力

- **意图分类**：自动识别 GENERAL / PRECISE / VAGUE / BROAD 四种查询类型
- **混合检索**：BGE-M3 dense + sparse 双路向量检索
- **精排重排**：BGE-Reranker CrossEncoder 对候选结果重新打分
- **多轮记忆**：MemorySaver + 滑动窗口 + 对话摘要
- **父子块**：子块负责精确检索，父块提供完整上下文
- **保守兜底**：知识库不足时明确告知，不编造商品/价格/库存/物流信息

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.11 |
| API | FastAPI + Uvicorn |
| Agent | LangGraph |
| LLM | OpenAI 兼容接口（LangChain） |
| Embedding | BGE-M3 (dense + sparse) |
| 向量库 | Milvus |
| 重排 | BGE-Reranker / CrossEncoder |
| 分类 | sentence-transformers/all-MiniLM-L6-v2 |
| 配置 | Pydantic Settings + .env |

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# macOS / Linux 激活
source .venv/bin/activate

# 安装依赖
python -m pip install -r requirements.txt
```

### 2. 配置文件

```bash
# 复制配置模板
# .env_ 是模板文件，包含所有待填写的配置项
# 填写完成后重命名为 .env
```

编辑 `.env`，填入：
- LLM API Key、Base URL、模型名
- Milvus 连接地址和 Token
- BGE-M3、Reranker、分类模型的本地路径

### 3. 构建知识库

```bash
python -m scripts.build_index
```

### 4. 验证检索

```bash
python -m scripts.test_retrieval
```

### 5. 启动 API

```bash
python -m uvicorn app.main:app --reload
```

### 6. 启动前端（Streamlit）

```bash
# 新开一个终端
python -m streamlit run frontend/streamlit_app.py
```

前端默认地址：http://localhost:8501

前端配置项（已在 .env 中）：
- `FRONTEND_API_BASE_URL` — 后端地址，默认 `http://127.0.0.1:8000`
- `FRONTEND_REQUEST_TIMEOUT` — 请求超时秒数，默认 `120`

### 7. 接口示例

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 发送聊天
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "session_id": "session_001",
    "message": "TR90 镜框有什么优点？"
  }'
```

## 运行测试

```bash
# 全部测试
python -m pytest -q

# 单个测试文件
python -m pytest tests/test_nodes.py -q
```

## 项目结构

```text
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # Pydantic Settings 全局配置
│   ├── api/chat.py          # /api/chat 路由
│   ├── agent/               # LangGraph Agent
│   │   ├── state.py         # EcommerceState 定义
│   │   ├── prompts.py       # 所有提示词
│   │   ├── nodes.py         # 各节点实现
│   │   ├── graph.py         # StateGraph 组装
│   │   └── memory.py        # MemorySaver 管理
│   ├── classifier/          # 意图分类
│   │   ├── rules.py         # 轻量关键词规则
│   │   └── intent_classifier.py  # BERT 二分类模型
│   ├── rag/                 # RAG 核心
│   │   ├── document_loader.py
│   │   ├── splitter.py      # 父子分块
│   │   ├── embeddings.py    # BGE-M3
│   │   ├── milvus_store.py  # Milvus 操作
│   │   ├── retriever.py     # 混合检索
│   │   ├── reranker.py      # 精排
│   │   └── ingest.py        # 建库流程
│   ├── llm/factory.py       # LLM 工厂
│   ├── schemas/chat.py      # 请求/响应模型
│   └── utils/               # 日志与文本工具
├── data/                    # 原始数据
├── models/                  # 本地模型文件
├── scripts/                 # 建库、检索测试脚本
└── tests/                   # 单元与端到端测试
```

## 数据来源

本项目使用模拟的跨境眼镜电商业务数据，包括：
- 商品知识（product）
- 常见问题（faq）
- 售后政策（policy）
- 用户评论（review）

**重要声明**：数据集为模拟数据，不是真实网站爬取数据。价格、实时库存、真实订单状态等确定性信息不得由 RAG 或大模型推测。

## 功能边界

本版本**不实现**：
- MCP Server / Client
- 联网搜索
- PostgreSQL 长期记忆
- 前端页面
- 用户登录与权限
- 真实订单/支付/库存写入
- 多租户隔离
