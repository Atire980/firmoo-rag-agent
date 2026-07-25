# CLAUDE.md

This file provides guidance to Claude Code when building and modifying this repository.

## 1. 项目概述

本项目是一个面向眼镜跨境电商场景的 **LangGraph RAG 客服 Agent**。

系统主要回答以下问题：

- 商品与镜框咨询
- 镜片、度数与材质知识
- 尺码、脸型与佩戴建议
- 物流、退换货与售后政策
- 基于多轮上下文的追问
- 普通闲聊与通用问题

系统采用“意图分类 + 多策略检索 + 混合召回 + 重排 + 多轮记忆”的工作流型 Agent。第一版不使用 MCP，不使用联网搜索，不实现下单、支付、库存修改或真实订单查询。

核心目标：

1. 将旧版单体 `RAGSystem.generate_answer()` 改造成 LangGraph 节点化工作流。
2. 保留旧项目的电商知识库、父子块、BGE-M3、Milvus 混合检索与 Reranker。
3. 引入 GENERAL / PRECISE / VAGUE / BROAD 路由。
4. 使用 MemorySaver 支持多轮会话。
5. 提供简洁、可运行、可测试的 FastAPI 接口。

---

## 2. 第一版功能边界

### 必须实现

- LangGraph `StateGraph`
- `messages` 使用 `add_messages` 累加
- 通用问题与电商专业问题分类
- PRECISE：原问题直接检索
- VAGUE：结合历史生成 HyDE 文档后检索
- BROAD：Multi-Query 拆分后并行检索
- BGE-M3 dense + sparse 向量
- Milvus 混合检索
- BGE-Reranker / CrossEncoder 精排
- 父子块：子块负责检索，父块负责生成上下文
- 高置信度 RAG 回答
- 低置信度保守兜底，不编造商品、价格、库存、物流与售后信息
- MemorySaver 多轮记忆
- 最近 10 轮滑动窗口
- 对话摘要节点，摘要先保存在 LangGraph State 中
- FastAPI `/api/chat` 与 `/health`
- Pytest 单元测试和基础端到端测试

### 暂不实现

- MCP Server / MCP Client
- Tavily、DuckDuckGo 或任何联网搜索
- PostgreSQL 长期记忆
- 前端页面
- 用户登录与权限系统
- 真实订单、支付、库存写入
- 多租户与多商家隔离
- Docker Compose 自动部署

不得为“看起来更完整”而擅自加入以上功能。

---

## 3. 技术栈

- **语言**：Python 3.11
- **API**：FastAPI + Uvicorn
- **Agent 编排**：LangGraph
- **LLM 调用**：LangChain，使用 OpenAI-compatible 接口
- **意图分类**：使用 `sentence-transformers/all-MiniLM-L6-v2` 
- **Embedding**：BGE-M3，生成 dense + sparse
- **向量数据库**：Milvus
- **重排模型**：BGE-Reranker 或兼容的 CrossEncoder
- **配置**：Pydantic Settings + `.env`
- **测试**：Pytest + HTTPX
- **日志**：Python logging 或 structlog，统一封装

---

## 4. Agent 总体流程

```text
START
  ↓
prepare_query
  ↓
classify_query
  ├── GENERAL ─────────────────────→ generate_general
  ├── PRECISE ─────────────────────→ retrieve
  ├── VAGUE → hyde_generate ───────→ retrieve
  └── BROAD → multi_query_rewrite ─→ retrieve

retrieve
  ↓
route_by_confidence
  ├── HIGH → generate_rag
  └── LOW  → generate_fallback

generate_general ─┐
generate_rag ──────┼→ save_memory → END
generate_fallback ─┘
```

### 分支定义

- `GENERAL`：闲聊、问候、与电商知识库无关的通用问题。
- `PRECISE`：对象和目标明确，可直接使用原问题检索。
- `VAGUE`：含“它、这个、刚才那款、没懂”等指代，必须结合最近历史生成 HyDE 文档。
- `BROAD`：主题明确但范围过大，需要拆成 3～5 个具体子问题。

---

## 5. 推荐目录结构

```text
ecommerce-rag-agent/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── chat.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── prompts.py
│   │   ├── nodes.py
│   │   ├── graph.py
│   │   └── memory.py
│   │
│   ├── classifier/
│   │   ├── __init__.py
│   │   ├── rules.py
│   │   └── intent_classifier.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   ├── milvus_store.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── ingest.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── factory.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── chat.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── text.py
│
├── data/
│   ├── raw/
│   │   ├── rag_knowledge_base.csv
│   │   ├── products.csv
│   │   └── policies/
│   └── processed/
│
├── models/
│   └── .gitkeep
│
├── scripts/
│   ├── build_index.py
│   ├── inspect_collection.py
│   └── test_retrieval.py
│
├── tests/
│   ├── test_classifier.py
│   ├── test_splitter.py
│   ├── test_retriever.py
│   ├── test_nodes.py
│   └── test_api.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pytest.ini
├── README.md
└── CLAUDE.md
```

---

## 6. 每个文件的职责

### 根目录

#### `requirements.txt`

只放第一版真实使用的依赖。不要加入 MCP、前端、PostgreSQL 或无关包。

#### `.env.example`

列出所有配置名，不允许填写真实密钥。

#### `README.md`

说明项目目标、安装方法、建库步骤、启动命令、接口示例和测试命令。

---

### `app/main.py`

- 创建 FastAPI 应用。
- 注册 `/api/chat` 路由。
- 提供 `/health`。
- 不挂载 MCP。
- 不在该文件中编写 RAG 或 Agent 业务逻辑。

### `app/config.py`

使用 `pydantic-settings` 管理配置：

- LLM API Key、Base URL、模型名
- Milvus URI、Token、Collection 名
- BGE-M3 模型路径
- Reranker 模型路径
- BERT 分类模型路径
- 检索数量与置信度阈值
- 父块、子块大小与 overlap
- 历史窗口大小

所有阈值必须可配置，禁止散落在业务文件中硬编码。

---

### `app/api/chat.py`

- 定义聊天接口。
- 把请求转换成 LangGraph 初始 State。
- 使用 `thread_id` 调用编译后的图。
- 返回 `answer`、`sources`、`confidence`、`answer_mode`。
- API 层不直接调用 Milvus、Embedding 或 LLM。

---

### `app/schemas/chat.py`

定义：

- `ChatRequest`
- `ChatResponse`
- `SourceItem`（需要时）

建议请求字段：

```python
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
```

建议响应字段：

```python
class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    answer_mode: str
```

---

### `app/agent/state.py`

定义 `EcommerceState`：

```python
class EcommerceState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]

    user_id: str
    session_id: str
    original_query: str

    intent: str
    query_type: str
    hyde_document: str | None
    rewritten_queries: list[str]

    ranked_chunks: list[dict]
    confidence: float
    is_high_confidence: bool

    existing_summary: str | None
    should_summarize: bool

    answer: str
    sources: list[str]
    answer_mode: str
    structured_output: dict
```

`messages` 必须使用 `add_messages`，其他列表字段不要错误地使用消息 Reducer。

---

### `app/agent/prompts.py`

集中保存：

- 系统提示词
- HyDE 提示词
- Multi-Query 提示词
- RAG 回答提示词
- GENERAL 回答提示词
- 低置信度兜底提示词
- 对话摘要提示词

电商回答必须遵守：

1. 商品属性、价格、库存、物流和售后只能来自检索资料。
2. 资料不足时明确说明，不允许编造。
3. 建议类回答要区分“知识库事实”和“一般建议”。
4. 不输出不存在的 SKU、优惠、承诺或时效。

---

### `app/agent/memory.py`

- 为 Agent 创建单例 `MemorySaver`。
- 根据 `user_id + session_id` 构造稳定的 `thread_id`。
- 构造 LangGraph config。
- 提供最近 10 轮滑动窗口。
- 提供是否触发摘要的判断。
- 第一版摘要保存在 State 中，不连接 PostgreSQL。

---

### `app/agent/nodes.py`

只实现 LangGraph 节点，不在本文件重复实现 Milvus、Embedding 或模型加载。

必须包含：

#### `prepare_query_node`

- 从最后一条 `HumanMessage` 提取 `original_query`。
- 初始化本轮临时字段。

#### `classify_query_node`

- 先执行通用规则和电商关键词规则。
- 规则无法确定时调用 BERT 二分类模型。
- 输出 `GENERAL / PRECISE / VAGUE / BROAD`。
- `GENERAL` 不进入知识库。

#### `hyde_generate_node`

- 只用于 `VAGUE`。
- 输入最近历史 + 当前问题。
- 输出一个字符串 `hyde_document`。
- HyDE 文档只用于检索，不是最终答案。

#### `multi_query_rewrite_node`

- 只用于 `BROAD`。
- 将问题拆成 3～5 个子问题。
- 输出 `rewritten_queries: list[str]`。
- 解析失败时回退为 `[original_query]`。

#### `retrieve_node`

- PRECISE 使用 `original_query`。
- VAGUE 使用 `hyde_document`。
- BROAD 并行检索全部 `rewritten_queries`。
- 合并结果、按 `chunk_id` 或稳定键去重。
- 调用 Reranker。
- 写入 `ranked_chunks`、`confidence`、`is_high_confidence`。

#### `generate_rag_node`

- 使用精排后的父块内容构造 context。
- 注入系统提示、摘要和最近历史。
- 调用 LLM 生成答案。
- Python 代码统一追加来源。
- `answer_mode = "rag"`。

#### `generate_general_node`

- 不查询知识库。
- 用于问候、闲聊和通用问题。
- `answer_mode = "general"`。

#### `generate_fallback_node`

- 用于知识库低置信度或无结果。
- 不允许大模型猜测商品、库存、价格、物流和政策。
- 返回“暂未找到足够可靠资料”并引导用户补充商品名、SKU 或具体问题。
- `answer_mode = "fallback"`。

#### `save_memory_node`

- 判断是否达到摘要阈值。
- 达到阈值时更新 `existing_summary`。
- 不写数据库。

---

### `app/agent/graph.py`

- 使用 `StateGraph(EcommerceState)`。
- 注册全部节点。
- 添加固定边与条件边。
- 使用 `MemorySaver` 编译。
- 导出单例 `get_ecommerce_graph()`。
- 不使用 `create_react_agent`。

路由函数保持纯函数，不调用 LLM：

- `route_after_classify`
- `route_after_retrieve`

---

### `app/classifier/rules.py`

定义轻量规则：

- 问候、感谢、闲聊 → GENERAL
- 商品、镜框、镜片、尺码、退货、物流、SKU 等关键词 → SPECIALIZED
- “它、这个、刚才那款、没懂、解释一下” → VAGUE 候选
- “全面介绍、都有哪些、怎么选择、整体讲讲” → BROAD 候选

规则只作为快速通道，避免写成巨大的 if-else 文件。

### `app/classifier/intent_classifier.py`

- 加载旧项目 BERT 二分类模型。
- 对外只暴露 `classify(text) -> tuple[str, float]`。
- 标签统一为 `general / specialized`。
- 使用单例，禁止每次请求重新加载模型。
- 同步模型推理在异步节点中通过 `asyncio.to_thread()` 调用。

---

### `app/rag/document_loader.py`

- 读取 CSV、Markdown、TXT。
- 将每条商品、FAQ 或政策记录转换为 LangChain `Document`。
- 清洗空值和重复数据。
- 保存 metadata：
  - `source`
  - `source_name`
  - `sku`
  - `category`
  - `route_intent`
  - `document_type`

### `app/rag/splitter.py`

实现父子块：

- 父块保留较完整上下文。
- 子块用于向量检索。
- 每个子块必须保存 `parent_id`。
- Milvus 记录中同时保留 `child_content` 和 `parent_content`，避免第一版再引入独立父块数据库。
- 使用稳定 hash 生成 `chunk_id` 和 `parent_id`。

### `app/rag/embeddings.py`

- 单例加载 BGE-M3。
- 提供文档批量编码和 Query 编码。
- 同时返回 dense vector 与 sparse vector。
- 不在其他文件重复加载模型。

### `app/rag/milvus_store.py`

- 创建 Collection Schema。
- 创建 dense 与 sparse 索引。
- 批量写入文档。
- 提供混合检索基础方法。
- Collection 至少保存：
  - `chunk_id`
  - `parent_id`
  - `child_content`
  - `parent_content`
  - `dense_vector`
  - `sparse_vector`
  - `source_name`
  - `sku`
  - `category`
  - `route_intent`

### `app/rag/retriever.py`

- 调用 BGE-M3 生成 Query dense + sparse。
- 使用 Milvus hybrid search。
- 使用 WeightedRanker 合并 dense 与 sparse 排名。
- 返回候选文档，不负责生成答案。
- 支持可选的 `sku`、`category`、`route_intent` 过滤条件，但第一版 API 可以暂不暴露这些参数。

### `app/rag/reranker.py`

- 单例加载 Reranker。
- 输入 `(query, candidate_document)` 对。
- 对候选结果重新打分。
- 返回统一的 `RankedDocument` 数据结构。
- 生成阶段使用 `parent_content`。

### `app/rag/ingest.py`

负责完整建库流程：

```text
读取原始数据
→ 清洗
→ 父子分块
→ BGE-M3 批量编码
→ 创建 Milvus Collection
→ 批量写入
```

该文件不得包含 FastAPI 或 LangGraph 逻辑。

---

### `app/llm/factory.py`

- 统一创建 LLM。
- 支持业务类型：`qa`、`rewrite`、`summary`。
- 缓存模型实例。
- API Key、Base URL、模型名全部来自配置。
- 不允许在节点中直接实例化 ChatOpenAI。

---

### `app/utils/logger.py`

- 统一日志格式。
- 禁止使用散落的 `print()` 作为正式日志。

### `app/utils/text.py`

放置通用文本辅助函数：

- 消息内容安全提取
- 文本清洗
- 稳定 hash
- 来源格式化

不得把业务节点塞入工具文件。

---

### `scripts/build_index.py`

使用模块方式运行完整建库：

```bash
python -m scripts.build_index
```

### `scripts/inspect_collection.py`

查看 Milvus Collection 数量、Schema 和少量样例，不修改数据。

### `scripts/test_retrieval.py`

单独测试：

```text
query
→ hybrid retrieval
→ rerank
→ 打印 Top-K、来源与分数
```

在装配 LangGraph 之前，必须先保证该脚本可运行。

---

## 7. 数据与 metadata 约定

知识库可复用旧项目的数据：

- `rag_knowledge_base.csv`
- `products.csv`
- 物流与退换货政策 Markdown/TXT

统一 metadata：

```python
{
    "source": "rag_knowledge_base.csv",
    "source_name": "退换货政策 > 退货期限",
    "sku": "FRAME-001",
    "category": "eyeglasses",
    "route_intent": "after_sales",
    "document_type": "policy",
    "parent_id": "...",
    "chunk_id": "...",
}
```

不存在的字段使用空字符串，不使用随意变化的数据结构。

---

## 8. 配置项

`.env.example` 至少包含：

```env
APP_ENV=development
APP_HOST=127.0.0.1
APP_PORT=8000

LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=

MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION=ecommerce_knowledge

BGE_M3_MODEL_PATH=
RERANKER_MODEL_PATH=
CLASSIFIER_MODEL_PATH=

DENSE_WEIGHT=0.6
SPARSE_WEIGHT=0.4
RETRIEVAL_TOP_K=12
RERANK_TOP_K=4
RAG_CONFIDENCE_THRESHOLD=0.5

PARENT_CHUNK_SIZE=1200
CHILD_CHUNK_SIZE=350
CHUNK_OVERLAP=60
MEMORY_WINDOW_SIZE=10
SUMMARY_TRIGGER_ROUNDS=10
```

数值只是默认值，代码必须允许通过环境变量修改。

---

## 9. 常用命令

```bash
# 创建虚拟环境
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# 安装依赖
python -m pip install -r requirements.txt

# 构建知识库
python -m scripts.build_index

# 单独测试检索
python -m scripts.test_retrieval

# 启动 API
python -m uvicorn app.main:app --reload

# 运行全部测试
python -m pytest -q

# 运行单个测试文件
python -m pytest tests/test_nodes.py -q
```

禁止在源码中使用 `sys.path.append()` 或 `sys.path.insert()` 修复导包。必须从项目根目录使用 `python -m ...` 运行。

---

## 10. 开发规范

- 所有公共函数必须有类型标注。
- Pydantic 模型、TypedDict、dataclass 分工明确。
- 单个函数尽量不超过 80 行，复杂逻辑拆分为辅助函数。
- 异步函数中不得直接执行长时间同步模型推理；使用 `asyncio.to_thread()`。
- 模型、Milvus Client、LLM 和 LangGraph 图使用单例或缓存，禁止每次请求重新初始化。
- API 层、Agent 层、RAG 层必须解耦。
- 节点只返回 State 更新字典，不原地随意修改 State。
- 不捕获异常后静默忽略；必须记录日志并返回明确兜底。
- 不允许在回答中虚构 SKU、价格、库存、优惠、物流时效或退换货规则。
- 任何密钥只能来自环境变量。
- 不要生成无用的抽象基类、Repository 层或复杂设计模式。
- 第一版以可运行、可读、可测试为优先，不追求过度工程化。

---

## 11. 测试要求

### 分类测试

至少覆盖：

```text
你好                              → GENERAL
TR90 镜框有什么优点？             → PRECISE
它还有其他颜色吗？                → VAGUE
怎么选择适合自己的眼镜？           → BROAD
```

### 检索测试

- dense 和 sparse 请求都能生成。
- Milvus 返回候选结果。
- Reranker 输出按分数降序排列。
- 父子块检索后生成阶段使用父块正文。
- Multi-Query 合并后不重复返回相同 chunk。

### Agent 测试

- GENERAL 不调用检索器。
- PRECISE 直接进入 retrieve。
- VAGUE 先生成 `hyde_document`。
- BROAD 先生成 `rewritten_queries`。
- 高置信度返回 `answer_mode="rag"`。
- 低置信度返回 `answer_mode="fallback"`。
- 同一 `thread_id` 的第二轮能读取第一轮消息。
- 不同 `session_id` 的历史互不干扰。

### API 测试

请求：

```json
{
  "user_id": "user_001",
  "session_id": "session_001",
  "message": "TR90 镜框有什么优点？"
}
```

响应至少包含：

```json
{
  "answer": "...",
  "sources": ["..."],
  "confidence": 0.0,
  "answer_mode": "rag"
}
```

---

## 12. 实施顺序

Claude Code 不要一次性生成全部业务代码。严格按以下顺序实施，每一步完成后先运行测试：

1. 创建目录、配置、Schema 与 README。
2. 实现文档读取和父子分块。
3. 实现 BGE-M3 与 Milvus 建库。
4. 实现 hybrid retrieval 与 reranker。
5. 使用 `scripts/test_retrieval.py` 验证 RAG 核心。
6. 实现 State、Prompts、分类器和 MemorySaver。
7. 实现 PRECISE / VAGUE / BROAD 节点。
8. 实现生成与兜底节点。
9. 组装 LangGraph。
10. 实现 FastAPI。
11. 完成单元测试和端到端测试。

每一步开始前：

- 先说明准备创建或修改哪些文件。
- 不修改当前步骤之外的无关文件。
- 完成后列出改动和运行命令。
- 出现不确定的模型接口或数据字段时，不得猜测，先检查现有文件。

---

## 13. 验收标准

项目至少满足：

1. `python -m scripts.build_index` 能完成建库。
2. `python -m scripts.test_retrieval` 能输出合理 Top-K。
3. `python -m pytest -q` 通过。
4. `python -m uvicorn app.main:app --reload` 能启动。
5. `/health` 返回正常状态。
6. `/api/chat` 能完成 GENERAL、PRECISE、VAGUE、BROAD 四条路径。
7. 相同 session 能连续追问。
8. 无检索资料时不编造电商事实。
9. 项目中不存在 MCP、Web Search、`sys.path.append()` 和硬编码密钥。





## 需要用户本地补充的文件与配置 本项目中的模型文件、模型路径、API Key、数据库连接信息等内容，需要由我在本地准备或手动填写。 Claude Code 在创建此类待填写文件时，应遵循以下约定： 1. 尚未填写完成的配置文件，文件名末尾添加下划线 `_`。 2. 文件内部必须使用注释明确标记需要用户修改的位置。 3. 不得擅自猜测用户电脑上的模型路径。 4. 不得在代码中硬编码个人电脑路径、API Key 或数据库密码。 5. 不得使用虚假路径让程序看起来可以运行。 6. 用户填写完成后，再将文件末尾的 `_` 删除，改为程序实际使用的文件名。 ### 文件命名示例 ```text .env_ model_paths_.yaml local_settings_.py 