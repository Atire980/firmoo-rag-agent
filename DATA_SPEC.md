# 电商 RAG Agent 数据与知识库规范

> 本文件用于指导 Vibe Coding 工具理解项目数据、设计数据加载流程，并生成与真实数据结构一致的代码。  
> 建议将本文件保存为项目根目录下的 `DATA_SPEC.md`，与 `CLAUDE.md` 配合使用。

---

## 1. 文档目的

本项目将原有眼镜电商 RAG 项目升级为一个基于 LangGraph 的工作流型客服 Agent。

数据层负责支持以下能力：

- 电商意图识别
- 商品知识问答
- 镜片与度数咨询
- 尺码、材质与脸型建议
- 物流、退换货与售后政策问答
- 多轮上下文测试
- RAG 检索效果评估
- 商品、订单与对话等结构化数据管理

第一版项目保留 LangGraph、BGE-M3、Milvus、Reranker、HyDE、Multi-Query 和 MemorySaver，不使用 MCP，不使用联网搜索。

---

## 2. 数据集说明

本项目使用的是参考跨境眼镜电商业务构建的模拟数据集，不是真实网站爬取数据，也不包含真实用户隐私。

### 2.1 文件清单

| 文件名 | 数据量 | 主要用途 | 推荐位置 |
|---|---:|---|---|
| `intent_train.csv` | 5000 条 | 训练电商意图分类模型 | `data/raw/intent_train.csv` |
| `rag_knowledge_base.csv` | 8380 条 | 商品、FAQ、政策、评论的统一 RAG 知识库 | `data/raw/rag_knowledge_base.csv` |
| `products.csv` | 1000 条 | 商品价格、库存、尺寸、材质等结构化信息 | `data/raw/products.csv` |
| `orders.csv` | 2000 条 | 模拟订单与物流信息 | `data/raw/orders.csv` |
| `conversations.csv` | 4000 条 | 多轮客服对话样本 | `data/raw/conversations.csv` |
| `rag_test_questions.csv` | 300 条 | 检索与回答效果测试集 | `data/raw/rag_test_questions.csv` |
| `schema.sql` | 1 份 | MySQL 建表脚本 | `sql/schema.sql` |
| `DATA_SPEC.md` | 1 份 | 数据规则与开发约束 | 项目根目录 |

### 2.2 上传文件重命名规则

当前上传文件名中可能带有 `(1)`、`(2)` 等后缀。放入项目后统一改为以下标准名称：

```text
conversations.csv
intent_train.csv
orders.csv
products.csv
rag_knowledge_base.csv
rag_test_questions.csv
schema.sql
```

业务代码中只能引用标准文件名，不得把 `(1)`、`(2)` 写死在路径中。

---

## 3. 推荐目录结构

```text
ecommerce-rag-agent/
├── CLAUDE.md
├── DATA_SPEC.md
│
├── data/
│   ├── raw/
│   │   ├── intent_train.csv
│   │   ├── rag_knowledge_base.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── conversations.csv
│   │   └── rag_test_questions.csv
│   │
│   ├── samples/
│   │   ├── intent_train_sample.csv
│   │   ├── rag_knowledge_base_sample.csv
│   │   └── products_sample.csv
│   │
│   └── processed/
│       ├── classifier/
│       ├── chunks/
│       └── reports/
│
├── sql/
│   └── schema.sql
│
├── models/
│   ├── intent_classifier/
│   ├── bge_m3/
│   └── reranker/
│
├── scripts/
│   ├── inspect_data.py
│   ├── build_index.py
│   ├── train_classifier.py
│   └── evaluate_rag.py
│
└── app/
```

### 3.1 目录规则

- `data/raw/` 中的原始文件只读，程序不得修改或覆盖。
- 清洗、转换、分块后的结果写入 `data/processed/`。
- 用于快速测试的小样本写入 `data/samples/`。
- 模型文件统一放入 `models/`，禁止散落在业务目录。
- SQL 文件统一放入 `sql/`。
- 文件路径必须从配置读取，不得在业务代码中写死绝对路径。

---

## 4. `intent_train.csv`

### 4.1 文件用途

该文件用于训练电商意图识别模型。

核心流程：

```text
用户问题
→ 意图分类器
→ 输出业务意图
→ 决定是否走 RAG
→ 决定知识库过滤范围
```

### 4.2 字段说明

| 字段 | 类型 | 含义 |
|---|---|---|
| `sample_id` | string | 样本唯一编号 |
| `query` | string | 用户问题 |
| `label` | string | 电商业务意图标签 |

### 4.3 意图标签

| 标签 | 数量 | 处理建议 |
|---|---:|---|
| `商品咨询` | 850 | 检索 product、review、faq |
| `镜片/度数咨询` | 850 | 检索 product、policy、faq |
| `退换货售后` | 850 | 检索 policy、faq |
| `物流查询` | 650 | 第一版检索物流政策；后续可查询订单表 |
| `订单修改/取消` | 550 | 第一版检索政策；后续可查询订单表 |
| `投诉/人工客服` | 450 | 输出转人工建议并记录会话 |
| `优惠/支付` | 450 | 检索 faq、policy |
| `闲聊/其他` | 350 | 走 GENERAL 分支，不查询知识库 |

### 4.4 与 LangGraph 路由的关系

当前数据是八分类数据，而 Agent 的第一层路由是 `GENERAL / SPECIALIZED`。

建议采用以下映射：

```text
闲聊/其他
→ GENERAL

其余七类
→ SPECIALIZED
```

业务意图标签仍然保存在 State 中，用于知识库过滤：

```python
business_intent = "退换货售后"
query_scope = ["policy", "faq"]
```

`PRECISE / VAGUE / BROAD` 不是这八个业务标签，而是第二层检索策略：

```text
第一层：GENERAL / SPECIALIZED
第二层：商品咨询、物流查询、退换货售后等业务意图
第三层：PRECISE / VAGUE / BROAD 检索策略
```

不得把这三种分类混成一个模型标签集合。

---

## 5. `rag_knowledge_base.csv`

### 5.1 文件用途

该文件是统一 RAG 知识库，包含：

- 商品知识
- FAQ
- 业务政策
- 用户评论

只有该文件中的知识内容应直接进入 Milvus。

### 5.2 字段说明

| 字段 | 类型 | 含义 | Milvus 用途 |
|---|---|---|---|
| `doc_id` | string | 知识记录唯一编号 | 主键或外部标识 |
| `source_type` | string | `product / faq / policy / review` | 过滤字段 |
| `source_id` | string | 原始来源编号 | 溯源字段 |
| `sku` | string / null | 关联商品编号 | 商品过滤字段 |
| `category` | string | 知识类别 | 过滤字段 |
| `title` | string | 文档标题 | 上下文与来源展示 |
| `content` | string | 用于向量化的正文 | dense / sparse 向量输入 |
| `route_intent` | string | 适用业务意图 | 意图过滤字段 |
| `metadata_json` | JSON string | 扩展元数据 | 解析后保存为 metadata |

### 5.3 来源分布

| `source_type` | 数量 |
|---|---:|
| `review` | 5000 |
| `product` | 3000 |
| `faq` | 300 |
| `policy` | 80 |

### 5.4 向量化规则

必须向量化：

```text
title + content
```

建议输入格式：

```python
embedding_text = f"{title}\n{content}".strip()
```

不得向量化：

- `doc_id`
- 原始 JSON 字符串
- 纯价格数字
- 纯库存状态
- 无意义编号

### 5.5 Metadata 规则

写入 Milvus 的 metadata 至少保留：

```python
{
    "doc_id": "...",
    "source_type": "...",
    "source_id": "...",
    "sku": "...",
    "category": "...",
    "title": "...",
    "route_intent": ["商品咨询", "镜片/度数咨询"],
    "metadata": {...}
}
```

`route_intent` 原始值可能是：

```text
商品咨询;镜片/度数咨询
```

加载时必须按分号拆成列表：

```python
["商品咨询", "镜片/度数咨询"]
```

### 5.6 空值规则

`sku` 有部分空值，这是正常现象。

非商品类知识，例如 FAQ、政策，不一定关联具体商品。不得因为 `sku` 为空就删除该知识。

处理方式：

```python
sku = None if pd.isna(row["sku"]) else str(row["sku"]).strip()
```

### 5.7 检索过滤建议

| 业务意图 | 推荐 `source_type` |
|---|---|
| 商品咨询 | product、review、faq |
| 镜片/度数咨询 | product、policy、faq |
| 物流查询 | policy、faq |
| 退换货售后 | policy、faq |
| 订单修改/取消 | policy、faq |
| 优惠/支付 | faq、policy |
| 投诉/人工客服 | policy、faq |

过滤不是强制只保留单一类型。可以先按 `route_intent` 筛选，再进行 dense + sparse 混合检索。

### 5.8 分块策略

该文件本身已经是知识块集合，第一版不要再次进行过度细分。

推荐规则：

- `content` 较短时，直接作为一个检索子块。
- `content` 超过配置阈值时，才做二次切分。
- `title + source_id` 可作为父级语义信息。
- 子块负责检索，生成回答时尽量补充同一 `source_id` 下的关联内容。
- 不得把不同 SKU 或不同政策的内容合并成同一个父块。

---

## 6. `products.csv`

### 6.1 文件用途

该文件存放结构化商品数据，适合导入 MySQL，也可以在第一版中通过只读 Repository 查询。

适合回答：

- 商品价格
- 当前库存状态
- 镜框尺寸
- 镜框材质
- 商品颜色
- 是否支持处方、蓝光、渐进、双光或变色镜片
- 商品评分

### 6.2 字段说明

主要字段包括：

```text
sku
title
product_type
gender
style
frame_type
frame_shape
material
color
size
total_width_mm
lens_width_mm
bridge_width_mm
temple_length_mm
weight_g
face_shape_recommendation
prescription_supported
blue_light_supported
progressive_supported
bifocal_supported
photochromic_supported
sunglasses_lens_supported
stock_status
price_usd
average_rating
review_count
features
created_at
```

### 6.3 使用原则

以下问题优先精确查询结构化商品表：

```text
FM00001 多少钱？
FM00001 有没有库存？
这款镜框宽度是多少？
这款是否支持渐进镜片？
```

以下问题适合 RAG 语义检索：

```text
什么镜框适合圆脸？
轻量又有弹性的材质有哪些？
长期办公适合什么镜片？
```

禁止让大模型推测价格、库存、SKU、尺寸和支持能力。

### 6.4 第一版边界

第一版只允许读取商品信息，不允许：

- 修改库存
- 修改价格
- 创建商品
- 删除商品
- 下单
- 支付

---

## 7. `orders.csv`

### 7.1 文件用途

该文件包含模拟订单与物流数据。

主要字段：

```text
order_id
user_id
sku
order_date
order_status
shipping_country
tracking_number
shipped_date
estimated_delivery_date
is_prescription_order
lens_type
amount_usd
```

### 7.2 第一版处理方式

第一版 Agent 暂不实现真实订单查询节点。

该文件用于：

- 后续订单查询节点开发
- Repository 和 SQL 测试
- 物流问答场景设计
- 端到端演示数据准备

第一版用户询问具体订单时，应明确说明当前版本无法访问真实订单状态，不得根据知识库或语言模型编造。

### 7.3 空值说明

`tracking_number` 和 `shipped_date` 存在空值，主要出现在：

- `processing`
- `lens_processing`
- `cancelled`

这些空值符合业务逻辑，不得全部填充为虚假数据，也不得直接删除相关订单。

---

## 8. `conversations.csv`

### 8.1 文件用途

该文件是多轮客服对话样本。

字段包括：

| 字段 | 含义 |
|---|---|
| `conversation_id` | 会话编号 |
| `message_id` | 消息编号 |
| `turn_index` | 消息轮次 |
| `role` | `user / assistant` |
| `intent` | 当前业务意图 |
| `content` | 消息正文 |
| `created_at` | 创建日期 |

### 8.2 使用方式

该文件主要用于：

- 构造多轮测试样例
- 测试 VAGUE 指代问题
- 测试 MemorySaver
- 测试滑动窗口
- 分析不同业务意图中的对话模式

不得把历史 CSV 全部一次性加载进每次模型 Prompt。

### 8.3 与 MemorySaver 的关系

```text
conversations.csv
→ 离线样例和评估数据

MemorySaver
→ Agent 运行时的当前会话记忆
```

两者不是同一个系统。

---

## 9. `rag_test_questions.csv`

### 9.1 文件用途

该文件用于自动测试 RAG 检索效果。

字段包括：

| 字段 | 含义 |
|---|---|
| `test_id` | 测试编号 |
| `question` | 测试问题 |
| `expected_source` | 理想知识来源 |
| `difficulty` | 问题难度 |

### 9.2 评估指标

至少实现：

- Recall@K
- MRR
- Top-1 来源命中率
- Top-3 来源命中率
- 平均检索耗时
- Reranker 前后命中率对比

回答质量可以额外统计：

- 是否引用来源
- 是否出现知识库外编造
- 是否正确拒绝低置信度回答

### 9.3 测试流程

```text
读取 question
→ 执行混合检索
→ 获取 Top-K
→ 对比 expected_source
→ 输出评估报告
```

评估结果写入：

```text
data/processed/reports/
```

---

## 10. `schema.sql`

### 10.1 文件用途

该脚本创建以下 MySQL 表：

- `products`
- `orders`
- `conversations`

RAG 知识库不写入 MySQL，向量化后写入 Milvus。

### 10.2 数据存储边界

```text
Milvus
→ 非结构化知识与语义检索

MySQL
→ 商品、订单、对话等结构化业务数据

MemorySaver
→ 当前 Agent 会话状态
```

不得把三者混为一套存储。

---

## 11. Agent 数据流

### 11.1 通用问题

```text
用户问题
→ 意图分类为 闲聊/其他
→ GENERAL
→ LLM 直接回答
```

### 11.2 电商专业问题

```text
用户问题
→ 八分类业务意图
→ SPECIALIZED
→ 判断 PRECISE / VAGUE / BROAD
→ BGE-M3 dense + sparse
→ Milvus 混合检索
→ Reranker
→ 置信度判断
→ 生成带来源的回答
```

### 11.3 VAGUE 分支

```text
最近对话历史 + 当前模糊问题
→ HyDE 生成完整检索文本
→ Milvus 检索
```

例如：

```text
上一轮：这款 TR90 镜框重量很轻。
当前轮：它适合圆脸吗？
```

### 11.4 BROAD 分支

```text
宽泛问题
→ Multi-Query 拆为 3～5 个子问题
→ 分别检索
→ 合并与去重
→ Reranker
```

例如：

```text
怎么选择适合自己的眼镜？
```

可拆为：

```text
如何根据脸型选择镜框？
如何根据脸宽选择尺寸？
不同镜框材质有什么区别？
如何根据度数选择镜片？
```

---

## 12. 数据加载与检查要求

### 12.1 建库前检查

必须检查：

- 文件是否存在
- 编码是否为 UTF-8
- 列名是否完整
- 主键是否重复
- `content` 是否为空
- `route_intent` 是否合法
- `metadata_json` 是否能解析
- 数值列是否能转换
- 日期列是否能解析

### 12.2 原始数据保护

禁止：

- 直接覆盖 `data/raw/`
- 静默修改列名
- 自动删除无法解析的记录
- 用语言模型补造缺失价格、库存、物流单号
- 将完整原始数据提交到日志

异常数据必须记录到报告：

```text
data/processed/reports/data_validation_report.json
```

### 12.3 数据加载器返回格式

推荐统一返回：

```python
{
    "content": "...",
    "metadata": {
        "doc_id": "...",
        "source_type": "...",
        "source_id": "...",
        "sku": None,
        "category": "...",
        "title": "...",
        "route_intent": [...],
        "metadata": {...}
    }
}
```

---

## 13. Milvus 建议字段

Milvus Collection 至少包含：

```text
id
doc_id
content
dense_vector
sparse_vector
source_type
source_id
sku
category
title
route_intent
metadata_json
```

### 13.1 索引规则

- `dense_vector`：用于语义检索。
- `sparse_vector`：用于关键词检索。
- 两路召回通过 WeightedRanker 或同类策略融合。
- 融合结果交给 BGE-Reranker / CrossEncoder 精排。
- 最终只把 Top-K 高质量内容交给生成节点。

### 13.2 过滤规则

Milvus 表达式必须由代码安全构造。

禁止直接拼接未经清洗的用户字符串。

---

## 14. 来源展示规则

最终 RAG 回答必须返回来源。

推荐格式：

```text
📚 参考来源
- 商品知识：FM00001
- 退换货政策
- FAQ：处方镜片填写
```

来源优先使用：

```text
title
source_type
source_id
sku
```

不得让大模型自己凭空生成来源名称。来源必须由 Python 根据检索 metadata 追加。

---

## 15. Vibe Coding 开发约束

Claude Code 或其他 Vibe Coding 工具开始编码前，必须先完成以下步骤：

1. 阅读 `CLAUDE.md`。
2. 阅读 `DATA_SPEC.md`。
3. 检查 `data/raw/` 中真实文件的列名。
4. 输出数据检查结果。
5. 展示准备创建的目录和文件。
6. 经确认后再开始实现。

### 15.1 禁止事项

- 不使用 MCP。
- 不添加联网搜索。
- 不擅自更换字段名。
- 不将八分类意图直接改成无依据的新标签。
- 不用向量检索回答精确价格和库存问题。
- 不编造订单、物流、价格、库存或售后政策。
- 不一次生成全部项目后才测试。
- 不在每个请求中重复加载 BGE-M3、BERT 或 Reranker。
- 不把 API Key 写入代码或 Markdown。
- 不修改原始 CSV。

### 15.2 推荐实施顺序

```text
阶段 1：检查数据与目录
阶段 2：实现配置和数据加载器
阶段 3：实现意图分类器
阶段 4：实现 BGE-M3 与 Milvus 建库
阶段 5：实现混合检索与 Reranker
阶段 6：单独测试 retrieve()
阶段 7：实现 LangGraph State 和节点
阶段 8：实现 PRECISE / VAGUE / BROAD
阶段 9：实现 MemorySaver
阶段 10：接入 FastAPI
阶段 11：执行 RAG 测试集评估
```

---

## 16. 第一条 Vibe Coding 指令模板

```text
请先完整阅读项目根目录中的 CLAUDE.md 和 DATA_SPEC.md。

然后只执行以下工作：

1. 检查 data/raw/ 下全部 CSV 的文件名、行数、列名、空值和重复值；
2. 检查 schema.sql 中的表结构；
3. 生成 data_validation_report.json；
4. 展示你建议的项目目录树；
5. 说明每个数据文件会被哪个模块使用；
6. 暂时不要实现 BGE-M3、Milvus、LangGraph、FastAPI 和大模型调用。

不得修改任何原始数据文件。
```

---

## 17. 验收标准

数据层完成后必须满足：

- 能正常读取全部 CSV。
- 能识别并保留合法空值。
- 能解析 `route_intent`。
- 能解析 `metadata_json`。
- 能区分 Milvus 数据与 MySQL 数据。
- 能输出统一 Document 数据结构。
- 能按业务意图构造检索过滤条件。
- 能从测试集运行检索评估。
- 原始文件没有被修改。
- 日志中不包含完整用户数据或密钥。
- 项目代码不依赖 MCP 或联网搜索。

---

## 18. 项目数据定位总结

```text
intent_train.csv
→ 训练电商业务意图分类器

rag_knowledge_base.csv
→ 构建 Milvus RAG 知识库

products.csv
→ 精确查询商品结构化属性

orders.csv
→ 后续订单与物流查询扩展

conversations.csv
→ 多轮对话测试与会话样例

rag_test_questions.csv
→ RAG 检索评估

schema.sql
→ 创建 MySQL 结构化业务表
```

项目的核心原则是：

> 语义知识交给 RAG，确定性字段交给结构化查询，多轮状态交给 LangGraph MemorySaver，任何缺失业务事实都不得由大模型自行编造。
