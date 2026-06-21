# 🎬 LangGraph 多 Agent RAG 电影推荐系统

基于 **LangGraph + LangChain + FAISS** 的多 Agent 协同 RAG 推荐平台。

> 自然语言描述 → 3 路并行召回 (Send API) → 去重 → LLM 生成中文推荐

---

## 目录

- [系统架构](#系统架构)
- [LangGraph 多 Agent 协作](#langgraph-多-agent-协作)
- [LangChain 集成](#langchain-集成)
- [RAG 检索增强生成](#rag-检索增强生成)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 文档](#api-文档)
- [存储架构](#存储架构)
- [提示词配置](#提示词配置)
- [配置项](#配置项)
- [Docker 部署](#docker-部署)

---

## 系统架构

```
  用户: "推荐烧脑悬疑片，结局要有反转"
      │
      ▼
┌──────────────────────────────────────────────────┐
│              recall_dispatcher                   │
│              (LangGraph Send API 并行调度)        │
│                                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ semantic     │ │ keyword      │ │ llm_expand│ │
│  │ _recall      │ │ _recall      │ │ _recall   │ │
│  │ (FAISS 语义) │ │ (中英关键词)  │ │ (LLM 扩写)  │ │
│  └──────┬───────┘ └──────┬───────┘ └─────┬─────┘ │
│         │                │                │      │
│         └────────────────┴────────────────┘      │
│                          │ (operator.add 合并)    │
│                  ┌───────▼───────┐               │
│                  │recommend_agent│               │
│                  │ 去重 + LLM 推荐│               │
│                  └───────┬───────┘               │
│                          │                       │
│          ┌───────────────┼───────────────┐       │
│          ▼               ▼               ▼       │
│    短期记忆(内存)   长期记忆(SQLite)   全局记忆       │
│    缓存(Redis)     FAISS 向量索引      数据库(MySQL)│
└──────────────────────────────────────────────────┘
      │
      ▼
  **致命魔术**（The Prestige, 2006）
  类型：剧情、悬疑、惊悚
  推荐理由：诺兰通过双魔术师的对决构建错综复杂的叙事结构，层层反转...
```

---

## LangGraph 多 Agent 协作

### 什么是 LangGraph

LangGraph 是 LangChain 生态中的 Agent 编排框架。它用**有向图 (StateGraph)** 定义多个处理节点和节点间的流转逻辑，支持条件路由、循环和并行执行。

本系统使用 LangGraph 的核心特性：

| 特性 | 用途 | 代码位置 |
|---|---|---|
| `StateGraph` | 定义 6 个节点的处理流水线 | `agent.py` |
| `MessagesState` | 内置消息管理 (`operator.add` 归并) | `state.py` |
| `Send API` | 3 路并行召回同时执行 | `nodes.py:recall_dispatcher` |
| `operator.add` | 多路结果自动拼接合并 | `state.py:candidates` |
| `add_conditional_edges` | 条件路由（Send 分发） | `agent.py` |

### 图结构 (6 节点, 3 路并行)

```
START
  │
user_agent              ← 解析查询 + 加载三层记忆上下文
  │
recall_dispatcher       ← 返回 list[Send], LangGraph 并行执行 3 路
  ├── semantic_recall   ← SentenceTransformer → FAISS 余弦检索
  ├── keyword_recall    ← 57 组中英关键词映射
  └── llm_expand_recall ← LLM 改写查询 → FAISS 二次检索
  │ (operator.add 自动合并)
  ▼
recommend_agent         ← 去重 + LLM 精排 + 缓存写入 + 持久化记忆
  │
 END
```

### Send API 并行原理

```python
# nodes.py
def recall_dispatcher(state: RecommenderState) -> list[Send]:
    return [
        Send("semantic_recall", state),   # 同时执行
        Send("keyword_recall", state),    # 同时执行
        Send("llm_expand_recall", state), # 同时执行
    ]

# agent.py
agent_builder.add_conditional_edges("user_agent", recall_dispatcher)  # 分发
agent_builder.add_edge("semantic_recall", "recommend_agent")   # 汇聚
agent_builder.add_edge("keyword_recall", "recommend_agent")    # 汇聚
agent_builder.add_edge("llm_expand_recall", "recommend_agent") # 汇聚
```

`recall_dispatcher` 返回 `list[Send]` 后，LangGraph 同时启动 3 个子节点。每个子节点独立执行，返回的 `candidates` 通过 `operator.add` 自动累积合并。3 路全部完成后，汇聚到 `recommend_agent`。

### 状态管理

```python
class RecommenderState(MessagesState):
    query: str                              # 用户自然语言
    session_id: str                         # 会话 ID (记忆追踪)
    candidates: Annotated[list[dict], operator.add]  # 并行召回自动合并
    memory_context: dict                    # 三层记忆上下文
    final_response: str                     # LLM 推荐回答
```

---

## LangChain 集成

### LLM 调用

```python
# tools.py — 模型初始化
model = ChatOpenAI(model="qwen-turbo", temperature=0)

# nodes.py — LLM 生成推荐
prompt = prompts.render("recommend.system", query=query, movies=movies_text)
response = model.invoke([{"role": "system", "content": prompt}])
```

系统使用 LangChain 的 `ChatOpenAI` 客户端调用 LLM，共调用 2 次：

| 调用位置 | 用途 | Prompt 来源 |
|---|---|---|
| `llm_expand_recall` | 将用户中文查询改写为英文关键词 | `prompts.yml:llm_expand` |
| `recommend_agent` | 阅读候选集 + 用户查询 → 生成推荐回答 | `prompts.yml:recommend` |

### 提示词工程

提示词配置在 `prompts.yml` 中，通过 `prompts.py` 加载和模板渲染：

```yaml
recommend:
  system: |
    你是一位资深的电影推荐专家...
    ⚠️ 严格遵循以下格式：
    1. **中文片名**（English Title, 年份）
       - **类型**：中文类型1、中文类型2
       - **推荐理由**：写2-3句中文推荐理由
```

运行时 `prompts.render("recommend.system", query=..., movies=...)` 填充模板变量后传给 LLM。修改提示词只需编辑 YAML，无需改代码。

### 工具定义 (LangChain Tools)

```python
@tool
def semantic_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """FAISS 语义检索：直接用用户原始查询检索。"""
    ...

@tool
def keyword_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """关键词匹配：中英文关键词映射 → 匹配电影标签和类型。"""
    ...

@tool
def llm_expand_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """LLM 扩展查询 → FAISS 检索。用 LLM 将用户查询改写为更丰富的英文关键词后再检索。"""
    ...
```

三个工具均用 `@tool` 装饰器注册，可被 LangGraph 节点直接 `.invoke()` 调用。

---

## RAG 检索增强生成

### 完整 RAG 流程

```
用户自然语言: "推荐烧脑悬疑片，结局要有反转"
      │
      ▼
┌─────────────────┐
│  查询向量化      │  SentenceTransformer 编码 → 384 维向量
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS 检索     │  余弦相似度搜索 100 部电影向量
│  (3 路并行)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  候选去重融合    │  movie_id 去重 + 分数择优
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM 增强生成   │  提示词: 候选列表 + 用户查询 → 推荐回答
└─────────────────┘
```

### FAISS 索引

- **模型**: `all-MiniLM-L6-v2` (SentenceTransformer, 384 维)
- **数据**: 100 部电影的 `title_en + genres + tags + description` 拼接文本
- **相似度**: 内积 (Inner Product) = 归一化后的余弦相似度
- **持久化**: `.faiss_index/movie_index.faiss` (150KB) + `movie_meta.pkl` (28KB)
- **降级策略**: SBERT → TF-IDF (scikit-learn) → 关键词匹配

### 3 路召回对比

| 路 | 输入 | 检索方式 | 返回数 |
|---|---|---|---|
| `semantic_recall` | 用户原始查询 | FAISS 语义相似 | Top 8 |
| `keyword_recall` | 中文关键词映射 | 标签/类型匹配 | Top 8 |
| `llm_expand_recall` | LLM 英文扩写 | FAISS 语义相似 | Top 8 |

三路并行执行，`operator.add` 自动合并，`recommend_agent` 统一去重后传给 LLM。

---

## 项目结构

```
langgraph_study/
├── README.md
├── .env                              # 环境变量 (LLM Key, DB, Redis, HF)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
└── backend/
    ├── prompts.yml                   # 提示词 YAML 配置
    ├── database.py                   # SQLAlchemy engine & session
    ├── cache.py                      # Redis / 内存双模缓存
    ├── data.py                       # 100 部电影数据定义
    ├── seed_db.py                    # 建表 + 灌种子数据
    │
    ├── server.py                     # FastAPI 入口 (REST + WebSocket)
    ├── streamlit_app.py              # Streamlit 聊天式界面
    ├── main.py                       # CLI 测试工具
    │
    ├── models/                       # SQLAlchemy ORM
    │   ├── movie.py                  #   Movie (id, title_en/cn, year, genres, tags, description)
    │   └── user.py                   #   User (id, name)
    ├── schemas/                      # Pydantic 请求/响应
    │   ├── movie.py                  #   MovieOut, MovieCreate, MovieUpdate, MovieListOut
    │   └── user.py                   #   UserOut, UserListOut
    ├── crud/                         # 数据库 CRUD 操作
    │   ├── movie.py                  #   get/create/update/delete, seed
    │   └── user.py                   #   get/list, seed
    ├── routers/                      # FastAPI 路由
    │   ├── movie.py                  #   GET/POST/PUT/DELETE /api/movies
    │   └── user.py                   #   GET /api/users
    │
    └── my_agent/                     # LangGraph 推荐引擎
        ├── agent.py                  #   StateGraph 构建 (6 节点, Send API)
        └── utils/
            ├── state.py              #   RecommenderState 定义
            ├── nodes.py              #   6 Agent 节点 + 记忆集成
            ├── tools.py              #   3 路召回工具 + LLM 模型
            ├── rag.py                #   FAISS 索引 & 检索 (三级降级)
            ├── memory.py             #   三层记忆系统 (短期/长期/全局)
            └── prompts.py            #   提示词 YAML 加载 & 模板渲染
```

---

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 配置 `.env`

```env
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
HF_ENDPOINT=https://hf-mirror.com     # 国内必须
DB_TYPE=sqlite                         # sqlite 零配置 / mysql 需先建库
HOST=127.0.0.1
PORT=8000
```

### 3. 种子数据

```bash
python seed_db.py          # 建表 + 灌入 100 部电影
```

### 4. 启动

```bash
# API → http://127.0.0.1:8000/docs
python server.py

# 管理后台 → http://127.0.0.1:8501
streamlit run streamlit_app.py

# CLI
python main.py -q "推荐烧脑悬疑片，结局要有反转"
```

---

## API 文档

### 推荐接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/recommend` | 自然语言推荐 `{"query":"推荐科幻片"}` |
| `GET` | `/health` | 健康检查 |
| `GET` | `/stats` | 服务统计 |
| `GET` | `/api/agent/graph` | Agent 图结构信息 |
| `GET` | `/api/server/info` | 服务运行状态 |

### 电影 CRUD

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/api/movies` | 全部电影 |
| `GET` | `/api/movies/hot` | 热门电影 |
| `GET` | `/api/movies/{id}` | 单部电影 |
| `POST` | `/api/movies` | 新增电影 |
| `PUT` | `/api/movies/{id}` | 更新电影 |
| `DELETE` | `/api/movies/{id}` | 删除电影 |

### 请求示例

```json
POST /recommend
{ "query": "推荐类似致命魔术的电影，魔术师对决那种" }

→ {
  "query": "推荐类似致命魔术的电影...",
  "answer": "**致命魔术**（The Prestige, 2006）\n类型：剧情、悬疑、惊悚\n推荐理由：...",
  "retrieved_count": 16,
  "elapsed_ms": 17807
}
```

---

## 存储架构

| 存储层 | 技术 | 位置 | 内容 |
|---|---|---|---|
| 电影数据 | MySQL / SQLite | `recommender` 库 | 100 部电影 (title_en/cn, year, genres, tags, description) |
| 向量索引 | FAISS | `.faiss_index/` | 384 维电影语义嵌入 (150KB) |
| 三层记忆 | SQLite | `recommender_memory.db` | interaction_history, user_preferences, recommendation_log |
| 缓存 | Redis / 内存 | — | 查询级缓存 (TTL 600s，相同问题秒回) |

### 记忆系统

| 层级 | 存储 | 生命周期 | 内容 |
|---|---|---|---|
| 短期 | 内存 dict | 会话级 | 当前 session 浏览过的电影 ID |
| 长期 | SQLite | 持久化 | 交互记录、偏好画像、推荐日志 |
| 全局 | SQLite + 内存 | 定时刷新 | 热门电影排行 |

记忆在推荐流程中：`user_agent` 加载上下文传给 LLM → `recommend_agent` 写入新的交互记录。

---

## 提示词配置

编辑 `backend/prompts.yml`，两个 Prompt 模板：

| 节点 | 用途 | 说明 |
|---|---|---|
| `llm_expand.system` | LLM 查询扩展 | 中文 → 英文关键词 |
| `recommend.system` | LLM 生成推荐 | 候选集 + 用户查询 → 中文推荐 |

支持 `{query}` 和 `{movies}` 模板变量，运行时由 `prompts.py` 渲染。

---

## 配置项

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | — | LLM API Key |
| `DASHSCOPE_BASE_URL` | — | LLM API 地址 |
| `HF_ENDPOINT` | — | 国内必设 `https://hf-mirror.com` |
| `DB_TYPE` | sqlite | sqlite / mysql |
| `DB_HOST` | localhost | MySQL 地址 |
| `DB_PATH` | recommender_memory.db | SQLite 记忆库路径 |
| `FAISS_INDEX_DIR` | .faiss_index | 向量索引目录 |
| `REDIS_HOST` | localhost | Redis 地址 (可选) |
| `REDIS_PROTOCOL` | 3 | Redis<6.0 改为 2 |
| `CACHE_TTL` | 300 | 缓存过期时间 (秒) |
| `HOST` | 127.0.0.1 | API 地址 |
| `PORT` | 8000 | API 端口 |

---

## Docker

```bash
docker-compose up -d
```

| 服务 | 端口 | 地址 |
|---|---|---|
| FastAPI | 8000 | http://localhost:8000/docs |
| Streamlit | 8501 | http://localhost:8501 |
| Redis | 6379 | 缓存 |

---

## License

MIT
