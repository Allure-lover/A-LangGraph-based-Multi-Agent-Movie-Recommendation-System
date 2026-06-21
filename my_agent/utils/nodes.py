"""多 Agent RAG 节点模块（集成三层记忆）。

Agent:
    user_agent          解析查询 + 记忆上下文
    recall_dispatcher   Send API 3 路并行调度
    semantic_recall     FAISS 语义检索
    keyword_recall      中英关键词匹配
    llm_expand_recall   LLM 扩写查询 → FAISS
    recommend_agent     去重 + LLM 推荐 + 持久化记忆
"""

import uuid

from langgraph.types import Send

from . import prompts
from .memory import global_memory, short_term_memory, sqlite_manager
from .state import RecommenderState
from .tools import keyword_retrieve, llm_expand_retrieve, model, semantic_retrieve


# ── Agent 1：解析查询 + 记忆 ──
def user_agent(state: RecommenderState) -> dict:
    """提取查询，初始化三层记忆，加载上下文。"""
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""

    # 会话 ID
    session_id = state.get("session_id", str(uuid.uuid4())[:8])

    # 短期记忆：初始化会话
    short_term_memory.init_session(session_id)

    # 长期记忆：加载历史交互
    recent_long = sqlite_manager.get_user_interactions(session_id, limit=5)

    # 全局记忆：热门电影
    global_popular = global_memory.get_top_movies(10)

    # 组装记忆上下文传递给后续节点
    memory_context = {
        "short_term": short_term_memory.get_session(session_id),
        "long_term_recent": [dict(r) for r in recent_long],
        "global_popular": global_popular,
    }

    return {
        "query": query,
        "session_id": session_id,
        "memory_context": memory_context,
    }


# ── 并行调度器 ──
def recall_dispatcher(state: RecommenderState) -> list[Send]:
    """3 路并行召回。"""
    return [
        Send("semantic_recall", state),
        Send("keyword_recall", state),
        Send("llm_expand_recall", state),
    ]


# ── 路 1：语义检索 ──
def semantic_recall(state: RecommenderState) -> dict:
    q = state.get("query", "")
    results = semantic_retrieve.invoke({"query": q, "top_k": 8})
    for r in results:
        r.setdefault("source", "semantic")
    return {"candidates": results}


# ── 路 2：关键词匹配 ──
def keyword_recall(state: RecommenderState) -> dict:
    q = state.get("query", "")
    results = keyword_retrieve.invoke({"query": q, "top_k": 8})
    return {"candidates": results}


# ── 路 3：LLM 扩展 → 语义检索 ──
def llm_expand_recall(state: RecommenderState) -> dict:
    q = state.get("query", "")
    results = llm_expand_retrieve.invoke({"query": q, "top_k": 8})
    return {"candidates": results}


# ── Agent 4：LLM 推荐（含去重 + 缓存）──
def recommend_agent(state: RecommenderState) -> dict:
    """去重 + LLM 精排 + 生成推荐回答。相同查询走缓存。"""
    import hashlib

    from cache import cache as _cache

    query = state.get("query", "")
    raw = state.get("candidates", [])

    # 检查缓存
    cache_key = f"recommend:{hashlib.md5(query.encode()).hexdigest()}"
    cached = _cache.get_json(cache_key)
    if cached:
        return {"final_response": cached["answer"]}

    # 按 movie_id 去重，保留最高分
    best: dict[int, dict] = {}
    for c in raw:
        mid = c.get("movie_id", 0)
        if mid not in best or c.get("score", 0) > best[mid].get("score", 0):
            best[mid] = c
    candidates = sorted(best.values(), key=lambda x: x.get("score", 0), reverse=True)

    if candidates:
        lines = []
        for i, m in enumerate(candidates, 1):
            genres = ", ".join(m.get("genres", []))
            tags = ", ".join(m.get("tags", []))
            src = m.get("source", "?")
            title_en = m.get("title_en", m.get("title", "?"))
            lines.append(f"{i}. {title_en} | Genres: {genres} | Tags: {tags} | source={src}")
        movies_text = "\n".join(lines)
    else:
        movies_text = "(No candidates — use your knowledge.)"

    # 记忆上下文
    mem = state.get("memory_context", {})
    memory_note = ""
    if mem.get("long_term_recent"):
        memory_note = f"\n\n📌 此用户最近的交互记录（避免重复推荐）：{mem['long_term_recent']}"
    if mem.get("global_popular"):
        memory_note += f"\n\n🌍 当前全站热门电影 ID：{mem['global_popular'][:5]}"

    prompt = prompts.render("recommend.system", query=query, movies=movies_text)
    prompt += memory_note

    import time as _time
    _t0 = _time.time()
    response = model.invoke([{"role": "system", "content": prompt}])
    _elapsed = int((_time.time() - _t0) * 1000)

    # ── 三层记忆持久化 ──
    session_id = state.get("session_id", "default")

    # 短期记忆：记录本次推荐
    for m in candidates[:5]:
        short_term_memory.record_view(session_id, m.get("movie_id", 0))

    # 写入缓存（TTL 600s）
    _cache.set_json(cache_key, {"answer": response.content}, ttl=600)

    # 长期记忆 (SQLite)：持久化交互
    mid = candidates[0].get("movie_id", 0) if candidates else 0
    sqlite_manager.upsert_user_preferences(session_id)
    sqlite_manager.log_interaction(session_id, mid, "view", True)
    sqlite_manager.increment_interaction(session_id)

    # 全局记忆：写入后刷新
    global_memory.popularity

    return {"final_response": response.content}
