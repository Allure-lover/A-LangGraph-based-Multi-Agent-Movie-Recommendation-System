"""
LangGraph RAG 推荐系统 — FastAPI 服务。

启动:
    cd backend && python server.py

接口:
    POST /recommend    自然语言推荐
    GET  /health       健康检查
    GET  /stats        服务统计
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from cache import cache
from my_agent import agent
from my_agent.utils.rag import build_faiss_index

# ── 模型 ──


class RecommendRequest(BaseModel):
    query: str = Field(..., description="自然语言描述想要的电影")
    session_id: str = Field(default="", description="会话 ID（可选，用于记忆追踪）")


class RecommendResponse(BaseModel):
    query: str
    answer: str
    retrieved_count: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    cache_backend: str
    uptime_seconds: float


# ── 统计 ──
_stats = {"requests": 0}
_start_time = time.time()


# ── 生命周期 ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Server] Initializing...")
    build_faiss_index()
    try:
        from database import init_db
        init_db()
    except Exception:
        pass
    print("[Server] Ready.\n")
    yield
    print("\n[Server] Shutting down...")


app = FastAPI(
    title="LangGraph RAG Recommender",
    description="自然语言 → 向量检索 → LLM 推荐",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# crud 路由
try:
    from routers.movie import router as movie_router
    from routers.user import router as user_router

    app.include_router(movie_router)
    app.include_router(user_router)
except Exception:
    pass


# ── 路由 ──


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/agent/graph")
async def agent_graph():
    """返回 Agent 图结构信息。"""
    g = agent.get_graph()
    nodes = [n for n in g.nodes.keys() if not n.startswith("__")]
    edges = []
    for e in g.edges:
        src = e.source if e.source and not e.source.startswith("__") else "START"
        tgt = e.target if not e.target.startswith("__") else "END"
        edges.append({"from": src, "to": tgt, "conditional": e.conditional})
    return {
        "total_nodes": len(nodes),
        "nodes": nodes,
        "total_edges": len(edges),
        "edges": edges,
        "description": {
            "user_agent": "解析查询 + 加载记忆上下文",
            "recall_dispatcher": "Send API 3 路并行调度",
            "semantic_recall": "FAISS 语义检索 (SentenceTransformer)",
            "keyword_recall": "中英关键词匹配 (57 组映射)",
            "llm_expand_recall": "LLM 扩写查询 -> FAISS",
            "recommend_agent": "去重 + LLM 精排 + 持久化记忆",
        },
    }


@app.get("/api/server/info")
async def server_info():
    """返回服务运行信息。"""
    from my_agent.utils.rag import _vectorizer_mode
    from my_agent.utils.memory import DB_PATH as _mem_db
    import os as _os

    return {
        "version": "2.0",
        "vectorizer": _vectorizer_mode or "not initialized",
        "cache": cache.backend,
        "memory_db": _mem_db,
        "memory_db_exists": _os.path.exists(_mem_db) if _mem_db else False,
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        cache_backend=cache.backend,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/stats")
async def stats():
    return {**_stats, "uptime_seconds": round(time.time() - _start_time, 1)}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    _stats["requests"] += 1
    t0 = time.time()

    sid = req.session_id or str(uuid.uuid4())[:8]
    result = agent.invoke({"messages": [HumanMessage(content=req.query)], "session_id": sid})

    elapsed = (time.time() - t0) * 1000
    return RecommendResponse(
        query=req.query,
        answer=result.get("final_response", ""),
        retrieved_count=len({c["movie_id"] for c in result.get("candidates", []) if "movie_id" in c}),
        elapsed_ms=round(elapsed, 1),
    )


# ── 入口 ──

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"\n{'='*60}")
    print(f"  LangGraph RAG Recommender")
    print(f"  http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"{'='*60}\n")
    uvicorn.run("server:app", host=host, port=port, reload=True)
