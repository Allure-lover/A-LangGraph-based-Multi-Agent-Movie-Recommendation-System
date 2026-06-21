"""
LangGraph RAG 推荐系统 — Streamlit 管理后台。

启动:
    cd backend && streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

import streamlit as st
from langchain_core.messages import HumanMessage

from my_agent import agent
from my_agent.utils.rag import build_faiss_index

# ── 页面 ──
st.set_page_config(page_title="基于 LangGraph 的 RAG 电影推荐系统", page_icon="🎬", layout="centered")

# ── 初始化 ──
if "faiss_ready" not in st.session_state:
    with st.spinner("加载 FAISS 索引 & LLM 模型..."):
        build_faiss_index()
    st.session_state.faiss_ready = True

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = []


def _run_query(q: str) -> dict:
    t0 = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=q)], "session_id": st.session_state.session_id})
    elapsed = (time.time() - t0) * 1000
    return {
        "query": q,
        "answer": result.get("final_response", ""),
        "retrieved": len({c["movie_id"] for c in result.get("candidates", []) if "movie_id" in c}),
        "elapsed_ms": elapsed,
    }


# ── 侧边栏 ──
with st.sidebar:
    st.markdown("## 🎬 多 Agent 电影推荐")
    st.caption("自然语言描述 → 3 路并行检索 → LLM 智能推荐")
    st.divider()
    st.caption("💡 试试这些：")
    examples = [
        "推荐烧脑悬疑片，结局要有反转",
        "想看类似教父的黑帮片",
        "适合全家一起看的动画电影",
        "浪漫的爱情片，结局要感人",
        "类似星际穿越的科幻片，有深度的",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}"):
            st.session_state.pending_query = ex

    st.divider()
    st.caption("📊 系统信息")
    try:
        resp = __import__("requests").get("http://127.0.0.1:8000/api/server/info", timeout=2)
        if resp.status_code == 200:
            info = resp.json()
            st.markdown(f"向量器: `{info['vectorizer']}`")
            st.markdown(f"缓存: `{info['cache']}`")
            st.markdown(f"记忆库: {'✅' if info['memory_db_exists'] else '❌'}")
    except Exception:
        pass

    with st.expander("🕸️ Agent 图结构"):
        st.markdown("""
        ```
        START
          │
        user_agent (解析查询 + 记忆)
          │
        recall_dispatcher (Send API)
          ├── semantic_recall (FAISS)
          ├── keyword_recall (关键词)
          └── llm_expand_recall (LLM扩写)
          │ (operator.add 合并)
          ▼
        recommend_agent (去重 + LLM推荐)
          │
         END
        ```
        """)

    st.divider()
    if st.session_state.history and st.button("🗑 清空对话"):
        st.session_state.history = []
        st.rerun()

# ── 对话历史 ──
for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["query"])
    with st.chat_message("assistant"):
        st.caption(f"检索 {item['retrieved']} 部 · {item['elapsed_ms']:.0f}ms")
        st.markdown(item["answer"])

# ── 处理侧边栏快捷输入 ──
if st.session_state.get("pending_query"):
    query = st.session_state.pop("pending_query")
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("⏳ 3 路并行检索 + LLM 生成推荐中..."):
            item = _run_query(query)
        st.caption(f"检索 {item['retrieved']} 部 · {item['elapsed_ms']:.0f}ms")
        st.markdown(item["answer"])
    st.session_state.history.append(item)

# ── 底部输入框 ──
if prompt := st.chat_input("描述你想看的电影，例如：推荐几部烧脑悬疑片，结局要有反转"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("⏳ 3 路并行检索 + LLM 生成推荐中..."):
            item = _run_query(prompt)
        st.caption(f"检索 {item['retrieved']} 部 · {item['elapsed_ms']:.0f}ms")
        st.markdown(item["answer"])
    st.session_state.history.append(item)
