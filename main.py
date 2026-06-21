"""
RAG 推荐系统 CLI 测试。

用法:
    python main.py
    python main.py -q "推荐几部科幻片"
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from langchain_core.messages import HumanMessage

from my_agent import agent
from my_agent.utils.rag import build_faiss_index

EXAMPLES = [
    "推荐几部烧脑悬疑片，结局要有反转",
    "I want a mind-bending thriller with unexpected plot twists",
    "推荐适合全家一起看的动画电影，温情治愈",
    "想看类似教父那种黑帮片，有深度有内涵的",
    "推荐浪漫的爱情片，结局要感人",
]


def run_query(q: str):
    print(f"\n{'='*60}")
    print(f"  查询: {q}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = agent.invoke({"messages": [HumanMessage(content=q)]})
    elapsed = (time.time() - t0) * 1000
    print(f"  检索到 {len({c['movie_id'] for c in result.get('candidates', []) if 'movie_id' in c})} 部候选 · {elapsed:.0f}ms\n")
    print(result.get("final_response", ""))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Movie Recommender CLI")
    parser.add_argument("-q", "--query", type=str, help="自然语言查询")
    args = parser.parse_args()

    build_faiss_index()

    if args.query:
        run_query(args.query)
    else:
        for q in EXAMPLES:
            run_query(q)
