"""多 Agent RAG 推荐状态。

candidates 用 operator.add 归并，3 路并行召回自动合并。
"""

import operator
from typing import Annotated

from langgraph.graph import MessagesState


class RecommenderState(MessagesState):
    query: str
    session_id: str

    candidates: Annotated[list[dict], operator.add]
    """3 路并行召回自动合并"""

    memory_context: dict
    """记忆上下文: {short_term, long_term, global_popular}"""

    final_response: str
    """LLM 生成的推荐回答"""
