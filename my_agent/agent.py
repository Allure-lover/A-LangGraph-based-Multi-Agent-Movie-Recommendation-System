"""LangGraph 多 Agent RAG 推荐系统。

流水线 (6 节点, 3 路并行):
    START → user_agent → recall_dispatcher
                              ├── semantic_recall
                              ├── keyword_recall
                              └── llm_expand_recall
                              (parallel, operator.add)
                                    ↓
                              recommend_agent (去重 + LLM 推荐)
                                    ↓
                                   END
"""

from langgraph.graph import END, START, StateGraph

from .utils.nodes import (
    keyword_recall,
    llm_expand_recall,
    recall_dispatcher,
    recommend_agent,
    semantic_recall,
    user_agent,
)
from .utils.state import RecommenderState

agent_builder = StateGraph(RecommenderState)

agent_builder.add_node("user_agent", user_agent)
agent_builder.add_node("recommend_agent", recommend_agent)
agent_builder.add_node("semantic_recall", semantic_recall)
agent_builder.add_node("keyword_recall", keyword_recall)
agent_builder.add_node("llm_expand_recall", llm_expand_recall)

agent_builder.add_edge(START, "user_agent")
agent_builder.add_conditional_edges("user_agent", recall_dispatcher)
agent_builder.add_edge("semantic_recall", "recommend_agent")
agent_builder.add_edge("keyword_recall", "recommend_agent")
agent_builder.add_edge("llm_expand_recall", "recommend_agent")
agent_builder.add_edge("recommend_agent", END)

agent = agent_builder.compile()
