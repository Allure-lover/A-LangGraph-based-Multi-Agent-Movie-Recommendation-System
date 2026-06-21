"""多 Agent RAG 工具模块。"""

import os

import dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("DASHSCOPE_BASE_URL")

model = ChatOpenAI(model="qwen-turbo", temperature=0.3)


# ── 路 1：语义检索 ──

@tool
def semantic_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """FAISS 语义检索：直接用用户原始查询检索。"""
    from .rag import build_faiss_index, rag_retrieve

    build_faiss_index()
    return rag_retrieve(query, top_k=top_k)


# ── 路 2：关键词匹配 ──

# 中英文关键词映射（中文 query → 英文电影 genre/tag）
_KEYWORD_MAP = {
    "悬疑": ["Thriller", "Mystery", "twist", "detective", "psychopath", "dark"],
    "烧脑": ["twist", "mystery", "nonlinear", "reality", "dream", "identity", "layers"],
    "反转": ["twist", "mystery", "nonlinear", "betrayal"],
    "动画": ["Animation", "pixar"],
    "治愈": ["hope", "friendship", "love", "courage", "kindness", "journey"],
    "全家": ["Animation", "Adventure", "Comedy", "family", "friendship", "journey"],
    "黑帮": ["mafia", "gangster", "Crime", "crime"],
    "犯罪": ["Crime", "mafia", "hitman", "serial killer", "heist"],
    "科幻": ["Sci-Fi", "space", "robot", "dream", "simulation"],
    "爱情": ["Romance", "love", "tears", "tragedy"],
    "浪漫": ["Romance", "love", "paris", "jazz"],
    "感人": ["tears", "love", "courage", "humanity", "sacrifice", "friendship"],
    "动作": ["Action", "kungfu", "battle", "war", "fight"],
    "战争": ["war", "sacrifice", "battle", "freedom"],
    "恐怖": ["Thriller", "serial killer", "psychopath", "dark"],
    "喜剧": ["Comedy", "quirky", "pixar"],
    "音乐": ["Music", "Musical", "jazz", "drum"],
    "历史": ["History", "holocaust", "rome", "scotland"],
    "励志": ["hope", "freedom", "courage", "dreams", "running"],
    "武侠": ["kungfu", "battle", "Action"],
    "青春": ["love", "friendship", "dreams", "courage"],
    "温情": ["friendship", "love", "hope", "courage", "kindness", "family"],
    "温情": ["friendship", "love", "hope", "courage", "kindness", "father"],
    "思考": ["philosophy", "reality", "identity", "humanity", "twist"],
    "心理": ["psychopath", "identity", "twist", "dream", "reality"],
    "经典": ["mafia", "hope", "love", "freedom", "holocaust"],
    "悬疑片": ["Thriller", "Mystery", "twist", "detective"],
    "烧脑片": ["twist", "mystery", "nonlinear", "reality", "dream"],
    "动画片": ["Animation", "pixar"],
    "科幻片": ["Sci-Fi", "space", "robot"],
    "爱情片": ["Romance", "love"],
    "动作片": ["Action", "kungfu", "battle"],
}


@tool
def keyword_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """关键词匹配：中英文关键词映射 → 匹配电影标签和类型。"""
    from data import get_all_movies

    movies = get_all_movies()

    # 从 query 中提取命中词对应的英文关键词
    keywords: set[str] = set()
    for cn_word, en_words in _KEYWORD_MAP.items():
        if cn_word in query:
            keywords.update(en_words)

    # 也加入 query 中的英文词
    query_lower = query.lower()
    for w in query_lower.replace(",", " ").replace("。", " ").split():
        if w.isascii() and len(w) > 1:
            keywords.add(w)

    if not keywords:
        return []

    scored = []
    for m in movies:
        tags_text = " ".join(m.get("tags", [])).lower()
        genres_text = " ".join(m.get("genres", [])).lower()
        combined = f"{genres_text} {tags_text}"
        hits = sum(1 for kw in keywords if kw.lower() in combined)
        if hits > 0:
            score = hits / max(len(keywords), 1)
            scored.append((m, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        {**m, "score": round(s, 2), "source": "keyword"}
        for m, s in scored[:top_k]
    ]


# ── 路 3：LLM 查询扩展 → 语义检索 ──

@tool
def llm_expand_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """LLM 扩展查询 → FAISS 检索。用 LLM 将用户查询改写为更丰富的英文关键词后再检索。"""
    from langchain_core.messages import SystemMessage

    from .rag import build_faiss_index, rag_retrieve

    from . import prompts

    prompt = SystemMessage(content=prompts.get_llm_expand_prompt())
    response = model.invoke([prompt] + [{"role": "user", "content": query}])
    expanded = response.content.strip()

    build_faiss_index()
    results = rag_retrieve(expanded, top_k=top_k)
    for r in results:
        r["source"] = "llm_expand"
    return results
