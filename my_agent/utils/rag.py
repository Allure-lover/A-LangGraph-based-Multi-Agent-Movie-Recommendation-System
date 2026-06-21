"""RAG 检索增强生成模块。

提供：
    - FAISS 向量索引构建 / 检索（在线模式：SentenceTransformer 嵌入）
    - TF-IDF 检索（离线 Demo 模式：scikit-learn，无需网络）
    - 关键词匹配兜底（纯本地，零依赖）

模式自动降级: SentenceTransformer → TF-IDF → 关键词匹配
"""

import os
import pickle
from pathlib import Path

import numpy as np

from data import get_all_movies

# 索引持久化路径（可通过环境变量 FAISS_INDEX_DIR 覆盖）
_FAISS_DIR = os.getenv(
    "FAISS_INDEX_DIR",
    str(Path(__file__).resolve().parent.parent.parent / ".faiss_index"),
)
INDEX_DIR = Path(_FAISS_DIR)
INDEX_PATH = INDEX_DIR / "movie_index.faiss"
META_PATH = INDEX_DIR / "movie_meta.pkl"

# 全局状态
_embedding_model = None       # SentenceTransformer 或 TfidfVectorizer
_faiss_index = None           # FAISS 索引
_movie_meta: list[dict] = []  # 电影元数据
_vectorizer_mode: str = ""    # "sbert" | "tfidf" | "keyword"


def _build_movie_text(movie: dict) -> str:
    """将电影元数据拼接为 FAISS 嵌入文本（用英文）。"""
    title = movie.get("title_en", movie.get("title", ""))
    genres = ", ".join(movie.get("genres", []))
    tags = ", ".join(movie.get("tags", []))
    desc = movie.get("description", "")
    return f"{title}. Genres: {genres}. Keywords: {tags}. Plot: {desc}"


# ── 模式 1: SentenceTransformer (在线) ──

def _try_load_sbert():
    """尝试加载 SentenceTransformer 模型。失败返回 None。"""
    global _embedding_model, _vectorizer_mode
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _vectorizer_mode = "sbert"
        _embedding_model = model
        print("[RAG] 模式: SentenceTransformer (在线)")
        return model
    except Exception as e:
        print(f"[RAG] SentenceTransformer 不可用 ({e})，尝试 TF-IDF 降级...")
        return None


# ── 模式 2: TF-IDF (离线 Demo) ──

def _try_load_tfidf():
    """尝试加载 scikit-learn TF-IDF。失败返回 None。"""
    global _embedding_model, _vectorizer_mode
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        movies = get_all_movies()
        texts = [_build_movie_text(m) for m in movies]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        vectorizer.fit(texts)
        _vectorizer_mode = "tfidf"
        _embedding_model = vectorizer
        print("[RAG] 模式: TF-IDF (离线 Demo)")
        return vectorizer
    except Exception as e:
        print(f"[RAG] TF-IDF 不可用 ({e})，使用关键词匹配兜底...")
        return None


# ── 模式 3: 关键词匹配 (纯本地兜底) ──

def _keyword_match(query: str, top_k: int, exclude: set[int]) -> list[dict]:
    """纯关键词匹配：按 query 中出现的词在电影文本中出现的次数打分。"""
    movies = get_all_movies()
    query_words = set(query.lower().replace(",", " ").replace(".", " ").split())

    scored = []
    for m in movies:
        if m["movie_id"] in exclude:
            continue
        text = _build_movie_text(m).lower()
        # 计算 query 词在文本中的命中率
        hits = sum(1 for w in query_words if w in text)
        genres_hit = sum(1 for w in query_words if w in [g.lower() for g in m["genres"]])
        tags_hit = sum(1 for w in query_words if w in [t.lower() for t in m["tags"]])
        score = hits + genres_hit * 2 + tags_hit * 2  # 类型/标签命中加权
        if score > 0:
            scored.append((m, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for m, s in scored[:top_k]:
        results.append({
            **m,
            "score": round(min(s / max(len(query_words), 1), 1.0), 2),
            "source": "rag",
            "reason": f"关键词匹配 (score={s})",
        })
    return results


# ── 统一接口 ──

def build_faiss_index(force_rebuild: bool = False) -> None:
    """
    构建向量索引（自动选择可用模式）。

    优先级: SentenceTransformer > TF-IDF > 关键词匹配

    Args:
        force_rebuild: 为 True 时强制重建。
    """
    global _faiss_index, _movie_meta, _embedding_model, _vectorizer_mode

    if not force_rebuild and _embedding_model is not None:
        return  # 已有模型，无需重建

    if not force_rebuild and INDEX_PATH.exists() and META_PATH.exists():
        try:
            import faiss
            _faiss_index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH, "rb") as f:
                _movie_meta = pickle.load(f)
            # 需要对应的 SBERT 模型
            if _try_load_sbert():
                print(f"[RAG] 从磁盘加载 FAISS 索引: {len(_movie_meta)} 部电影")
                return
        except Exception:
            pass  # 磁盘索引损坏或无 FAISS，重建

    # ── 重建索引 ──
    movies = get_all_movies()
    _movie_meta = movies

    # 尝试 SBERT → FAISS
    sbert_model = _try_load_sbert()
    if sbert_model is not None:
        try:
            import faiss
            texts = [_build_movie_text(m) for m in movies]
            embeddings = sbert_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(embeddings)
            dim = embeddings.shape[1]
            _faiss_index = faiss.IndexFlatIP(dim)
            _faiss_index.add(embeddings)
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(_faiss_index, str(INDEX_PATH))
            with open(META_PATH, "wb") as f:
                pickle.dump(_movie_meta, f)
            print(f"[RAG] FAISS 索引已构建: {len(movies)} 部电影, 维度={dim}")
            return
        except Exception as e:
            print(f"[RAG] FAISS 构建失败 ({e})，降级到 TF-IDF...")

    # 尝试 TF-IDF
    if _try_load_tfidf() is not None:
        return

    # 兜底：关键词匹配
    _vectorizer_mode = "keyword"
    print(f"[RAG] 模式: 关键词匹配 (纯本地兜底)")


def rag_retrieve(
    query: str,
    top_k: int = 10,
    exclude_movie_ids: list[int] | None = None,
) -> list[dict]:
    """
    RAG 检索：根据查询文本检索最相关电影。

    自动选择: SBERT+FAISS > TF-IDF+余弦 > 关键词匹配
    """
    global _faiss_index, _movie_meta, _embedding_model, _vectorizer_mode

    exclude = set(exclude_movie_ids or [])

    # 确保模型已加载
    if _embedding_model is None:
        build_faiss_index()

    # ── SBERT + FAISS ──
    if _vectorizer_mode == "sbert" and _faiss_index is not None:
        try:
            model = _embedding_model
            query_vec = model.encode([query], convert_to_numpy=True)
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            fetch_k = min(top_k + len(exclude), len(_movie_meta))
            scores, indices = _faiss_index.search(query_vec, fetch_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(_movie_meta):
                    continue
                movie = _movie_meta[idx]
                if movie["movie_id"] in exclude:
                    continue
                results.append({
                    **movie,
                    "score": round(float(score), 4),
                    "source": "rag",
                    "reason": f"语义相似 (score={score:.4f})",
                })
                if len(results) >= top_k:
                    break
            return results
        except Exception as e:
            print(f"[RAG] FAISS 检索失败 ({e})，降级到 TF-IDF...")
            _vectorizer_mode = "tfidf"

    # ── TF-IDF ──
    if _vectorizer_mode == "tfidf" and _embedding_model is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            vectorizer = _embedding_model
            movies = get_all_movies()
            texts = [_build_movie_text(m) for m in movies]
            corpus_vecs = vectorizer.transform(texts)
            query_vec = vectorizer.transform([query])
            sims = cosine_similarity(query_vec, corpus_vecs)[0]
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            results = []
            for idx, score in ranked:
                m = movies[idx]
                if m["movie_id"] in exclude:
                    continue
                results.append({
                    **m,
                    "score": round(float(score), 4),
                    "source": "rag",
                    "reason": f"TF-IDF 相似 (score={score:.4f})",
                })
                if len(results) >= top_k:
                    break
            return results
        except Exception as e:
            print(f"[RAG] TF-IDF 检索失败 ({e})，降级到关键词匹配...")

    # ── 关键词匹配兜底 ──
    return _keyword_match(query, top_k, exclude)


def get_rag_query_from_profile(user_profile: dict, cold_start: bool) -> str:
    """
    根据用户画像构造 RAG 查询文本。

    冷启动: 默认偏好描述
    普通用户: 拼接偏好类型 + 高评分电影标签
    """
    if cold_start:
        genres = user_profile.get("preferred_genres", ["Drama", "Action"])
        return f"Movies about {', '.join(genres)}. Popular and highly rated."

    preferred = user_profile.get("preferred_genres", [])
    history = user_profile.get("rating_history", [])

    top_tags: list[str] = []
    for r in history[:5]:
        if r.get("rating", 0) >= 4.0:
            from data import MOVIES
            mid = r.get("movie_id")
            if mid and mid in MOVIES:
                top_tags.extend(MOVIES[mid].get("tags", []))

    top_tags = list(dict.fromkeys(top_tags))[:8]

    if preferred and top_tags:
        return f"User prefers {', '.join(preferred)}. Tags: {', '.join(top_tags)}."
    elif preferred:
        return f"{', '.join(preferred)} movies. Highly rated."
    else:
        return "Popular movies across all genres. Highly rated."
