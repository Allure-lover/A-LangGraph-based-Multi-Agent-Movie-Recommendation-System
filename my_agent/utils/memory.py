"""记忆系统模块。

实现分层记忆机制：

    短期记忆 (ShortTermMemory)
        - 维护当前会话的用户行为（浏览、点击）
        - 基于内存 dict，会话结束即清空
        - 支持快速回溯用户近期行为

    长期记忆 (LongTermMemory / SQLiteManager)
        - 持久化用户偏好画像
        - 存储交互历史（点击、喜欢、跳过）
        - 记录推荐反馈，用于优化后续推荐

    全局记忆 (GlobalMemory)
        - 记录全局热点统计（点击量、流行趋势）
        - 内存缓存，定期从 SQLite 刷新
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

# 数据库文件路径
# 优先级: 环境变量 DB_PATH > 默认值 (backend/recommender_memory.db)
_DEFAULT_DB = str(
    Path(__file__).resolve().parent.parent.parent / "recommender_memory.db"
)
_DB_PATH = os.getenv("DB_PATH", _DEFAULT_DB)


def _resolve_db_path(raw_path: str) -> str:
    """解析数据库路径，自动创建父目录，失败则回退到默认路径。"""
    path = Path(raw_path)
    # 确保父目录存在
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        path = Path(_DEFAULT_DB)
        path.parent.mkdir(parents=True, exist_ok=True)
    # 验证可写
    try:
        path.touch(exist_ok=True)
        return str(path)
    except (OSError, PermissionError):
        fallback = Path(_DEFAULT_DB)
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.touch(exist_ok=True)
        return str(fallback)


DB_PATH = _resolve_db_path(_DB_PATH)


# ==============================
# SQLite 长期记忆管理器
# ==============================
class SQLiteManager:
    """SQLite 持久化管理器，存储用户偏好与交互历史。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库表结构。"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    preferred_genres TEXT NOT NULL DEFAULT '[]',
                    genre_weights TEXT NOT NULL DEFAULT '{}',
                    interaction_count INTEGER DEFAULT 0,
                    last_active REAL DEFAULT 0.0,
                    created_at REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    movie_id INTEGER NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('view','click','like','dislike','skip')),
                    recommended BOOLEAN DEFAULT 1,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_preferences(user_id)
                );

                CREATE TABLE IF NOT EXISTS recommendation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    movie_ids TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    cold_start BOOLEAN DEFAULT 0,
                    timestamp REAL NOT NULL
                );
            """)

    # ── 用户偏好读写 ──

    def get_user_preferences(self, user_id: str) -> dict | None:
        """获取用户长期偏好。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "user_id": row["user_id"],
            "preferred_genres": json.loads(row["preferred_genres"]),
            "genre_weights": json.loads(row["genre_weights"]),
            "interaction_count": row["interaction_count"],
            "last_active": row["last_active"],
            "created_at": row["created_at"],
        }

    def upsert_user_preferences(
        self,
        user_id: str,
        preferred_genres: list[str] | None = None,
        genre_weights: dict | None = None,
    ) -> None:
        """创建或更新用户偏好。"""
        now = time.time()
        with self._get_conn() as conn:
            existing = conn.execute(
                "SELECT user_id FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            if existing:
                if preferred_genres is not None:
                    conn.execute(
                        "UPDATE user_preferences SET preferred_genres=?, last_active=? WHERE user_id=?",
                        (json.dumps(preferred_genres), now, user_id),
                    )
                if genre_weights is not None:
                    conn.execute(
                        "UPDATE user_preferences SET genre_weights=?, last_active=? WHERE user_id=?",
                        (json.dumps(genre_weights), now, user_id),
                    )
            else:
                conn.execute(
                    """INSERT INTO user_preferences
                       (user_id, preferred_genres, genre_weights, interaction_count, last_active, created_at)
                       VALUES (?, ?, ?, 0, ?, ?)""",
                    (
                        user_id,
                        json.dumps(preferred_genres or []),
                        json.dumps(genre_weights or {}),
                        now,
                        now,
                    ),
                )

    def increment_interaction(self, user_id: str) -> None:
        """增加用户交互计数。"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE user_preferences SET interaction_count = interaction_count + 1, last_active = ? WHERE user_id = ?",
                (time.time(), user_id),
            )

    # ── 交互历史 ──

    def log_interaction(
        self, user_id: str, movie_id: int, action: str, recommended: bool = True
    ) -> None:
        """记录单条用户交互。"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO interaction_history (user_id, movie_id, action, recommended, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, movie_id, action, int(recommended), time.time()),
            )

    def get_user_interactions(
        self, user_id: str, limit: int = 50
    ) -> list[dict]:
        """获取用户最近的交互历史。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM interaction_history
                   WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_interaction_stats(self, user_id: str) -> dict:
        """统计用户各操作类型的次数。"""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT
                     COUNT(*) as total,
                     SUM(CASE WHEN action='like' THEN 1 ELSE 0 END) as likes,
                     SUM(CASE WHEN action='dislike' THEN 1 ELSE 0 END) as dislikes,
                     SUM(CASE WHEN action='click' THEN 1 ELSE 0 END) as clicks,
                     SUM(CASE WHEN action='skip' THEN 1 ELSE 0 END) as skips
                   FROM interaction_history WHERE user_id = ?""",
                (user_id,),
            ).fetchone()
        return dict(row) if row else {}

    # ── 推荐日志 ──

    def log_recommendation(
        self, user_id: str, movie_ids: list[int], sources: list[str], cold_start: bool
    ) -> None:
        """记录一次推荐结果。"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO recommendation_log (user_id, movie_ids, sources, cold_start, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, json.dumps(movie_ids), json.dumps(sources), int(cold_start), time.time()),
            )

    def get_recent_recommendations(self, user_id: str, limit: int = 10) -> list[dict]:
        """获取用户最近的推荐记录。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM recommendation_log
                   WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── 全局统计 ──

    def get_global_popularity(self) -> dict[int, int]:
        """获取全局电影热度（按点击/喜欢次数）。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT movie_id, COUNT(*) as cnt
                   FROM interaction_history
                   WHERE action IN ('click', 'like', 'view')
                   GROUP BY movie_id ORDER BY cnt DESC"""
            ).fetchall()
        return {r["movie_id"]: r["cnt"] for r in rows}


# ==============================
# 短期记忆（会话级）
# ==============================
class ShortTermMemory:
    """
    短期会话记忆，存储当前请求周期内的用户行为。
    基于内存 dict，不持久化。
    """

    def __init__(self):
        self._store: dict[str, dict] = {}

    def init_session(self, session_id: str) -> None:
        """初始化或重置一个会话。"""
        self._store[session_id] = {
            "viewed_movie_ids": [],
            "clicked_movie_ids": [],
            "liked_movie_ids": [],
            "skipped_movie_ids": [],
            "context_tags": [],
            "request_count": 0,
        }

    def record_view(self, session_id: str, movie_id: int) -> None:
        """记录浏览。"""
        if session_id in self._store:
            self._store[session_id]["viewed_movie_ids"].append(movie_id)

    def record_click(self, session_id: str, movie_id: int) -> None:
        """记录点击。"""
        if session_id in self._store:
            self._store[session_id]["clicked_movie_ids"].append(movie_id)

    def record_feedback(self, session_id: str, movie_id: int, action: str) -> None:
        """记录喜欢/跳过。"""
        if session_id not in self._store:
            return
        if action == "like":
            self._store[session_id]["liked_movie_ids"].append(movie_id)
        elif action == "skip":
            self._store[session_id]["skipped_movie_ids"].append(movie_id)

    def get_session(self, session_id: str) -> dict:
        """获取当前会话快照。"""
        return self._store.get(session_id, {})

    def get_excluded_ids(self, session_id: str) -> set[int]:
        """获取当前会话中已交互的电影 ID 集合（用于去重）。"""
        s = self._store.get(session_id, {})
        ids = set()
        for key in ("viewed_movie_ids", "clicked_movie_ids", "skipped_movie_ids"):
            ids.update(s.get(key, []))
        return ids

    def increment_request(self, session_id: str) -> None:
        """会话请求计数 +1。"""
        if session_id in self._store:
            self._store[session_id]["request_count"] += 1


# ==============================
# 全局记忆
# ==============================
class GlobalMemory:
    """
    全局记忆：缓存热点统计，定期从 SQLite 刷新。
    """

    def __init__(self, sqlite_mgr: SQLiteManager, refresh_interval: float = 300.0):
        self._sqlite = sqlite_mgr
        self._refresh_interval = refresh_interval
        self._last_refresh = 0.0
        self._popularity: dict[int, int] = {}
        self._total_interactions: int = 0

    @property
    def popularity(self) -> dict[int, int]:
        """获取全局热度排行（自动刷新）。"""
        now = time.time()
        if now - self._last_refresh > self._refresh_interval:
            self._popularity = self._sqlite.get_global_popularity()
            self._total_interactions = sum(self._popularity.values())
            self._last_refresh = now
        return self._popularity

    @property
    def total_interactions(self) -> int:
        """全局总交互次数。"""
        self.popularity  # 触发刷新检查
        return self._total_interactions

    def get_top_movies(self, top_n: int = 10) -> list[int]:
        """获取全局最热电影 ID 列表。"""
        return list(self.popularity.keys())[:top_n]


# ==============================
# 全局单例
# ==============================
sqlite_manager = SQLiteManager()
short_term_memory = ShortTermMemory()
global_memory = GlobalMemory(sqlite_manager)
