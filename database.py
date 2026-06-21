"""SQLAlchemy 引擎 & 会话管理。

环境变量:
    DB_TYPE     — sqlite | mysql (默认 sqlite)
    DB_HOST     — MySQL 主机 (默认 localhost)
    DB_PORT     — MySQL 端口 (默认 3306)
    DB_USER     — MySQL 用户 (默认 root)
    DB_PASSWORD — MySQL 密码
    DB_NAME     — 数据库名 (默认 recommender)
    SQLITE_PATH — SQLite 文件路径 (默认 recommender.db)
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DB_TYPE = os.getenv("DB_TYPE", "sqlite")

if DB_TYPE == "sqlite":
    SQLITE_PATH = os.getenv("SQLITE_PATH", "recommender.db")
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"
    ENGINE_KWARGS = {"connect_args": {"check_same_thread": False}}
else:
    _host = os.getenv("DB_HOST", "localhost")
    _port = os.getenv("DB_PORT", "3306")
    _user = os.getenv("DB_USER", "root")
    _pass = os.getenv("DB_PASSWORD", "")
    _name = os.getenv("DB_NAME", "recommender")
    DATABASE_URL = f"mysql+mysqlconnector://{_user}:{_pass}@{_host}:{_port}/{_name}"
    ENGINE_KWARGS = {"pool_size": 5, "pool_recycle": 3600}

engine = create_engine(DATABASE_URL, echo=False, **ENGINE_KWARGS)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """创建所有表（导入 models 后调用）。"""
    from models import movie, user  # noqa: F401
    Base.metadata.create_all(bind=engine)
    print(f"[DB] Tables created (type={DB_TYPE}).")


def get_db():
    """FastAPI 依赖注入：获取数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
