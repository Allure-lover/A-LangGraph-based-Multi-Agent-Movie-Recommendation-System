"""
数据库初始化 & 灌入种子数据。

用法:
    cd backend && python seed_db.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from database import SessionLocal, init_db
from crud.movie import seed_movies
from crud.user import seed_users
from data import MOVIES, USERS

if __name__ == "__main__":
    print("Creating tables...")
    init_db()

    db = SessionLocal()
    try:
        print("Seeding movies...")
        seed_movies(db, MOVIES)
        print("Seeding users...")
        seed_users(db, USERS)
        print(f"Done: {len(MOVIES)} movies, {len(USERS)} users.")
    finally:
        db.close()
