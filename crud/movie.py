"""电影 CRUD。"""

import json
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from models.movie import Movie


def get_movie(db: Session, movie_id: int) -> Optional[Movie]:
    return db.query(Movie).filter(Movie.id == movie_id).first()


def get_all_movies(db: Session) -> list[Movie]:
    return db.query(Movie).order_by(Movie.id).all()


def get_movies_by_ids(db: Session, movie_ids: list[int]) -> list[Movie]:
    return db.query(Movie).filter(Movie.id.in_(movie_ids)).all()


def get_hot_movies(db: Session, top_n: int = 10) -> list[Movie]:
    """热门电影（按 ID 排序，可扩展为按评分）。"""
    return db.query(Movie).order_by(Movie.id).limit(top_n).all()


def create_movie(db: Session, data: dict) -> Movie:
    m = Movie(
        id=data["movie_id"],
        title_en=data["title_en"],
        title_cn=data.get("title_cn", ""),
        year=data.get("year", 0),
        genres=json.dumps(data.get("genres", [])),
        tags=json.dumps(data.get("tags", [])),
        description=data.get("description", ""),
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def update_movie(db: Session, movie_id: int, data: dict) -> Optional[Movie]:
    m = get_movie(db, movie_id)
    if not m:
        return None
    if "title_en" in data and data["title_en"] is not None:
        m.title_en = data["title_en"]
    if "title_cn" in data and data["title_cn"] is not None:
        m.title_cn = data["title_cn"]
    if "year" in data and data["year"] is not None:
        m.year = data["year"]
    if "genres" in data and data["genres"] is not None:
        m.genres = json.dumps(data["genres"])
    if "tags" in data and data["tags"] is not None:
        m.tags = json.dumps(data["tags"])
    if "description" in data and data["description"] is not None:
        m.description = data["description"]
    db.commit()
    db.refresh(m)
    return m


def delete_movie(db: Session, movie_id: int) -> bool:
    m = get_movie(db, movie_id)
    if not m:
        return False
    db.delete(m)
    db.commit()
    return True


def seed_movies(db: Session, movies: dict[int, dict]) -> None:
    if db.query(Movie).count() > 0:
        return
    for mid, info in movies.items():
        db.add(Movie(
            id=mid,
            title_en=info["title_en"],
            title_cn=info.get("title_cn", ""),
            year=info.get("year", 0),
            genres=json.dumps(info["genres"]),
            tags=json.dumps(info["tags"]),
            description=info.get("description", ""),
        ))
    db.commit()
