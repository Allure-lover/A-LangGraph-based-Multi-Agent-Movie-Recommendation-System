"""用户 CRUD。"""

from typing import Optional

from sqlalchemy.orm import Session

from models.user import User


def get_user(db: Session, user_id: str) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_all_users(db: Session) -> list[User]:
    return db.query(User).all()


def seed_users(db: Session, users: dict[str, dict]) -> None:
    if db.query(User).count() > 0:
        return
    for uid, info in users.items():
        db.add(User(id=uid, name=info["name"]))
    db.commit()
