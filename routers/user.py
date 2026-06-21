"""用户路由 — /api/users。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from crud import user as crud
from database import get_db
from schemas.user import UserListOut, UserOut

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=UserListOut)
def list_users(db: Session = Depends(get_db)):
    users = crud.get_all_users(db)
    return UserListOut(total=len(users), users=[u.to_dict() for u in users])


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: str, db: Session = Depends(get_db)):
    u = crud.get_user(db, user_id)
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    return u.to_dict()
