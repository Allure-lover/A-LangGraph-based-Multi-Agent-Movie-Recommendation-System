"""用户 Schema。"""

from pydantic import BaseModel


class UserOut(BaseModel):
    user_id: str
    name: str

    model_config = {"from_attributes": True}


class UserListOut(BaseModel):
    total: int
    users: list[UserOut]
