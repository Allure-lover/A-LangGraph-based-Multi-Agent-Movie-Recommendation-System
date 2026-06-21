"""用户模型（简化，仅 REST demo 用）。"""

from sqlalchemy import Column, String

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(20), primary_key=True)
    name = Column(String(50), nullable=False)

    def to_dict(self) -> dict:
        return {"user_id": self.id, "name": self.name}
