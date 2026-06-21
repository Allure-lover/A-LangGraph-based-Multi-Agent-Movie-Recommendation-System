"""电影模型。"""

import json

from sqlalchemy import Column, Integer, String, Text

from database import Base


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, autoincrement=False)
    title_en = Column(String(200), nullable=False, comment="英文片名")
    title_cn = Column(String(200), default="", comment="中文片名")
    year = Column(Integer, default=0, comment="上映年份")
    genres = Column(Text, nullable=False, comment="JSON 类型列表")     # ["Action","Drama"]
    tags = Column(Text, nullable=False, comment="JSON 关键词列表")     # ["mafia","family"]
    description = Column(Text, default="", comment="英文剧情简介")

    @property
    def genre_list(self) -> list[str]:
        return json.loads(self.genres)

    @property
    def tag_list(self) -> list[str]:
        return json.loads(self.tags)

    def to_dict(self) -> dict:
        return {
            "movie_id": self.id,
            "title_en": self.title_en,
            "title_cn": self.title_cn,
            "year": self.year,
            "genres": self.genre_list,
            "tags": self.tag_list,
            "description": self.description,
        }

    def display_name(self) -> str:
        """中文名（英文名, 年份）"""
        if self.title_cn:
            return f"**{self.title_cn}**（{self.title_en}, {self.year}）"
        return f"**{self.title_en}** ({self.year})"

    @property
    def embed_text(self) -> str:
        """用于 FAISS 嵌入的文本。"""
        genres = ", ".join(self.genre_list)
        tags = ", ".join(self.tag_list)
        return f"{self.title_en}. {genres}. {tags}. {self.description}"
