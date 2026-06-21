"""电影 Schema。"""

from pydantic import BaseModel, Field


class MovieCreate(BaseModel):
    movie_id: int
    title_en: str
    title_cn: str = ""
    year: int = 0
    genres: list[str] = []
    tags: list[str] = []
    description: str = ""


class MovieUpdate(BaseModel):
    title_en: str | None = None
    title_cn: str | None = None
    year: int | None = None
    genres: list[str] | None = None
    tags: list[str] | None = None
    description: str | None = None


class MovieOut(BaseModel):
    movie_id: int
    title_en: str
    title_cn: str
    year: int
    genres: list[str]
    tags: list[str]
    description: str

    model_config = {"from_attributes": True}


class MovieListOut(BaseModel):
    total: int
    movies: list[MovieOut]
