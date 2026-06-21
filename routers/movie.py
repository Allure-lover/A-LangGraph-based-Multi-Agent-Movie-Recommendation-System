"""电影路由 — /api/movies。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from crud import movie as crud
from database import get_db
from schemas.movie import MovieCreate, MovieListOut, MovieOut, MovieUpdate

router = APIRouter(prefix="/api/movies", tags=["movies"])


@router.get("", response_model=MovieListOut)
def list_movies(db: Session = Depends(get_db)):
    movies = crud.get_all_movies(db)
    return MovieListOut(total=len(movies), movies=[m.to_dict() for m in movies])


@router.get("/hot", response_model=MovieListOut)
def hot_movies(top_n: int = 10, db: Session = Depends(get_db)):
    movies = crud.get_hot_movies(db, top_n)
    return MovieListOut(total=len(movies), movies=[m.to_dict() for m in movies])


@router.get("/{movie_id}", response_model=MovieOut)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    m = crud.get_movie(db, movie_id)
    if not m:
        raise HTTPException(status_code=404, detail="Movie not found")
    return m.to_dict()


@router.post("", response_model=MovieOut, status_code=201)
def add_movie(body: MovieCreate, db: Session = Depends(get_db)):
    if crud.get_movie(db, body.movie_id):
        raise HTTPException(status_code=409, detail="Movie already exists")
    m = crud.create_movie(db, body.model_dump())
    return m.to_dict()


@router.put("/{movie_id}", response_model=MovieOut)
def edit_movie(movie_id: int, body: MovieUpdate, db: Session = Depends(get_db)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    m = crud.update_movie(db, movie_id, updates)
    if not m:
        raise HTTPException(status_code=404, detail="Movie not found")
    return m.to_dict()


@router.delete("/{movie_id}", status_code=204)
def remove_movie(movie_id: int, db: Session = Depends(get_db)):
    if not crud.delete_movie(db, movie_id):
        raise HTTPException(status_code=404, detail="Movie not found")
