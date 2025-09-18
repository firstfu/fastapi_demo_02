from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from crud import PostCRUD, UserCRUD
import schemas

router = APIRouter(
    prefix="/posts",
    tags=["posts"],
)

@router.post("/", response_model=schemas.Post, status_code=status.HTTP_201_CREATED)
def create_post(post: schemas.PostCreate, db: Session = Depends(get_db)):
    db_user = UserCRUD.get_user(db, user_id=post.author_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="Author not found")

    return PostCRUD.create_post(db=db, post=post)

@router.get("/", response_model=List[schemas.Post])
def read_posts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    posts = PostCRUD.get_posts(db, skip=skip, limit=limit)
    return posts

@router.get("/with-authors", response_model=List[schemas.PostWithAuthor])
def read_posts_with_authors(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    posts = PostCRUD.get_posts_with_authors(db, skip=skip, limit=limit)
    return posts

@router.get("/{post_id}", response_model=schemas.Post)
def read_post(post_id: int, db: Session = Depends(get_db)):
    db_post = PostCRUD.get_post(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post

@router.get("/{post_id}/with-author", response_model=schemas.PostWithAuthor)
def read_post_with_author(post_id: int, db: Session = Depends(get_db)):
    db_post = PostCRUD.get_post_with_author(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post

@router.get("/user/{user_id}", response_model=List[schemas.Post])
def read_posts_by_user(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    db_user = UserCRUD.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    posts = PostCRUD.get_posts_by_user(db, user_id=user_id, skip=skip, limit=limit)
    return posts

@router.put("/{post_id}", response_model=schemas.Post)
def update_post(post_id: int, post_update: schemas.PostUpdate, db: Session = Depends(get_db)):
    db_post = PostCRUD.update_post(db, post_id=post_id, post_update=post_update)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post

@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: int, db: Session = Depends(get_db)):
    success = PostCRUD.delete_post(db, post_id=post_id)
    if not success:
        raise HTTPException(status_code=404, detail="Post not found")