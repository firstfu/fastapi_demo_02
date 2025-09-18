from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_
from typing import List, Optional
import models
import schemas

class UserCRUD:
    @staticmethod
    def get_user(db: Session, user_id: int) -> Optional[models.User]:
        return db.query(models.User).filter(models.User.id == user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
        return db.query(models.User).filter(models.User.email == email).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[models.User]:
        return db.query(models.User).filter(models.User.username == username).first()

    @staticmethod
    def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[models.User]:
        return db.query(models.User).offset(skip).limit(limit).all()

    @staticmethod
    def create_user(db: Session, user: schemas.UserCreate) -> models.User:
        db_user = models.User(**user.dict())
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate) -> Optional[models.User]:
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        if db_user:
            update_data = user_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_user, field, value)
            db.commit()
            db.refresh(db_user)
        return db_user

    @staticmethod
    def delete_user(db: Session, user_id: int) -> bool:
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        if db_user:
            db.delete(db_user)
            db.commit()
            return True
        return False

    @staticmethod
    def get_user_with_posts(db: Session, user_id: int) -> Optional[models.User]:
        return db.query(models.User).options(joinedload(models.User.posts)).filter(models.User.id == user_id).first()

class PostCRUD:
    @staticmethod
    def get_post(db: Session, post_id: int) -> Optional[models.Post]:
        return db.query(models.Post).filter(models.Post.id == post_id).first()

    @staticmethod
    def get_posts(db: Session, skip: int = 0, limit: int = 100) -> List[models.Post]:
        return db.query(models.Post).offset(skip).limit(limit).all()

    @staticmethod
    def get_posts_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Post]:
        return db.query(models.Post).filter(models.Post.author_id == user_id).offset(skip).limit(limit).all()

    @staticmethod
    def create_post(db: Session, post: schemas.PostCreate) -> models.Post:
        db_post = models.Post(**post.dict())
        db.add(db_post)
        db.commit()
        db.refresh(db_post)
        return db_post

    @staticmethod
    def update_post(db: Session, post_id: int, post_update: schemas.PostUpdate) -> Optional[models.Post]:
        db_post = db.query(models.Post).filter(models.Post.id == post_id).first()
        if db_post:
            update_data = post_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_post, field, value)
            db.commit()
            db.refresh(db_post)
        return db_post

    @staticmethod
    def delete_post(db: Session, post_id: int) -> bool:
        db_post = db.query(models.Post).filter(models.Post.id == post_id).first()
        if db_post:
            db.delete(db_post)
            db.commit()
            return True
        return False

    @staticmethod
    def get_post_with_author(db: Session, post_id: int) -> Optional[models.Post]:
        return db.query(models.Post).options(joinedload(models.Post.author)).filter(models.Post.id == post_id).first()

    @staticmethod
    def get_posts_with_authors(db: Session, skip: int = 0, limit: int = 100) -> List[models.Post]:
        return db.query(models.Post).options(joinedload(models.Post.author)).offset(skip).limit(limit).all()