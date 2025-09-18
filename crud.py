#!/usr/bin/env python3
"""
CRUD 操作模組 - 資料庫業務邏輯層

此模組實現了所有資料庫的 CRUD（Create, Read, Update, Delete）操作，
作為 API 路由層和資料庫模型層之間的業務邏輯層。提供了完整的資料操作功能，
包括基本的 CRUD 操作和進階的關聯查詢。

設計模式：
- 使用靜態方法組織 CRUD 操作，便於管理和測試
- 分離關注點：每個實體類型有獨立的 CRUD 類別
- 統一的錯誤處理和返回值模式
- 支援分頁查詢和關聯資料載入

主要功能：
1. UserCRUD: 用戶相關的資料庫操作
2. PostCRUD: 文章相關的資料庫操作
3. CategoryCRUD: 分類相關的資料庫操作

特殊功能：
- 支援 JOIN 查詢載入關聯資料
- 部分更新支援（只更新提供的欄位）
- 分頁查詢支援
- 安全的資料驗證和處理

作者: FastAPI Demo Project
創建日期: 2024
"""

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_
from typing import List, Optional
import models
import schemas


class UserCRUD:
    """
    用戶 CRUD 操作類別

    提供用戶相關的所有資料庫操作，包括基本的增刪改查功能，
    以及特殊的查詢功能如根據 email/username 查找用戶。

    主要功能：
    - 基本 CRUD 操作
    - 按不同條件查詢用戶（ID、email、username）
    - 載入用戶的關聯文章資料
    - 分頁查詢支援
    """

    @staticmethod
    def get_user(db: Session, user_id: int) -> Optional[models.User]:
        """
        根據用戶 ID 獲取單個用戶

        Args:
            db: 資料庫會話
            user_id: 用戶唯一識別碼

        Returns:
            User 模型實例，如果未找到則返回 None

        Example:
            user = UserCRUD.get_user(db, 1)
            if user:
                print(f"Found user: {user.username}")
        """
        return db.query(models.User).filter(models.User.id == user_id).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
        """
        根據電子郵件地址獲取用戶

        用於登入驗證和唯一性檢查。

        Args:
            db: 資料庫會話
            email: 電子郵件地址

        Returns:
            User 模型實例，如果未找到則返回 None
        """
        return db.query(models.User).filter(models.User.email == email).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[models.User]:
        """
        根據用戶名稱獲取用戶

        用於登入驗證和唯一性檢查。

        Args:
            db: 資料庫會話
            username: 用戶名稱

        Returns:
            User 模型實例，如果未找到則返回 None
        """
        return db.query(models.User).filter(models.User.username == username).first()

    @staticmethod
    def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[models.User]:
        """
        獲取用戶列表（支援分頁）

        Args:
            db: 資料庫會話
            skip: 跳過的記錄數量，用於分頁
            limit: 返回的最大記錄數量

        Returns:
            User 模型實例列表

        Example:
            # 獲取第2頁的用戶，每頁20個
            users = UserCRUD.get_users(db, skip=20, limit=20)
        """
        return db.query(models.User).offset(skip).limit(limit).all()

    @staticmethod
    def create_user(db: Session, user: schemas.UserCreate) -> models.User:
        """
        創建新用戶

        Args:
            db: 資料庫會話
            user: 用戶創建資料模型

        Returns:
            新創建的 User 模型實例

        Raises:
            IntegrityError: 如果 email 或 username 已存在

        Example:
            user_data = schemas.UserCreate(
                username="john_doe",
                email="john@example.com",
                full_name="John Doe"
            )
            new_user = UserCRUD.create_user(db, user_data)
        """
        db_user = models.User(**user.dict())
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate) -> Optional[models.User]:
        """
        更新用戶資訊

        支援部分更新，只更新提供的欄位。

        Args:
            db: 資料庫會話
            user_id: 要更新的用戶 ID
            user_update: 包含更新資料的模型

        Returns:
            更新後的 User 模型實例，如果用戶不存在則返回 None

        Example:
            update_data = schemas.UserUpdate(full_name="John Smith")
            updated_user = UserCRUD.update_user(db, 1, update_data)
        """
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        if db_user:
            # 只更新提供的欄位，忽略未設定的欄位
            update_data = user_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_user, field, value)
            db.commit()
            db.refresh(db_user)
        return db_user

    @staticmethod
    def delete_user(db: Session, user_id: int) -> bool:
        """
        刪除用戶

        Args:
            db: 資料庫會話
            user_id: 要刪除的用戶 ID

        Returns:
            bool: 刪除成功返回 True，用戶不存在返回 False

        Note:
            刪除用戶前應該考慮處理相關的文章資料
        """
        db_user = db.query(models.User).filter(models.User.id == user_id).first()
        if db_user:
            db.delete(db_user)
            db.commit()
            return True
        return False

    @staticmethod
    def get_user_with_posts(db: Session, user_id: int) -> Optional[models.User]:
        """
        獲取用戶及其所有文章

        使用 JOIN 查詢同時載入用戶資料和關聯的文章資料，
        避免 N+1 查詢問題。

        Args:
            db: 資料庫會話
            user_id: 用戶 ID

        Returns:
            User 模型實例（包含 posts 關聯資料），如果未找到則返回 None

        Example:
            user = UserCRUD.get_user_with_posts(db, 1)
            if user:
                print(f"User {user.username} has {len(user.posts)} posts")
        """
        return db.query(models.User).options(joinedload(models.User.posts)).filter(models.User.id == user_id).first()

class PostCRUD:
    """
    文章 CRUD 操作類別

    提供文章相關的所有資料庫操作，包括基本的增刪改查功能，
    以及按作者查詢、載入作者資訊等進階功能。

    主要功能：
    - 基本 CRUD 操作
    - 按作者查詢文章
    - 載入文章的關聯作者資料
    - 分頁查詢支援
    """

    @staticmethod
    def get_post(db: Session, post_id: int) -> Optional[models.Post]:
        """
        根據文章 ID 獲取單篇文章

        Args:
            db: 資料庫會話
            post_id: 文章唯一識別碼

        Returns:
            Post 模型實例，如果未找到則返回 None
        """
        return db.query(models.Post).filter(models.Post.id == post_id).first()

    @staticmethod
    def get_posts(db: Session, skip: int = 0, limit: int = 100) -> List[models.Post]:
        """
        獲取文章列表（支援分頁）

        Args:
            db: 資料庫會話
            skip: 跳過的記錄數量
            limit: 返回的最大記錄數量

        Returns:
            Post 模型實例列表
        """
        return db.query(models.Post).offset(skip).limit(limit).all()

    @staticmethod
    def get_posts_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Post]:
        """
        獲取特定用戶的文章列表

        Args:
            db: 資料庫會話
            user_id: 作者用戶 ID
            skip: 跳過的記錄數量
            limit: 返回的最大記錄數量

        Returns:
            該用戶的 Post 模型實例列表

        Example:
            # 獲取用戶ID為1的所有文章
            posts = PostCRUD.get_posts_by_user(db, 1)
        """
        return db.query(models.Post).filter(models.Post.author_id == user_id).offset(skip).limit(limit).all()

    @staticmethod
    def create_post(db: Session, post: schemas.PostCreate) -> models.Post:
        """
        創建新文章

        Args:
            db: 資料庫會話
            post: 文章創建資料模型

        Returns:
            新創建的 Post 模型實例

        Raises:
            ForeignKeyError: 如果指定的 author_id 不存在

        Example:
            post_data = schemas.PostCreate(
                title="My First Post",
                content="This is my first blog post.",
                author_id=1
            )
            new_post = PostCRUD.create_post(db, post_data)
        """
        db_post = models.Post(**post.dict())
        db.add(db_post)
        db.commit()
        db.refresh(db_post)
        return db_post

    @staticmethod
    def update_post(db: Session, post_id: int, post_update: schemas.PostUpdate) -> Optional[models.Post]:
        """
        更新文章內容

        支援部分更新，只更新提供的欄位。

        Args:
            db: 資料庫會話
            post_id: 要更新的文章 ID
            post_update: 包含更新資料的模型

        Returns:
            更新後的 Post 模型實例，如果文章不存在則返回 None

        Note:
            不支援更改文章作者（author_id）
        """
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
        """
        刪除文章

        Args:
            db: 資料庫會話
            post_id: 要刪除的文章 ID

        Returns:
            bool: 刪除成功返回 True，文章不存在返回 False
        """
        db_post = db.query(models.Post).filter(models.Post.id == post_id).first()
        if db_post:
            db.delete(db_post)
            db.commit()
            return True
        return False

    @staticmethod
    def get_post_with_author(db: Session, post_id: int) -> Optional[models.Post]:
        """
        獲取文章及其作者資訊

        使用 JOIN 查詢同時載入文章和作者資料。

        Args:
            db: 資料庫會話
            post_id: 文章 ID

        Returns:
            Post 模型實例（包含 author 關聯資料），如果未找到則返回 None

        Example:
            post = PostCRUD.get_post_with_author(db, 1)
            if post:
                print(f"Post '{post.title}' by {post.author.username}")
        """
        return db.query(models.Post).options(joinedload(models.Post.author)).filter(models.Post.id == post_id).first()

    @staticmethod
    def get_posts_with_authors(db: Session, skip: int = 0, limit: int = 100) -> List[models.Post]:
        """
        獲取文章列表及其作者資訊

        使用 JOIN 查詢同時載入所有文章和對應的作者資料，
        適用於需要顯示作者資訊的文章列表頁面。

        Args:
            db: 資料庫會話
            skip: 跳過的記錄數量
            limit: 返回的最大記錄數量

        Returns:
            Post 模型實例列表（每個都包含 author 關聯資料）

        Example:
            posts = PostCRUD.get_posts_with_authors(db, skip=0, limit=10)
            for post in posts:
                print(f"{post.title} by {post.author.username}")
        """
        return db.query(models.Post).options(joinedload(models.Post.author)).offset(skip).limit(limit).all()

class CategoryCRUD:
    """
    分類 CRUD 操作類別

    提供分類相關的所有資料庫操作，包括基本的增刪改查功能。
    分類用於組織和分類文章內容，未來可擴展為多對多關聯。

    主要功能：
    - 基本 CRUD 操作
    - 分頁查詢支援
    - 為未來擴展保留接口
    """

    @staticmethod
    def get_category(db: Session, category_id: int) -> Optional[models.Category]:
        """
        根據分類 ID 獲取單個分類

        Args:
            db: 資料庫會話
            category_id: 分類唯一識別碼

        Returns:
            Category 模型實例，如果未找到則返回 None
        """
        return db.query(models.Category).filter(models.Category.id == category_id).first()

    @staticmethod
    def get_categories(db: Session, skip: int = 0, limit: int = 100) -> List[models.Category]:
        """
        獲取分類列表（支援分頁）

        Args:
            db: 資料庫會話
            skip: 跳過的記錄數量
            limit: 返回的最大記錄數量

        Returns:
            Category 模型實例列表

        Example:
            # 獲取所有分類
            categories = CategoryCRUD.get_categories(db)
        """
        return db.query(models.Category).offset(skip).limit(limit).all()

    @staticmethod
    def create_category(db: Session, category: schemas.CategoryCreate) -> models.Category:
        """
        創建新分類

        Args:
            db: 資料庫會話
            category: 分類創建資料模型

        Returns:
            新創建的 Category 模型實例

        Example:
            category_data = schemas.CategoryCreate(
                name="Technology",
                description="Tech-related articles"
            )
            new_category = CategoryCRUD.create_category(db, category_data)
        """
        db_category = models.Category(**category.dict())
        db.add(db_category)
        db.commit()
        db.refresh(db_category)
        return db_category

    @staticmethod
    def update_category(db: Session, category_id: int, category_update: schemas.CategoryUpdate) -> Optional[models.Category]:
        """
        更新分類資訊

        支援部分更新，只更新提供的欄位。

        Args:
            db: 資料庫會話
            category_id: 要更新的分類 ID
            category_update: 包含更新資料的模型

        Returns:
            更新後的 Category 模型實例，如果分類不存在則返回 None

        Example:
            update_data = schemas.CategoryUpdate(
                description="Updated description"
            )
            updated_category = CategoryCRUD.update_category(db, 1, update_data)
        """
        db_category = db.query(models.Category).filter(models.Category.id == category_id).first()
        if db_category:
            update_data = category_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_category, field, value)
            db.commit()
            db.refresh(db_category)
        return db_category

    @staticmethod
    def delete_category(db: Session, category_id: int) -> bool:
        """
        刪除分類

        Args:
            db: 資料庫會話
            category_id: 要刪除的分類 ID

        Returns:
            bool: 刪除成功返回 True，分類不存在返回 False

        Note:
            刪除分類前應該考慮處理相關的文章關聯（如果存在）
        """
        db_category = db.query(models.Category).filter(models.Category.id == category_id).first()
        if db_category:
            db.delete(db_category)
            db.commit()
            return True
        return False