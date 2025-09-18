#!/usr/bin/env python3
"""
文章管理 API 路由模組

此模組定義了文章相關的所有 API 端點，提供完整的文章 CRUD 操作。
支援文章創建、查詢、更新、刪除，以及獲取文章關聯作者資訊等功能。

API 端點：
- POST /posts/: 創建新文章
- GET /posts/: 獲取文章列表（支援分頁）
- GET /posts/with-authors: 獲取文章列表及作者資訊
- GET /posts/{post_id}: 獲取特定文章
- GET /posts/{post_id}/with-author: 獲取文章及作者資訊
- GET /posts/user/{user_id}: 獲取特定用戶的文章
- PUT /posts/{post_id}: 更新文章內容
- DELETE /posts/{post_id}: 刪除文章

關聯查詢特性：
- 支援 JOIN 查詢載入作者資訊，避免 N+1 查詢問題
- 提供靈活的資料檢索選項
- 適當的權限驗證和錯誤處理

作者: FastAPI Demo Project
創建日期: 2024
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from crud import PostCRUD, UserCRUD
import schemas

# 創建文章路由器
# prefix="/posts" 表示所有路由都以 /posts 開頭
# tags=["posts"] 用於 OpenAPI 文檔中的分組
router = APIRouter(
    prefix="/posts",
    tags=["posts"],
)


@router.post("/", response_model=schemas.Post, status_code=status.HTTP_201_CREATED)
def create_post(post: schemas.PostCreate, db: Session = Depends(get_db)):
    """
    創建新文章

    驗證作者存在後創建新文章。如果指定的作者不存在，返回 404 錯誤。

    Args:
        post: 文章創建資料，包含 title、content 和 author_id
        db: 資料庫會話依賴項

    Returns:
        schemas.Post: 新創建的文章資訊（包含生成的 ID 和時間戳記）

    Raises:
        HTTPException 404: 指定的作者不存在

    Example:
        POST /posts/
        {
            "title": "My First Blog Post",
            "content": "This is the content of my first blog post...",
            "author_id": 1
        }

    Response:
        Status: 201 Created
        {
            "id": 1,
            "title": "My First Blog Post",
            "content": "This is the content of my first blog post...",
            "author_id": 1,
            "created_at": "2024-01-15T11:00:00Z",
            "updated_at": null
        }
    """
    # 驗證作者是否存在
    db_user = UserCRUD.get_user(db, user_id=post.author_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="Author not found")

    # 創建新文章
    return PostCRUD.create_post(db=db, post=post)


@router.get("/", response_model=List[schemas.Post])
def read_posts(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    獲取文章列表

    支援分頁查詢，返回文章基本資訊列表（不包含作者詳細資訊）。

    Args:
        skip: 跳過的記錄數量，用於分頁（預設 0）
        limit: 返回的最大記錄數量（預設 100，最大 100）
        db: 資料庫會話依賴項

    Returns:
        List[schemas.Post]: 文章資訊列表

    Example:
        GET /posts/?skip=0&limit=10

    Response:
        [
            {
                "id": 1,
                "title": "My First Post",
                "content": "Content...",
                "author_id": 1,
                "created_at": "2024-01-15T11:00:00Z",
                "updated_at": null
            },
            ...
        ]
    """
    posts = PostCRUD.get_posts(db, skip=skip, limit=limit)
    return posts


@router.get("/with-authors", response_model=List[schemas.PostWithAuthor])
def read_posts_with_authors(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    獲取文章列表及作者資訊

    使用 JOIN 查詢同時載入文章和作者資料，適用於需要顯示作者資訊的場景。

    Args:
        skip: 跳過的記錄數量，用於分頁（預設 0）
        limit: 返回的最大記錄數量（預設 100，最大 100）
        db: 資料庫會話依賴項

    Returns:
        List[schemas.PostWithAuthor]: 包含作者資訊的文章列表

    Example:
        GET /posts/with-authors?skip=0&limit=5

    Response:
        [
            {
                "id": 1,
                "title": "My First Post",
                "content": "Content...",
                "author_id": 1,
                "created_at": "2024-01-15T11:00:00Z",
                "updated_at": null,
                "author": {
                    "id": 1,
                    "username": "john_doe",
                    "email": "john@example.com",
                    "full_name": "John Doe",
                    ...
                }
            },
            ...
        ]
    """
    posts = PostCRUD.get_posts_with_authors(db, skip=skip, limit=limit)
    return posts


@router.get("/{post_id}", response_model=schemas.Post)
def read_post(post_id: int, db: Session = Depends(get_db)):
    """
    獲取特定文章

    根據文章 ID 返回單篇文章的詳細資訊（不包含作者詳細資訊）。

    Args:
        post_id: 文章唯一識別碼
        db: 資料庫會話依賴項

    Returns:
        schemas.Post: 文章詳細資訊

    Raises:
        HTTPException 404: 文章不存在

    Example:
        GET /posts/1

    Response:
        {
            "id": 1,
            "title": "My First Post",
            "content": "This is the full content of the post...",
            "author_id": 1,
            "created_at": "2024-01-15T11:00:00Z",
            "updated_at": null
        }
    """
    db_post = PostCRUD.get_post(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post


@router.get("/{post_id}/with-author", response_model=schemas.PostWithAuthor)
def read_post_with_author(post_id: int, db: Session = Depends(get_db)):
    """
    獲取文章及其作者資訊

    返回文章詳細資訊及完整的作者資料，適用於文章詳情頁面。

    Args:
        post_id: 文章唯一識別碼
        db: 資料庫會話依賴項

    Returns:
        schemas.PostWithAuthor: 包含作者資訊的文章詳情

    Raises:
        HTTPException 404: 文章不存在

    Example:
        GET /posts/1/with-author

    Response:
        {
            "id": 1,
            "title": "My First Post",
            "content": "This is the full content...",
            "author_id": 1,
            "created_at": "2024-01-15T11:00:00Z",
            "updated_at": null,
            "author": {
                "id": 1,
                "username": "john_doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "age": 30,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": null
            }
        }
    """
    db_post = PostCRUD.get_post_with_author(db, post_id=post_id)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post


@router.get("/user/{user_id}", response_model=List[schemas.Post])
def read_posts_by_user(user_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    獲取特定用戶的文章列表

    返回指定用戶發表的所有文章，支援分頁查詢。

    Args:
        user_id: 用戶唯一識別碼
        skip: 跳過的記錄數量，用於分頁（預設 0）
        limit: 返回的最大記錄數量（預設 100，最大 100）
        db: 資料庫會話依賴項

    Returns:
        List[schemas.Post]: 該用戶的文章列表

    Raises:
        HTTPException 404: 用戶不存在

    Example:
        GET /posts/user/1?skip=0&limit=10

    Response:
        [
            {
                "id": 1,
                "title": "User's First Post",
                "content": "Content...",
                "author_id": 1,
                "created_at": "2024-01-15T11:00:00Z",
                "updated_at": null
            },
            {
                "id": 3,
                "title": "User's Second Post",
                "content": "Content...",
                "author_id": 1,
                "created_at": "2024-01-16T09:00:00Z",
                "updated_at": null
            }
        ]
    """
    # 驗證用戶是否存在
    db_user = UserCRUD.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # 獲取該用戶的文章
    posts = PostCRUD.get_posts_by_user(db, user_id=user_id, skip=skip, limit=limit)
    return posts


@router.put("/{post_id}", response_model=schemas.Post)
def update_post(post_id: int, post_update: schemas.PostUpdate, db: Session = Depends(get_db)):
    """
    更新文章內容

    支援部分更新，只更新提供的欄位（title 和/或 content）。
    不支援更改文章作者。

    Args:
        post_id: 要更新的文章 ID
        post_update: 包含更新資料的模型（所有欄位都是選填的）
        db: 資料庫會話依賴項

    Returns:
        schemas.Post: 更新後的文章資訊

    Raises:
        HTTPException 404: 文章不存在

    Example:
        PUT /posts/1
        {
            "title": "Updated Post Title",
            "content": "Updated content..."
        }

    Response:
        {
            "id": 1,
            "title": "Updated Post Title",  // 已更新
            "content": "Updated content...",  // 已更新
            "author_id": 1,  // 不可更改
            "created_at": "2024-01-15T11:00:00Z",
            "updated_at": "2024-01-15T16:30:00Z"  // 自動更新
        }
    """
    db_post = PostCRUD.update_post(db, post_id=post_id, post_update=post_update)
    if db_post is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return db_post


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_post(post_id: int, db: Session = Depends(get_db)):
    """
    刪除文章

    永久刪除指定的文章記錄。成功刪除返回 204 No Content。

    Args:
        post_id: 要刪除的文章 ID
        db: 資料庫會話依賴項

    Returns:
        None: 成功刪除時不返回內容

    Raises:
        HTTPException 404: 文章不存在

    Example:
        DELETE /posts/1

    Response:
        Status: 204 No Content
        (無響應內容)

    Note:
        刪除文章後無法恢復，建議在生產環境中實施軟刪除策略。
    """
    success = PostCRUD.delete_post(db, post_id=post_id)
    if not success:
        raise HTTPException(status_code=404, detail="Post not found")