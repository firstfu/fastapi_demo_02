#!/usr/bin/env python3
"""
用戶管理 API 路由模組

此模組定義了用戶相關的所有 API 端點，提供完整的用戶 CRUD 操作。
支援用戶註冊、查詢、更新、刪除，以及獲取用戶關聯文章等功能。

API 端點：
- POST /users/: 創建新用戶
- GET /users/: 獲取用戶列表（支援分頁）
- GET /users/{user_id}: 獲取特定用戶資訊
- GET /users/{user_id}/with-posts: 獲取用戶及其文章
- PUT /users/{user_id}: 更新用戶資訊
- DELETE /users/{user_id}: 刪除用戶

安全特性：
- 驗證 email 和 username 的唯一性
- 適當的 HTTP 狀態碼和錯誤處理
- 資料驗證和類型檢查

作者: FastAPI Demo Project
創建日期: 2024
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from crud import UserCRUD
import schemas

# 創建用戶路由器
# prefix="/users" 表示所有路由都以 /users 開頭
# tags=["users"] 用於 OpenAPI 文檔中的分組
router = APIRouter(
    prefix="/users",
    tags=["users"],
)


@router.post("/", response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    創建新用戶

    檢查 email 和 username 的唯一性後創建新用戶。
    如果 email 或 username 已存在，返回 400 錯誤。

    Args:
        user: 用戶創建資料，包含 username、email 等資訊
        db: 資料庫會話依賴項

    Returns:
        schemas.User: 新創建的用戶資訊（包含生成的 ID 和時間戳記）

    Raises:
        HTTPException 400: Email 已被註冊
        HTTPException 400: Username 已被使用

    Example:
        POST /users/
        {
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "age": 30
        }

    Response:
        Status: 201 Created
        {
            "id": 1,
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "age": 30,
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": null
        }
    """
    # 檢查 email 是否已存在
    db_user = UserCRUD.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # 檢查 username 是否已存在
    db_user = UserCRUD.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    # 創建新用戶
    return UserCRUD.create_user(db=db, user=user)


@router.get("/", response_model=List[schemas.User])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    獲取用戶列表

    支援分頁查詢，返回用戶基本資訊列表。

    Args:
        skip: 跳過的記錄數量，用於分頁（預設 0）
        limit: 返回的最大記錄數量（預設 100，最大 100）
        db: 資料庫會話依賴項

    Returns:
        List[schemas.User]: 用戶資訊列表

    Example:
        GET /users/?skip=0&limit=10

    Response:
        [
            {
                "id": 1,
                "username": "john_doe",
                "email": "john@example.com",
                ...
            },
            ...
        ]
    """
    users = UserCRUD.get_users(db, skip=skip, limit=limit)
    return users


@router.get("/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    """
    獲取特定用戶資訊

    根據用戶 ID 返回單個用戶的詳細資訊。

    Args:
        user_id: 用戶唯一識別碼
        db: 資料庫會話依賴項

    Returns:
        schemas.User: 用戶詳細資訊

    Raises:
        HTTPException 404: 用戶不存在

    Example:
        GET /users/1

    Response:
        {
            "id": 1,
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "age": 30,
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": null
        }
    """
    db_user = UserCRUD.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/{user_id}/with-posts", response_model=schemas.UserWithPosts)
def read_user_with_posts(user_id: int, db: Session = Depends(get_db)):
    """
    獲取用戶及其所有文章

    返回用戶資訊及其發表的所有文章，使用 JOIN 查詢避免 N+1 問題。

    Args:
        user_id: 用戶唯一識別碼
        db: 資料庫會話依賴項

    Returns:
        schemas.UserWithPosts: 包含文章列表的用戶資訊

    Raises:
        HTTPException 404: 用戶不存在

    Example:
        GET /users/1/with-posts

    Response:
        {
            "id": 1,
            "username": "john_doe",
            "email": "john@example.com",
            ...,
            "posts": [
                {
                    "id": 1,
                    "title": "My First Post",
                    "content": "This is my first blog post.",
                    "author_id": 1,
                    "created_at": "2024-01-15T11:00:00Z",
                    "updated_at": null
                },
                ...
            ]
        }
    """
    db_user = UserCRUD.get_user_with_posts(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.put("/{user_id}", response_model=schemas.User)
def update_user(user_id: int, user_update: schemas.UserUpdate, db: Session = Depends(get_db)):
    """
    更新用戶資訊

    支援部分更新，只更新提供的欄位。未提供的欄位保持原值不變。

    Args:
        user_id: 要更新的用戶 ID
        user_update: 包含更新資料的模型（所有欄位都是選填的）
        db: 資料庫會話依賴項

    Returns:
        schemas.User: 更新後的用戶資訊

    Raises:
        HTTPException 404: 用戶不存在

    Example:
        PUT /users/1
        {
            "full_name": "John Smith",
            "age": 31
        }

    Response:
        {
            "id": 1,
            "username": "john_doe",  // 未更新
            "email": "john@example.com",  // 未更新
            "full_name": "John Smith",  // 已更新
            "age": 31,  // 已更新
            "created_at": "2024-01-15T10:30:00Z",
            "updated_at": "2024-01-15T15:45:00Z"  // 自動更新
        }
    """
    db_user = UserCRUD.update_user(db, user_id=user_id, user_update=user_update)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    刪除用戶

    永久刪除指定的用戶記錄。成功刪除返回 204 No Content。

    Args:
        user_id: 要刪除的用戶 ID
        db: 資料庫會話依賴項

    Returns:
        None: 成功刪除時不返回內容

    Raises:
        HTTPException 404: 用戶不存在

    Example:
        DELETE /users/1

    Response:
        Status: 204 No Content
        (無響應內容)

    Note:
        刪除用戶時，相關的文章可能會因為外鍵約束而導致刪除失敗。
        在生產環境中應該考慮軟刪除或級聯刪除策略。
    """
    success = UserCRUD.delete_user(db, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")