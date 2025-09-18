#!/usr/bin/env python3
"""
分類管理 API 路由模組

此模組定義了分類相關的所有 API 端點，提供完整的分類 CRUD 操作。
分類用於組織和分類文章內容，為未來的文章分類功能提供基礎。

API 端點：
- POST /categories/: 創建新分類
- GET /categories/: 獲取分類列表（支援分頁）
- GET /categories/{category_id}: 獲取特定分類
- PUT /categories/{category_id}: 更新分類資訊
- DELETE /categories/{category_id}: 刪除分類

設計特點：
- 簡潔的 CRUD 操作接口
- 支援分頁查詢
- 為未來擴展保留靈活性（如與文章的多對多關係）
- 適當的錯誤處理和狀態碼

作者: FastAPI Demo Project
創建日期: 2024
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
from crud import CategoryCRUD
import schemas

# 創建分類路由器
# prefix="/categories" 表示所有路由都以 /categories 開頭
# tags=["categories"] 用於 OpenAPI 文檔中的分組
router = APIRouter(
    prefix="/categories",
    tags=["categories"],
)


@router.post("/", response_model=schemas.Category, status_code=status.HTTP_201_CREATED)
def create_category(category: schemas.CategoryCreate, db: Session = Depends(get_db)):
    """
    創建新分類

    創建一個新的文章分類，用於組織和分類文章內容。

    Args:
        category: 分類創建資料，包含 name 和 description
        db: 資料庫會話依賴項

    Returns:
        schemas.Category: 新創建的分類資訊（包含生成的 ID）

    Example:
        POST /categories/
        {
            "name": "Technology",
            "description": "Articles about technology, programming, and software development"
        }

    Response:
        Status: 201 Created
        {
            "id": 1,
            "name": "Technology",
            "description": "Articles about technology, programming, and software development"
        }
    """
    return CategoryCRUD.create_category(db=db, category=category)


@router.get("/", response_model=List[schemas.Category])
def read_categories(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    獲取分類列表

    支援分頁查詢，返回所有分類的基本資訊列表。

    Args:
        skip: 跳過的記錄數量，用於分頁（預設 0）
        limit: 返回的最大記錄數量（預設 100，最大 100）
        db: 資料庫會話依賴項

    Returns:
        List[schemas.Category]: 分類資訊列表

    Example:
        GET /categories/?skip=0&limit=10

    Response:
        [
            {
                "id": 1,
                "name": "Technology",
                "description": "Articles about technology..."
            },
            {
                "id": 2,
                "name": "Science",
                "description": "Scientific articles and research..."
            },
            ...
        ]
    """
    categories = CategoryCRUD.get_categories(db, skip=skip, limit=limit)
    return categories


@router.get("/{category_id}", response_model=schemas.Category)
def read_category(category_id: int, db: Session = Depends(get_db)):
    """
    獲取特定分類

    根據分類 ID 返回單個分類的詳細資訊。

    Args:
        category_id: 分類唯一識別碼
        db: 資料庫會話依賴項

    Returns:
        schemas.Category: 分類詳細資訊

    Raises:
        HTTPException 404: 分類不存在

    Example:
        GET /categories/1

    Response:
        {
            "id": 1,
            "name": "Technology",
            "description": "Articles about technology, programming, and software development"
        }
    """
    db_category = CategoryCRUD.get_category(db, category_id=category_id)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    return db_category


@router.put("/{category_id}", response_model=schemas.Category)
def update_category(category_id: int, category_update: schemas.CategoryUpdate, db: Session = Depends(get_db)):
    """
    更新分類資訊

    支援部分更新，只更新提供的欄位（name 和/或 description）。
    未提供的欄位保持原值不變。

    Args:
        category_id: 要更新的分類 ID
        category_update: 包含更新資料的模型（所有欄位都是選填的）
        db: 資料庫會話依賴項

    Returns:
        schemas.Category: 更新後的分類資訊

    Raises:
        HTTPException 404: 分類不存在

    Example:
        PUT /categories/1
        {
            "description": "Updated description for technology category"
        }

    Response:
        {
            "id": 1,
            "name": "Technology",  // 未更新，保持原值
            "description": "Updated description for technology category"  // 已更新
        }
    """
    db_category = CategoryCRUD.update_category(db, category_id=category_id, category_update=category_update)
    if db_category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    return db_category


@router.delete("/{category_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_category(category_id: int, db: Session = Depends(get_db)):
    """
    刪除分類

    永久刪除指定的分類記錄。成功刪除返回 204 No Content。

    Args:
        category_id: 要刪除的分類 ID
        db: 資料庫會話依賴項

    Returns:
        None: 成功刪除時不返回內容

    Raises:
        HTTPException 404: 分類不存在

    Example:
        DELETE /categories/1

    Response:
        Status: 204 No Content
        (無響應內容)

    Note:
        在未來擴展分類與文章的關聯關係時，需要考慮級聯刪除或約束檢查，
        確保不會刪除仍有關聯文章的分類。
    """
    success = CategoryCRUD.delete_category(db, category_id=category_id)
    if not success:
        raise HTTPException(status_code=404, detail="Category not found")