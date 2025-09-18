#!/usr/bin/env python3
"""
Pydantic 資料模型（Schemas）定義

此模組定義了 API 的輸入輸出資料模型，使用 Pydantic 進行資料驗證和序列化。
這些模型確保 API 請求和響應的資料格式正確，並提供自動的資料驗證功能。

模型設計模式：
1. Base 模型：定義共通欄位，用於繼承
2. Create 模型：用於創建新記錄的輸入驗證
3. Update 模型：用於更新記錄的部分欄位驗證（所有欄位可選）
4. Response 模型：用於 API 響應，包含完整的資料庫記錄資訊
5. Composite 模型：包含關聯資料的複合模型，用於 JOIN 查詢結果

資料驗證特點：
- 自動類型檢查和轉換
- Email 格式驗證
- 選填欄位支援
- 自動生成 OpenAPI 文檔

作者: FastAPI Demo Project
創建日期: 2024
"""

from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime


# ===== 用戶相關模型 =====

class UserBase(BaseModel):
    """
    用戶基礎模型

    定義用戶的基本欄位，用於繼承到其他用戶相關模型。
    包含用戶的核心識別和個人資訊。

    Attributes:
        username: 用戶名稱，必填，用於登入和顯示
        email: 電子郵件地址，必填且自動驗證格式
        full_name: 完整姓名，選填
        age: 年齡，選填，必須為正整數
    """
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    age: Optional[int] = None


class UserCreate(UserBase):
    """
    用戶創建模型

    用於接收創建新用戶的 API 請求資料。
    直接繼承 UserBase 的所有欄位，不添加額外驗證。

    使用範例:
        POST /users/
        {
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "age": 30
        }
    """
    pass


class UserUpdate(BaseModel):
    """
    用戶更新模型

    用於接收更新用戶資訊的 API 請求資料。
    所有欄位都是選填的，支援部分更新。

    使用範例:
        PUT /users/{user_id}
        {
            "full_name": "John Smith",
            "age": 31
        }
    """
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    age: Optional[int] = None


class User(UserBase):
    """
    用戶響應模型

    用於 API 響應，包含完整的用戶資料庫記錄資訊。
    包含系統自動生成的欄位如 ID 和時間戳記。

    Attributes:
        id: 用戶唯一識別碼
        created_at: 帳戶創建時間
        updated_at: 最後更新時間，可能為空
    """
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        # 允許從 SQLAlchemy ORM 物件自動轉換
        from_attributes = True


# ===== 文章相關模型 =====

class PostBase(BaseModel):
    """
    文章基礎模型

    定義文章的基本內容欄位。

    Attributes:
        title: 文章標題，必填
        content: 文章內容，必填，支援長文本
    """
    title: str
    content: str


class PostCreate(PostBase):
    """
    文章創建模型

    用於接收創建新文章的 API 請求資料。
    除了基本內容外，還需要指定作者。

    Attributes:
        author_id: 文章作者的用戶 ID，必填

    使用範例:
        POST /posts/
        {
            "title": "My First Post",
            "content": "This is the content of my first post.",
            "author_id": 1
        }
    """
    author_id: int


class PostUpdate(BaseModel):
    """
    文章更新模型

    用於接收更新文章的 API 請求資料。
    支援部分更新，所有欄位都是選填的。

    Note:
        不允許更改文章作者，author_id 不包含在更新模型中
    """
    title: Optional[str] = None
    content: Optional[str] = None


class Post(PostBase):
    """
    文章響應模型

    用於 API 響應，包含完整的文章資料庫記錄資訊。

    Attributes:
        id: 文章唯一識別碼
        author_id: 作者用戶 ID
        created_at: 文章創建時間
        updated_at: 最後更新時間，可能為空
    """
    id: int
    author_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ===== 複合模型（包含關聯資料） =====

class UserWithPosts(User):
    """
    包含文章的用戶模型

    擴展用戶模型，包含該用戶發表的所有文章。
    用於提供用戶詳細資訊，包括其創作內容。

    Attributes:
        posts: 用戶發表的文章列表，預設為空列表

    使用場景:
        - 用戶個人資料頁面
        - 用戶文章管理介面
        - 統計分析用戶活躍度
    """
    posts: List[Post] = []


class PostWithAuthor(Post):
    """
    包含作者資訊的文章模型

    擴展文章模型，包含完整的作者資訊。
    用於顯示文章詳情時同時展示作者資料。

    Attributes:
        author: 文章作者的完整用戶資訊

    使用場景:
        - 文章詳情頁面
        - 文章列表（需要顯示作者）
        - 搜尋結果展示
    """
    author: User


# ===== 分類相關模型 =====

class CategoryBase(BaseModel):
    """
    分類基礎模型

    定義分類的基本資訊欄位。

    Attributes:
        name: 分類名稱，必填
        description: 分類描述，選填，用於詳細說明分類用途
    """
    name: str
    description: Optional[str] = None


class CategoryCreate(CategoryBase):
    """
    分類創建模型

    用於接收創建新分類的 API 請求資料。
    直接繼承 CategoryBase 的所有欄位。

    使用範例:
        POST /categories/
        {
            "name": "Technology",
            "description": "Articles about technology and programming"
        }
    """
    pass


class CategoryUpdate(BaseModel):
    """
    分類更新模型

    用於接收更新分類資訊的 API 請求資料。
    支援部分更新，所有欄位都是選填的。
    """
    name: Optional[str] = None
    description: Optional[str] = None


class Category(CategoryBase):
    """
    分類響應模型

    用於 API 響應，包含完整的分類資料庫記錄資訊。

    Attributes:
        id: 分類唯一識別碼
    """
    id: int

    class Config:
        from_attributes = True