#!/usr/bin/env python3
"""
SQLAlchemy ORM 資料庫模型定義

此模組定義了應用程式的所有資料庫表結構，使用 SQLAlchemy ORM 進行物件關聯映射。
包含用戶、文章、分類等核心業務實體的資料模型定義。

模型關聯說明：
- User (用戶) ↔ Post (文章)：一對多關係，一個用戶可以發表多篇文章
- Category (分類) ↔ Post (文章)：多對多關係（未來擴展）

資料庫設計特點：
- 支援時間戳記 (created_at, updated_at) 自動管理
- 使用外鍵約束確保資料完整性
- 建立適當的索引以優化查詢性能
- 支援軟刪除和版本控制（可擴展）

作者: FastAPI Demo Project
創建日期: 2024
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class User(Base):
    """
    用戶模型

    儲存系統用戶的基本資訊，包括認證資料和個人資料。
    每個用戶可以發表多篇文章，建立一對多的關聯關係。

    資料表名稱: users

    欄位說明:
        id: 主鍵，自動遞增的用戶唯一識別碼
        username: 用戶名稱，必須唯一，用於登入和顯示
        email: 電子郵件地址，必須唯一，用於通知和認證
        full_name: 用戶完整姓名，選填欄位
        age: 用戶年齡，選填欄位，可用於統計分析
        created_at: 帳戶創建時間，自動設定為當前時間
        updated_at: 最後更新時間，每次修改時自動更新

    關聯關係:
        posts: 與 Post 模型的一對多關係，獲取用戶發表的所有文章
    """
    __tablename__ = "users"

    # 主鍵欄位，自動遞增
    id = Column(Integer, primary_key=True, index=True)

    # 用戶名稱，唯一且建立索引以加速查詢
    username = Column(String(50), unique=True, index=True, nullable=False)

    # 電子郵件，唯一且建立索引
    email = Column(String(100), unique=True, index=True, nullable=False)

    # 完整姓名，可為空
    full_name = Column(String(100), nullable=True)

    # 年齡資訊，可為空
    age = Column(Integer, nullable=True)

    # 時間戳記欄位，自動管理創建和更新時間
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 關聯關係：一對多，一個用戶可以有多篇文章
    posts = relationship("Post", back_populates="author")


class Post(Base):
    """
    文章模型

    儲存用戶發表的文章內容，包含標題、內容和作者資訊。
    每篇文章都必須關聯到一個存在的用戶作為作者。

    資料表名稱: posts

    欄位說明:
        id: 主鍵，自動遞增的文章唯一識別碼
        title: 文章標題，必填欄位，最多200字元
        content: 文章內容，必填欄位，使用 Text 類型支援長文本
        author_id: 外鍵，關聯到 users 表的 id 欄位
        created_at: 文章發表時間，自動設定為當前時間
        updated_at: 最後修改時間，每次編輯時自動更新

    關聯關係:
        author: 與 User 模型的多對一關係，獲取文章作者資訊
    """
    __tablename__ = "posts"

    # 主鍵欄位，自動遞增
    id = Column(Integer, primary_key=True, index=True)

    # 文章標題，必填且限制長度
    title = Column(String(200), nullable=False)

    # 文章內容，使用 Text 類型支援長文本
    content = Column(Text, nullable=False)

    # 外鍵：關聯到用戶表，確保每篇文章都有作者
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 時間戳記欄位，自動管理創建和更新時間
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 關聯關係：多對一，多篇文章可以屬於同一個用戶
    author = relationship("User", back_populates="posts")


class Category(Base):
    """
    分類模型

    用於組織和分類文章，提供文章的主題分類功能。
    未來可以擴展為與文章的多對多關聯關係。

    資料表名稱: categories

    欄位說明:
        id: 主鍵，自動遞增的分類唯一識別碼
        name: 分類名稱，可為空但建議填寫
        description: 分類描述，詳細說明此分類的用途和範圍

    未來擴展:
        - 可與 Post 模型建立多對多關係
        - 支援分類層級結構（父子分類）
        - 添加分類的排序和顯示設定
    """
    __tablename__ = "categories"

    # 主鍵欄位，自動遞增
    id = Column(Integer, primary_key=True, index=True)

    # 分類名稱，最多200字元
    name = Column(String(200), nullable=True)

    # 分類描述，使用 Text 類型支援詳細說明
    description = Column(Text, nullable=True)