#!/usr/bin/env python3
"""
FastAPI CRUD Demo - 主應用程式模組

這是一個基於 FastAPI 的 CRUD 示範應用程式，展示了如何使用 FastAPI、SQLAlchemy
和 PostgreSQL 構建一個功能完整的 REST API。

主要功能：
- 用戶管理 (Users)：創建、查詢、更新、刪除用戶
- 文章管理 (Posts)：文章的 CRUD 操作，支援與用戶的關聯
- 分類管理 (Categories)：分類的 CRUD 操作，支援與文章的關聯
- 自動 OpenAPI 文檔生成
- 支援資料庫表間的 JOIN 查詢操作

技術棧：
- FastAPI: 高性能 Web 框架，自動生成 API 文檔
- SQLAlchemy: Python SQL 工具包和 ORM
- PostgreSQL: 關係型資料庫
- Pydantic: 資料驗證和序列化
- Uvicorn: ASGI 伺服器

項目結構：
- main.py: 主應用程式入口點，負責應用程式初始化和路由註冊
- database.py: 資料庫連接配置和會話管理
- models.py: SQLAlchemy ORM 模型定義
- schemas.py: Pydantic 資料模型定義，用於 API 輸入輸出驗證
- crud.py: 資料庫操作的業務邏輯層
- routers/: API 路由模組，按功能分類組織

作者: FastAPI Demo Project
版本: 1.0.0
創建日期: 2024
"""

from fastapi import FastAPI
from database import engine, Base
from routers import users, posts, categories

# 創建 FastAPI 應用程式實例
# 配置應用程式元資料，用於自動生成的 OpenAPI 文檔
app = FastAPI(
    title="FastAPI CRUD Demo",
    description="A simple CRUD API with FastAPI, SQLAlchemy, and JOIN operations",
    version="1.0.0"
)

# 初始化資料庫表結構
# 在應用程式啟動時自動創建所有定義的資料庫表
Base.metadata.create_all(bind=engine)

# 註冊路由模組
# 將各個功能模組的路由加入到主應用程式中
app.include_router(users.router)      # 用戶相關的 API 端點
app.include_router(posts.router)      # 文章相關的 API 端點
app.include_router(categories.router) # 分類相關的 API 端點

@app.get("/", tags=["Root"])
def root():
    """
    根路徑端點

    返回歡迎訊息，確認 API 服務正常運行

    Returns:
        dict: 包含歡迎訊息的字典
    """
    return {"message": "Welcome to FastAPI CRUD Demo API"}

@app.get("/health", tags=["Health"])
def health_check():
    """
    健康檢查端點

    用於監控和負載均衡器檢查服務狀態

    Returns:
        dict: 包含服務健康狀態的字典
    """
    return {"status": "healthy"}

# 如果直接執行此檔案，啟動開發伺服器
if __name__ == "__main__":
    import uvicorn
    # 啟動 Uvicorn ASGI 伺服器
    # host="0.0.0.0" 允許外部連接，port=8000 設定監聽埠
    uvicorn.run(app, host="0.0.0.0", port=8000)