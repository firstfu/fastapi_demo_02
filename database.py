#!/usr/bin/env python3
"""
資料庫配置模組

此模組負責配置和管理 SQLAlchemy 資料庫連接，提供資料庫會話管理功能。
支援多種資料庫後端，包括 SQLite（開發環境）和 PostgreSQL（生產環境）。

主要功能：
- 資料庫引擎初始化和配置
- 資料庫會話管理
- ORM 基礎類別定義
- 依賴注入式的資料庫會話提供

環境變數配置：
- DATABASE_URL: 資料庫連接字串
  - 開發環境預設：sqlite:///./app.db
  - 生產環境範例：postgresql://user:password@localhost:5432/dbname

使用方式：
- 在 API 路由中使用 get_db() 作為依賴項來獲取資料庫會話
- 繼承 Base 類別來定義 ORM 模型

作者: FastAPI Demo Project
創建日期: 2024
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# 載入環境變數配置檔案
# 從 .env 檔案中讀取環境變數，用於配置資料庫連接等參數
load_dotenv()

# 獲取資料庫連接 URL
# 優先使用環境變數 DATABASE_URL，如果未設定則使用 SQLite 作為預設資料庫
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# 創建資料庫引擎
# SQLAlchemy 引擎負責管理與資料庫的連接池和通信
# 針對 SQLite 添加特殊配置以支援多執行緒操作
engine = create_engine(
    DATABASE_URL,
    # SQLite 需要設定 check_same_thread=False 以支援 FastAPI 的異步操作
    # 其他資料庫（如 PostgreSQL）不需要此參數
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# 創建資料庫會話工廠
# SessionLocal 是一個會話類別，用於創建資料庫會話實例
# autocommit=False: 需要手動提交事務
# autoflush=False: 不會自動刷新未提交的更改到資料庫
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 創建 ORM 基礎類別
# 所有的資料庫模型都應該繼承此基礎類別
# SQLAlchemy 會自動處理表格創建和欄位映射
Base = declarative_base()

def get_db():
    """
    資料庫會話依賴項生成器

    此函數用作 FastAPI 的依賴項，為每個 API 請求提供一個獨立的資料庫會話。
    會話在請求處理完成後自動關閉，確保資源得到正確釋放。

    使用方式：
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            # 使用 db 會話進行資料庫操作
            return crud.get_items(db)

    Yields:
        Session: SQLAlchemy 資料庫會話物件

    Note:
        - 此函數使用生成器模式，確保在請求完成後正確關閉會話
        - 每個 API 請求都會獲得一個新的資料庫會話實例
        - 會話自動管理事務的開始和結束
    """
    # 創建新的資料庫會話
    db = SessionLocal()
    try:
        # 將會話提供給請求處理函數
        yield db
    finally:
        # 無論請求成功或失敗，都確保關閉會話
        db.close()