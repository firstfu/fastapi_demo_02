# FastAPI CRUD Demo

一個使用 FastAPI、SQLAlchemy 和 Alembic 構建的簡單 CRUD API 示例項目，展示了用戶和文章管理系統的基本操作。

## 功能特色

- **用戶管理**: 創建、讀取、更新、刪除用戶
- **文章管理**: 創建、讀取、更新、刪除文章
- **關聯查詢**: 支持用戶與文章的關聯查詢
- **數據驗證**: 使用 Pydantic 進行請求和響應驗證
- **數據庫遷移**: 使用 Alembic 管理數據庫版本
- **自動 API 文檔**: FastAPI 自動生成的 Swagger UI

## 技術棧

- **FastAPI**: 現代高性能的 Python Web 框架
- **SQLAlchemy**: Python SQL 工具包和 ORM
- **Alembic**: SQLAlchemy 的數據庫遷移工具
- **Pydantic**: 數據驗證和設置管理
- **SQLite**: 默認數據庫（可配置其他數據庫）
- **Uvicorn**: ASGI 服務器

## 項目結構

```
fastapi_demo_02/
├── main.py              # 應用程序入口點
├── models.py            # SQLAlchemy 數據模型
├── schemas.py           # Pydantic 模式定義
├── crud.py              # 數據庫操作邏輯
├── database.py          # 數據庫配置
├── requirements.txt     # 依賴列表
├── pyproject.toml      # 項目配置
├── routers/            # API 路由
│   ├── __init__.py
│   ├── users.py        # 用戶相關路由
│   └── posts.py        # 文章相關路由
└── alembic/            # 數據庫遷移文件
    └── env.py
```

## 安裝與設置

### 1. 環境要求

- Python 3.8+
- pip 或 uv

### 2. 安裝依賴

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 uv (推薦)
uv sync
```

### 3. 環境變量設置 (可選)

創建 `.env` 文件來配置數據庫連接：

```bash
# .env
DATABASE_URL=sqlite:///./app.db
# 或使用 PostgreSQL: postgresql://user:password@localhost/dbname
```

### 4. 數據庫初始化

應用程序會自動創建數據庫表，無需手動遷移。

## 運行應用程序

### 開發模式

```bash
# 方法 1: 直接運行
python main.py

# 方法 2: 使用 uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

應用程序將在 `http://localhost:8000` 啟動。

## API 文檔

啟動應用程序後，可以通過以下地址訪問 API 文檔：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API 端點

### 基礎端點

- `GET /` - 歡迎消息
- `GET /health` - 健康檢查

### 用戶管理 (`/users`)

| 方法 | 端點 | 描述 |
|------|------|------|
| POST | `/users/` | 創建新用戶 |
| GET | `/users/` | 獲取用戶列表 |
| GET | `/users/{user_id}` | 獲取特定用戶 |
| GET | `/users/{user_id}/with-posts` | 獲取用戶及其文章 |
| PUT | `/users/{user_id}` | 更新用戶信息 |
| DELETE | `/users/{user_id}` | 刪除用戶 |

### 文章管理 (`/posts`)

| 方法 | 端點 | 描述 |
|------|------|------|
| POST | `/posts/` | 創建新文章 |
| GET | `/posts/` | 獲取文章列表 |
| GET | `/posts/with-authors` | 獲取文章及作者信息 |
| GET | `/posts/{post_id}` | 獲取特定文章 |
| GET | `/posts/{post_id}/with-author` | 獲取文章及作者信息 |
| GET | `/posts/user/{user_id}` | 獲取特定用戶的文章 |
| PUT | `/posts/{post_id}` | 更新文章 |
| DELETE | `/posts/{post_id}` | 刪除文章 |

## 使用示例

### 創建用戶

```bash
curl -X POST "http://localhost:8000/users/" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "johndoe",
       "email": "john@example.com",
       "full_name": "John Doe"
     }'
```

### 創建文章

```bash
curl -X POST "http://localhost:8000/posts/" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "我的第一篇文章",
       "content": "這是文章內容...",
       "author_id": 1
     }'
```

### 獲取用戶及其文章

```bash
curl "http://localhost:8000/users/1/with-posts"
```

### 獲取文章及作者信息

```bash
curl "http://localhost:8000/posts/with-authors"
```

## 數據模型

### 用戶 (User)

```json
{
  "id": 1,
  "username": "johndoe",
  "email": "john@example.com",
  "full_name": "John Doe",
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

### 文章 (Post)

```json
{
  "id": 1,
  "title": "文章標題",
  "content": "文章內容...",
  "author_id": 1,
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T10:00:00Z"
}
```

## 開發與測試

### 運行測試

```bash
# 使用 pytest (如果已安裝)
pytest

# 或使用 uv
uv run pytest
```

### 開發建議

1. **代碼風格**: 遵循 PEP 8 標準
2. **錯誤處理**: 所有端點都包含適當的錯誤處理
3. **數據驗證**: 使用 Pydantic 模式進行輸入驗證
4. **關聯查詢**: 利用 SQLAlchemy 的 `joinedload` 進行高效查詢

## 擴展功能

這個項目可以進一步擴展：

- 添加用戶認證和授權
- 實現分頁和搜索功能
- 添加文章分類和標籤
- 集成緩存系統
- 添加單元測試和集成測試
- 部署到雲端平台

## 故障排除

### 常見問題

1. **端口被佔用**: 更改端口號或停止佔用端口的進程
2. **數據庫連接失敗**: 檢查 `DATABASE_URL` 環境變量
3. **依賴安裝失敗**: 確保使用正確的 Python 版本

### 日誌查看

應用程序運行時會在控制台輸出日誌信息，包括請求詳情和錯誤信息。

## 許可證

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！