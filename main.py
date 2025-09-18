from fastapi import FastAPI
from database import engine, Base
from routers import users, posts

app = FastAPI(
    title="FastAPI CRUD Demo",
    description="A simple CRUD API with FastAPI, SQLAlchemy, and JOIN operations",
    version="1.0.0"
)

Base.metadata.create_all(bind=engine)

app.include_router(users.router)
app.include_router(posts.router)

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI CRUD Demo API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)