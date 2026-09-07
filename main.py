from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="高级文本到 SQL 服务",
    description="一个基于 LlamaIndex 工作流的 API，实现了查询时表和行检索功能。",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    # 预热并初始化服务
    # 由于 text_to_sql_service 是在导入时创建的，这里不需要额外操作
    # 但这是一个放置启动逻辑的好地方
    print("应用启动，Text-to-SQL 服务已初始化。")


@app.get("/", summary="健康检查")
def read_root():
    return {"status": "ok"}


app.include_router(api_router, prefix="/api", tags=["Text-to-SQL"])

"""
启动服务
uvicorn main:app --reload --host 0.0.0.0 --port 9000
"""
