from fastapi import APIRouter, HTTPException

from app.models.api_models import QueryRequest, QueryResponse
from app.services.test_to_sql_service import text_to_sql_service

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    接收自然语言，返回 SQL 查询和最终答案
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    print("正在查询中...")
    try:
        print(f"收到查询：${request.question}")
        result = await text_to_sql_service.run_query(request.question)
        print(f"生成SQL：{result['sql_query']}")
        print(f"生成回答：{result['answer']}")
        return QueryResponse(sql_query=result['sql_query'], answer=result['answer'])
    except Exception as e:
        print(f"查询处理时发生错误：{e}")
        raise HTTPException(status_code=500, detail=str(e))
