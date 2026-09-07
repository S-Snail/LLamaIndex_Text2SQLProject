from llama_index.core.llms import ChatResponse
from llama_index.core.objects import SQLTableSchema
from llama_index.core.workflow import (
    Event,
    Workflow,
    step,
    Context,
    StartEvent,
    StopEvent,
)
from typing import List, Dict
import re
from pathlib import Path

from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.dashscope import DashScope
from sqlalchemy import create_engine
from app.core import config
from llama_index.core import (
    SQLDatabase,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings, PromptTemplate
)
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT


# 定义工作流事件---------------
class TableRetrieveEvent(Event):
    tab_context_str: str
    query: str


class TextToSQLEvent(Event):
    sql: str
    query: str


# --- 辅助函数 ---
def parse_response_to_sql(chat_response: ChatResponse) -> str:
    response = chat_response.message.content
    # 提取 "SQLQuery:" 之后的内容，并在 "SQLResult:" 之前截断（如果存在）
    sql_query_start = response.find("SQLQuery:")
    if sql_query_start != -1:
        response = response[sql_query_start:]
        if response.startswith("SQLQuery:"):
            response = response[len("SQLQuery:"):]
    sql_result_start = response.find("SQLResult:")
    if sql_result_start != -1:
        response = response[:sql_result_start]

    sql = response.strip().strip("```").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    return sql


# --- 核心工作流类 ---
class AdvancedTextToSQLWorkflow(Workflow):
    # ... (此工作流类完全不变)
    def __init__(
            self,
            llm,
            sql_database,
            obj_retriever,
            vector_index_dict,
            sql_retriever,
            text2sql_prompt,
            response_synthesis_prompt,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.sql_database = sql_database
        self.obj_retriever = obj_retriever
        self.vector_index_dict = vector_index_dict
        self.sql_retriever = sql_retriever
        self.text2sql_prompt = text2sql_prompt
        self.response_synthesis_prompt = response_synthesis_prompt

    def _get_table_context_and_rows_str(
            self, query_str: str, table_schema_objs: List[SQLTableSchema]
    ):
        context_strs = []
        for table_schema_obj in table_schema_objs:
            table_info = self.sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
            if table_schema_obj.context_str:
                table_info += f" 表的描述是: {table_schema_obj.context_str}"

            vector_retriever = self.vector_index_dict[
                table_schema_obj.table_name
            ].as_retriever(similarity_top_k=2)
            relevant_nodes = vector_retriever.retrieve(query_str)
            if len(relevant_nodes) > 0:
                table_row_context = (
                    "\n以下是一些相关的示例行 (值的顺序与上面的列相同):\n"
                )
                for node in relevant_nodes:
                    table_row_context += str(node.get_content()) + "\n"
                table_info += table_row_context
            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    @step
    def retrieve_tables(self, ctx: Context, ev: StartEvent) -> TableRetrieveEvent:
        # retrieve() 返回的是包含 TextNode 的 NodeWithScore 列表
        retrieved_nodes = self.obj_retriever.retrieve(ev.query)

        table_schema_objs = []
        # 正则表达式，用于从 TextNode 的文本中提取 table_name 和 context_str
        # 它会匹配 table_name='...' 和 context_str='...'
        pattern = re.compile(r"table_name='(.*?)'.*context_str='(.*?)'", re.DOTALL)

        for n in retrieved_nodes:
            text = n.get_content()
            match = pattern.search(text)

            if match:
                table_name = match.group(1)
                context_str = match.group(2)
                # 手动重新创建 SQLTableSchema 对象
                table_schema_objs.append(
                    SQLTableSchema(table_name=table_name, context_str=context_str)
                )

        if not table_schema_objs:
            # 如果没有匹配到任何表，可以返回一个空结果或记录日志
            print("Warning: Could not retrieve any valid table schemas for the query.")

        table_context_str = self._get_table_context_and_rows_str(
            ev.query, table_schema_objs
        )
        return TableRetrieveEvent(
            tab_context_str=table_context_str, query=ev.query
        )

    @step
    def generate_sql(self, ctx: Context, ev: TableRetrieveEvent) -> TextToSQLEvent:
        fmt_messages = self.text2sql_prompt.format_messages(
            query_str=ev.query, schema=ev.tab_context_str
        )
        chat_response = self.llm.chat(fmt_messages)
        sql = parse_response_to_sql(chat_response)
        print(f"生成SQL: {sql}")
        return TextToSQLEvent(sql=sql, query=ev.query)

    @step
    def execute_sql_and_synthesize(self, ctx: Context, ev: TextToSQLEvent) -> StopEvent:
        retrieved_rows = self.sql_retriever.retrieve(ev.sql)
        print(f"SQL查询结果: {retrieved_rows}")

        fmt_messages = self.response_synthesis_prompt.format_messages(
            sql_query=ev.sql,
            context_str=str(retrieved_rows),
            query_str=ev.query,
        )
        chat_response = self.llm.chat(fmt_messages)
        return StopEvent(
            result={"answer": str(chat_response.message.content), "sql_query": ev.sql}
        )


# --- 配置常量 ---
STORAGE_DIR = Path("./storage")
OBJ_INDEX_PATH = STORAGE_DIR / "obj_index"
ROW_INDEXES_PATH = STORAGE_DIR / "row_indexes"


# --- 服务类 ----
class TextToSQLService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TextToSQLService, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return
        print("Initializing TextToSQLService...")
        engine = create_engine(f"sqlite:///{config.DB_FILE}")
        sql_database = SQLDatabase(engine)
        llm = DashScope(model=config.LLM_MODEL, api_key=config.DASHSCOPE_API_KEY)
        embed_model = DashScopeEmbedding(
            model_name=config.EMBED_MODEL, api_key=config.DASHSCOPE_API_KEY
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        print("Loading Object Index from disk...")
        # 1. 创建 StorageContext 指向对象索引的保存位置
        storage_context = StorageContext.from_defaults(persist_dir=str(OBJ_INDEX_PATH))
        # 2. 使用 load_index_from_storage 加载完整的 ObjectIndex
        obj_index = load_index_from_storage(storage_context)
        obj_retriever = obj_index.as_retriever(similarity_top_k=3)
        print("Object Index loaded successfully.")
        vector_index_dict = self._load_row_indexes(sql_database)
        sql_retriever = SQLRetriever(sql_database)
        text2sql_prompt = DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
            dialect=engine.dialect.name
        )
        response_synthesis_prompt = PromptTemplate(
            # ... (prompt string不变)
            "给定一个输入问题和从SQL查询返回的结果，合成一个自然的语言回答。\n"
            "问题: {query_str}\n"
            "SQL 查询: {sql_query}\n"
            "SQL 结果: {context_str}\n"
            "回答: "
        )
        self.workflow = AdvancedTextToSQLWorkflow(
            llm=llm,
            sql_database=sql_database,
            obj_retriever=obj_retriever,
            vector_index_dict=vector_index_dict,
            sql_retriever=sql_retriever,
            text2sql_prompt=text2sql_prompt,
            response_synthesis_prompt=response_synthesis_prompt,
            timeout=120,
        )
        self.initialized = True
        print("TextToSQLService initialized successfully.")

    def _load_row_indexes(
            self, sql_database: SQLDatabase
    ) -> Dict[str, VectorStoreIndex]:
        vector_index_dict = {}
        for table_name in sql_database.get_usable_table_names():
            index_path = ROW_INDEXES_PATH / table_name
            if index_path.exists():
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(index_path)
                )
                vector_index_dict[table_name] = load_index_from_storage(
                    storage_context
                )
        return vector_index_dict

    async def run_query(self, question: str) -> dict:
        # 创建了工作流对象，初始化参数，并运行工作流
        result = await self.workflow.run(query=question)
        return result


text_to_sql_service = TextToSQLService()
