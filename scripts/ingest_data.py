import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

from llama_index.core import (
    SQLDatabase,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.objects import (
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.dashscope import DashScope
from llama_index.core import Settings


from llama_index.core.llms import ChatMessage
from llama_index.core.schema import TextNode
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    text,
)
from dotenv import load_dotenv
from app.core import config

load_dotenv()

# --- 配置常量 ---
DATA_DIR = Path("../data")
DB_FILE = "../storage/fangchan_table_questions.db"
TABLE_INFO_DIR = Path("../storage/table_info")
STORAGE_DIR = Path("../storage")
OBJ_INDEX_PATH = STORAGE_DIR / "obj_index"
ROW_INDEXES_PATH = STORAGE_DIR / "row_indexes"

class TableInfo(BaseModel):
    table_name: str = Field(..., description="表名 (必须是下划线且无空格)")
    table_summary: str = Field(..., description="对表的简短、精确的总结/说明")

prompt_str = """\
为下表提供一个 JSON 格式的总结。
- 表名必须对表是唯一的，并且在简洁的同时能描述表的内容。
- 不要输出一个通用的表名 (例如 table, my_table)。
不要使用以下任何一个作为表名: {exclude_table_name_list}

Table:
{table_str}

Summary: """
prompt_tmpl = ChatPromptTemplate(
    message_templates=[ChatMessage.from_str(prompt_str, role="user")]
)
def sanitize_column_name(col_name):
    return re.sub(r"\W+", "_", col_name)
def create_table_from_dataframe(df, table_name, engine, metadata_obj):
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)
    columns = [
        Column(col, String if dtype == "object" else Integer)
        for col, dtype in zip(df.columns, df.dtypes)
    ]
    table = Table(table_name, metadata_obj, *columns)
    metadata_obj.create_all(engine)
    with engine.connect() as conn:
        for _, row in df.iterrows():
            insert_stmt = table.insert().values(**row.to_dict())
            conn.execute(insert_stmt)
        conn.commit()


def run_ingestion():
    print("--- 步骤 1: 设置全局模型 ---")
    llm = DashScope(model=config.LLM_MODEL, api_key=os.getenv("DASHSCOPE_API_KEY"))

    embed_model = DashScopeEmbedding(
        model_name=config.EMBED_MODEL, api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    print("--- 步骤 2: 创建存储目录 ---")
    os.makedirs(TABLE_INFO_DIR, exist_ok=True)
    os.makedirs(ROW_INDEXES_PATH, exist_ok=True)

    print("--- 步骤 3: 加载 CSV 文件 ---")
    # ... (加载 CSV 的代码不变)
    csv_files = sorted([f for f in DATA_DIR.glob("*.csv") if f.is_file()])
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"解析错误 {csv_file}: {str(e)}")

    print(f"成功加载 {len(dfs)} 个 dataframes。")

    print("--- 步骤 4: 使用 LLM 提取表信息 ---")
    table_infos = []
    table_names = set()
    for idx, df in enumerate(dfs):
        info_file = TABLE_INFO_DIR / f"{idx}.json"
        if info_file.exists():
            print(f"从缓存加载表信息: {info_file}")
            with open(info_file, "r") as f:
                data = json.load(f)
            table_info = TableInfo.model_validate(data)
        else:
            while True:
                df_str = df.head(10).to_csv()
                try:
                    table_info = llm.structured_predict(
                        TableInfo,
                        prompt_tmpl,
                        table_str=df_str,
                        exclude_table_name_list=str(list(table_names)),
                    )
                    if table_info.table_name not in table_names:
                        with open(info_file, "w") as f:
                            json.dump(table_info.model_dump(), f)
                        break
                    else:
                        print(f"表名 {table_info.table_name} 已存在，重试。")
                except Exception as e:
                    print(f"LLM 调用失败，重试: {e}")
                    import time
                    time.sleep(1)
        
        table_names.add(table_info.table_name)
        table_infos.append(table_info)


    print("--- 步骤 5: 创建 SQLite 数据库 ---")
    # ... (创建数据库的代码不变)
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    engine = create_engine(f"sqlite:///{DB_FILE}")
    metadata_obj = MetaData()
    for idx, df in enumerate(dfs):
        table_info = table_infos[idx]
        print(f"创建表: {table_info.table_name}")
        create_table_from_dataframe(df, table_info.table_name, engine, metadata_obj)
    
    sql_database = SQLDatabase(engine)

    print("--- 步骤 5: 创建并持久化对象索引 (Object Index) ---")
    table_schema_objs = [
        SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
        for t in table_infos
    ]
    
    # 直接从对象创建 ObjectIndex
    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        index_cls=VectorStoreIndex,
    )
    
    # 将完整的 ObjectIndex 持久化到指定目录
    obj_index.index.storage_context.persist(persist_dir=str(OBJ_INDEX_PATH))
    print(f"对象索引已保存到: {OBJ_INDEX_PATH}")

    print("--- 步骤 6: 为每个表的行创建向量索引 ---")
    # ... (创建行索引的代码不变)
    for table_name in sql_database.get_usable_table_names():
        print(f"正在索引表中的行: {table_name}")
        table_index_path = ROW_INDEXES_PATH / table_name
        if not table_index_path.exists():
            with engine.connect() as conn:
                cursor = conn.execute(text(f'SELECT * FROM "{table_name}"'))
                result = cursor.fetchall()
                row_tups = [tuple(row) for row in result]
            
            nodes = [TextNode(text=str(t)) for t in row_tups]
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(str(table_index_path))
        else:
            print(f"表 {table_name} 的行索引已存在，跳过。")
    
    print("--- 数据摄取和索引构建全部完成！ ---")

if __name__ == "__main__":
    run_ingestion()