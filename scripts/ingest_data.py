"""
    将data中的文件向量化，存储到storage目录中
    整体流程：
        配置模型 → 建目录 → 读 CSV → LLM 生成表名/摘要 → 写入 SQLite → 建“表级”对象索引 → 建“行级”向量索引
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
from llama_index.core import ChatPromptTemplate, Settings, VectorStoreIndex, SQLDatabase
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.objects import SQLTableSchema, ObjectIndex
from llama_index.core.schema import TextNode
from llama_index.llms.dashscope import DashScope
from sqlalchemy import Column, String, Integer, Table, create_engine, MetaData, text

from app.core import config

from llama_index.embeddings.dashscope import DashScopeEmbedding

# ------------ 常量配置
DATA_DIR = Path("../data")
DB_FILE = "../storage/fangchan_table_questions.db"
TABLE_INFO_DIR = Path("../storage/table_info")
STORAGE_DIR = Path("../storage")
OBJ_INDEX_PATH = STORAGE_DIR / "obj_index"
ROW_INDEXES_PATH = STORAGE_DIR / "row_indexes"


class TableInfo(BaseModel):
    table_name: str = Field(..., description="表名（必须是下划线且无空格）")
    table_summary: str = Field(..., description="对表的简短、精确的总结/说明")


prompt_str = """\
为下表提供一个JSON格式的总结。
- 表名必须对表是唯一的，并且在简洁的同时能描述表的内容。
- 不要输出一个通用表名（例如：table,my_table）。
不要使用以下任何一个作为表名：{exclude_table_name_list}

Table:
{table_str}
Summary："""

prompt_tmpl = ChatPromptTemplate(
    message_templates=[ChatMessage.from_str(prompt_str, role="user")]
)


# 清洗列名：将列名中的“非单词字符（不是字母、数字、下划线）”，全部替换成“_”
def sanitize_column_name(col_name):
    return re.sub(r"\W+", "_", col_name)


def create_table_from_dataframe(df, table_name, engine, metadata_obj):
    """
    根据DataFrame在数据库中创建表，并写入数据
    Args:
     df: pandas.DataFrame，待入库的表格数据（通常从 CSV/Excel 读取）。
     table_name:  str，要创建的数据库表名。
     engine: sqlalchemy.Engine，数据库连接引擎，用于建表和写入。
     metadata_obj: sqlalchemy.MetaData，用于登记表结构（列名、类型等）。
    """
    sanitized_columns = {col: sanitize_column_name(col) for col in df.columns}
    df = df.rename(columns=sanitized_columns)
    columns = [
        Column(col, String if dtype == "object" else Integer) for col, dtype in zip(df.columns, df.dtypes)
    ]
    # *columns：把前面生成的 Column 列表拆开，作为单独参数传入。例如 columns = [Column("year", Integer), Column("city", String)]，等价于：Table(table_name, metadata_obj, Column("year", Integer), Column("city", String))
    table = Table(table_name, metadata_obj, *columns)
    # 真正在数据库中建表。根据 metadata_obj 里登记的所有表，通过 engine 去数据库执行 CREATE TABLE。只创建还不存在的表，已存在的不会覆盖、也不会改结构
    metadata_obj.create_all(engine)
    # 打开连接，逐行插入
    with engine.connect() as conn:  # 拿到一条数据库连接。with 结束时会自动关闭连接。
        for _, row in df.iterrows():  # 逐行遍历 DataFrame：_：行索引（用不到，所以用 _ 丢掉）；row：这一行的 Series，可用 row.to_dict() 转成字典
            insert_stmt = table.insert().values(
                **row.to_dict())  # table.insert()：生成 INSERT INTO 表名 ...；.values(**row.to_dict())：把字典拆成关键字参数，列名对上值
            conn.execute(insert_stmt)
        conn.commit()  # 循环结束后统一提交事务。没这句的话，SQLAlchemy 2.0 默认不会把插入持久化到数据库。


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
    # ... (加载 CSV 的代码不变) - 找出 data 目录下所有 CSV 文件，排好序后放进列表 csv_files
    # DATA_DIR.glob("*.csv")：用 pathlib 在 data 目录里按通配符找文件：*.csv：只匹配后缀是 .csv 的路径；返回的是迭代器，元素是 Path 对象，例如 ../data/houses.csv；默认不递归子目录，只扫 data 这一层。
    # sorted(...)：按路径名字母顺序排序，保证每次运行处理顺序一致，避免“这次先 a.csv、下次先 b.csv”。
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
        print(f"创建表：{table_info.table_name}")
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
        index_cls=VectorStoreIndex
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
                cursor = conn.execute(text(f"SELECT * FROM {table_name}"))
                result = cursor.fetchall()
                row_tups = [tuple(row) for row in result]

            nodes = [TextNode(text=str(t)) for t in row_tups]
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(str(table_index_path))
        else:
            print(f"表 {table_name} 的行索引已存在，跳过。")

    print("--- 数据摄取和索引构建全部完成！ ---")


if __name__ == '__main__':
    run_ingestion()
