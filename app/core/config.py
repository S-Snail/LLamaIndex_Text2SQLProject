import os

from dotenv import load_dotenv

load_dotenv()

# API Keys
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# File Paths (更新为新结构)
DB_FILE = "./storage/fangchan_table_questions.db"

# LLM Config
LLM_MODEL = "qwen-max"
EMBED_MODEL = "text-embedding-v1"

# Workflow Config
WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", 120))
