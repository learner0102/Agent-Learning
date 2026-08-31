# config.py
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

# 加载环境变量
load_dotenv()

# ============ 模型配置 ============
API_KEY = os.getenv("Qwen_API_KEY")
BASE_URL = os.getenv("DB_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.7-flash")

# ============ RAG配置 ============
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# ============ Agent配置 ============
MAX_AGENT_STEPS = 5  # 最大循环步数，防止死循环
TEMPERATURE = 0.7

# ============ 日志配置 ============
LOG_FILE = rf"study_line\eni\logs\agent_rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)