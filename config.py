"""
Конфигурация эксперимента
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Пути
# Используем относительный путь или переменную окружения
JSON_DIR = os.getenv("JSON_DIR", "Jsons_tables")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "raw_responses")
ANALYSIS_DIR = os.getenv("ANALYSIS_DIR", "analysis_results")

# API настройки
BOTHUB_API_KEY = os.getenv("BOTHUB_API_KEY", "YOUR_BOTHUB_API_KEY")
BOTHUB_BASE_URL = "https://api.bothub.chat/v1"

# Telegram для логирования
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Модели для тестирования
SMALL_MODELS = [
    "gemma-2-9b-it",
    "qwen-2.5-7b-instruct",
    "llama-3.1-8b-instruct",
    "mistral-7b-instruct",
    "ministral-8b",
    "phi-4",
    "qwen3-8b",
    "glm-4.5-air",
]

MEDIUM_MODELS = [
    "gemma-2-27b-it",
    "qwen-2.5-72b-instruct",
    "llama-3.3-70b-instruct",
    "deepseek-chat",
    "mistral-small-3.2-24b-instruct",
    "qwen3-32b",
    "ministral-14b-2512",
]

# Выбор моделей для эксперимента
# Опции: "small", "medium", "all"
MODEL_SET = "all"

def get_models():
    """Возвращает список моделей в зависимости от конфигурации"""
    if MODEL_SET == "small":
        return SMALL_MODELS
    elif MODEL_SET == "medium":
        return MEDIUM_MODELS
    elif MODEL_SET == "all":
        return SMALL_MODELS + MEDIUM_MODELS
    else:
        raise ValueError(f"Unknown MODEL_SET: {MODEL_SET}")

# API настройки
MAX_RETRIES = 3
REQUEST_DELAY = 0.5  # секунды между запросами
MAX_TOKENS = 2000
TEMPERATURE = 0.0

# Настройки сохранения
CHECKPOINT_INTERVAL = 10  # сохранять каждые N запросов

# Настройки логирования
LOG_FILE = "experiment.log"
LOG_LEVEL = "INFO"