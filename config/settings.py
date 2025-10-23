"""Centralized application configuration constants."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
LOTTIE_DIR = ASSETS_DIR / "lottie"
CONTENT_DIR = ASSETS_DIR / "content"

DEFAULT_CSV_NAME = "sample_sales_data.csv"
DEFAULT_CSV_URL = "https://raw.githubusercontent.com/samsontands/portfolio3/main/sample_sales_data.csv"
CSV_CACHE_TTL = 3600

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"

PERSONAL_INFO_PATH = CONTENT_DIR / "personal_info.md"
