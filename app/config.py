import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    MAX_FILE_MB = 50  # file size guard
    VECTOR_DIM = 384  # MiniLM dimension

settings = Settings()