from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Ollama 서버 주소
    ollama_base_url: str = "http://localhost:11434"

    # 모델명
    code_model: str = "qwen3.5:9b"
    general_model: str = "lfm2:24b"

    # 서버 설정
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    # Agent 설정
    max_tokens: int = 4096
    temperature: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()
