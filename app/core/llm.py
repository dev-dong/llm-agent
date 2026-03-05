from functools import lru_cache
from langchain_ollama import ChatOllama
from app.core.config import get_settings


class LLMFactory:

    @staticmethod
    @lru_cache(maxsize=2)
    def get_code_llm() -> ChatOllama:
        """코드 특화 모델 (qwen3.5:9b)"""
        s = get_settings()
        return ChatOllama(
            model=s.code_model,
            base_url=s.ollama_base_url,
            num_predict=s.max_tokens
        )

    @staticmethod
    @lru_cache(maxsize=2)
    def get_general_llm() -> ChatOllama:
        """범용 모델 (lfm2:24b) - 라우터/인프라/QA."""
        s = get_settings()
        return ChatOllama(
            model=s.general_model,
            base_url=s.ollama_base_url,
            temperature=s.temperature,
            num_predict=s.max_tokens
        )
