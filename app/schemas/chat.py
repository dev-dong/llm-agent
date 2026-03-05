from typing import Literal
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """클라이언트 → 서버 요청 형식."""

    query: str = Field(
        min_length=1,
        max_length=10000,
        description="사용자 질문",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"query": "Spring Boot에서 JPA N+1 문제 해결 방법 알려줘"}
        }
    }


class RouteInfo(BaseModel):
    """라우팅 결과 정보."""

    selected: str = Field(description="선택된 노드 (code/infra/dev_qa)")
    reason: str = Field(description="선택 이유")


class ChatResponse(BaseModel):
    """서버 → 클라이언트 응답 형식."""

    answer: str
    route: RouteInfo
    has_error: bool = False


class HealthResponse(BaseModel):
    """헬스체크 응답."""

    status: Literal["ok", "error"]
    version: str = "1.0.0"
    ollama_url: str