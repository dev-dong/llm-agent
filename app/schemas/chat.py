from typing import Literal
from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    """대화 히스토리 아이템."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    history: list[MessageItem] = Field(
        default_factory=list,
        description="이전 대화 기록 (Client가 관리)"
    )


class RouteInfo(BaseModel):
    selected: str
    reason: str


class ChatResponse(BaseModel):
    answer: str
    route: RouteInfo
    has_error: bool = False
