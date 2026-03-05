from typing import Literal
from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    """대화 히스토리 아이템."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    history: list[MessageItem] = Field(default_factory=list)
    summary: str = Field(default="", description="이전 대화 요약본")
    file_path: str = Field(default="", description="리뷰할 파일 경로 (선택)")


class RouteInfo(BaseModel):
    selected: str
    reason: str


class ChatResponse(BaseModel):
    answer: str
    route: RouteInfo
    has_error: bool = False
    summary: str = ""
