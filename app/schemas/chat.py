from typing import Literal

from pydantic import BaseModel, Field


class MessageItem(BaseModel):
    """대화 히스토리 아이템.

    Spring이 관리하는 히스토리를 AI 엔진으로 전달할 때 사용.
    content에는 순수 텍스트만 포함 (코드 스냅샷 제외).
    """
    role: Literal["user", "assistant"]
    content: str


class InvokeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    history: list[MessageItem] = Field(default_factory=list)
    summary: str = Field(default="", description="이전 대화 요약본 (Spring 관리)")
    code_snapshot: str = Field(default="", description="Spring이 판단해서 넣어주는 코드 스냅샷")


class RouteEvent(BaseModel):
    """라우터 노드 결과 이벤트."""
    route: Literal["code", "infra", "dev_qa", "unknown"]
    reason: str


class SummarizeRequest(BaseModel):
    """Spring이 요약 요청 시 전달하는 데이터"""
    history: list[MessageItem]
    current_summary: str = Field(default="", description="기존 요약본 (있으면 합쳐서 요약)")


class SummarizeResponse(BaseModel):
    """요약 결과 반환."""
    summary: str
