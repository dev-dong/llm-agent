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
    """AI 엔진 실행 요청.

    Spring이 비즈니스 로직을 처리한 뒤 AI 엔진에 전달하는 계약.
    - query   : 파일 내용이 포함된 완성된 질문 (Spring이 조합)
    - history : 코드 스냅샷이 제거된 순수 텍스트 히스토리 (Spring이 정리)
    - summary : 이전 대화 요약본 (Spring이 관리)
    """
    query: str = Field(min_length=1, max_length=20000)
    history: list[MessageItem] = Field(default_factory=list)
    summary: str = Field(default="", description="이전 대화 요약본 (Spring 관리)")


class RouteEvent(BaseModel):
    """라우터 노드 결과 이벤트."""
    route: Literal["code", "infra", "dev_qa", "unknown"]
    reason: str
