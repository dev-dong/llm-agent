from typing import Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

NodeType = Literal["code", "infra", "dev_qa", "unknown"]


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_query: str = Field(default="")
    history: list[dict] = Field(default_factory=list)
    route: NodeType = Field(default="unknown")
    routing_reason: str = Field(default="")
    final_answer: str = Field(default="")
    error: str | None = Field(default=None)


class RouterDecision(BaseModel):
    """라우터 LLM의 구조화 출력 스키마."""

    route: NodeType = Field(description="선택한 전문 분야 노드")
    reason: str = Field(description="선택한 이유 (한국어 1~2문장)")
