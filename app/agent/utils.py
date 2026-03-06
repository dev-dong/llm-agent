from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from app.core.config import get_settings


def build_history(history: list[dict], summary: str = "") -> list:
    """
    대화 히스토리를 LangChain 메시지 리스트로 변환.

    - 요약본이 있으면 SystemMessage로 맨 앞에 삽입
    - 최근 max_history 개수만 유지
    """
    settings = get_settings()
    result = []

    # 요약본이 있으면 system 메시지로 먼저 추가
    if summary:
        result.append(SystemMessage(content=f"[이전 대화 요약]\n{summary}"))

    # 최근 MAX_HISTORY개만 잘라서 추가
    recent = history[-settings.max_history:]
    for item in recent:
        if item.get("role") == "user":
            result.append(HumanMessage(content=item["content"]))
        elif item.get("role") == "assistant":
            result.append(AIMessage(content=item["content"]))
    return result
