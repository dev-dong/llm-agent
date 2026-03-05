import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent.prompts.templates import INFRA_PROMPT
from app.agent.state import AgentState
from app.core.config import get_settings
from app.core.llm import LLMFactory

logger = logging.getLogger(__name__)


def _build_history(history: list[dict], summary: str = "") -> list:
    """
    요약본이 있으면 맨 앞에 system 메시지로 삽입.
    히스토리는 최근 MAX_HISTORY개만 유지
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


async def infra_node(state: AgentState) -> dict:
    logger.info("[Infra] 시작 | query=%.50s", state.user_query)

    chain = INFRA_PROMPT | LLMFactory.get_general_llm()

    try:
        response = await chain.ainvoke({
            "user_query": state.user_query,
            "history": _build_history(state.history, state.summary)
        })
        logger.info("[Infra] 완료 | length=%d", len(response.content))
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content, name="infra_expert")],
        }
    except Exception as e:
        logger.error("[Infra] 실패: %s", e)
        return {"final_answer": f"인프라 노드 오류: {e}", "error": str(e)}