import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent.prompts.templates import ROUTER_PROMPT
from app.agent.state import AgentState, RouterDecision
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


async def router_node(state: AgentState) -> dict:
    logger.info("[Router] 시작 | query=%.50s", state.user_query)

    llm = LLMFactory.get_general_llm()
    chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)

    try:
        decision: RouterDecision = await chain.ainvoke({
            "user_query": state.user_query,
            "history": _build_history(state.history, state.summary)
        })
        logger.info("[Router] 결정 | route=%s", decision.route)

        return {
            "route": decision.route,
            "routing_reason": decision.reason,
            "messages": [
                HumanMessage(content=state.user_query),
                AIMessage(content=f"[라우터] {decision.reason} → {decision.route}"),
            ],
        }

    except Exception as e:
        logger.error("[Router] 실패: %s", e)
        return {
            "route": "dev_qa",
            "routing_reason": f"라우팅 실패, dev_qa로 폴백: {e}",
            "error": str(e),
        }
