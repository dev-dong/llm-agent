import logging
from langchain_core.messages import HumanMessage, AIMessage
from app.agent.state import AgentState
from app.agent.prompts.templates import INFRA_PROMPT
from app.core.llm import LLMFactory

logger = logging.getLogger(__name__)


def _build_history(history: list[dict]) -> list:
    result = []
    for item in history:
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
            "history": _build_history(state.history),
        })
        logger.info("[Infra] 완료 | length=%d", len(response.content))
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content, name="infra_expert")],
        }
    except Exception as e:
        logger.error("[Infra] 실패: %s", e)
        return {"final_answer": f"인프라 노드 오류: {e}", "error": str(e)}