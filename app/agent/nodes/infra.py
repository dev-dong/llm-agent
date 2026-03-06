import logging

from langchain_core.messages import AIMessage

from app.agent.prompts.templates import INFRA_PROMPT
from app.agent.state import AgentState
from app.agent.utils import build_history
from app.core.llm import LLMFactory

logger = logging.getLogger(__name__)


async def infra_node(state: AgentState) -> dict:
    logger.info("[Infra] 시작 | query=%.50s", state.user_query)

    chain = INFRA_PROMPT | LLMFactory.get_general_llm()

    try:
        response = await chain.ainvoke({
            "user_query": state.user_query,
            "history": build_history(state.history, state.summary)
        })
        logger.info("[Infra] 완료 | length=%d", len(response.content))
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content, name="infra_expert")],
        }
    except Exception as e:
        logger.error("[Infra] 실패: %s", e)
        return {"final_answer": f"인프라 노드 오류: {e}", "error": str(e)}
