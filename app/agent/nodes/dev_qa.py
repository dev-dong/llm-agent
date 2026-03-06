import logging

from langchain_core.messages import AIMessage

from app.agent.prompts.templates import DEV_QA_PROMPT
from app.agent.state import AgentState
from app.agent.utils import build_history
from app.core.llm import LLMFactory

logger = logging.getLogger(__name__)


async def dev_qa_node(state: AgentState) -> dict:
    logger.info("[DevQA] 시작 | query=%.50s", state.user_query)

    chain = DEV_QA_PROMPT | LLMFactory.get_general_llm()

    try:
        response = await chain.ainvoke({
            "user_query": state.user_query,
            "history": build_history(state.history, state.summary)
        })
        logger.info("[DevQA] 완료 | length=%d", len(response.content))
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content, name="dev_qa_expert")],
        }
    except Exception as e:
        logger.error("[DevQA] 실패: %s", e)
        return {"final_answer": f"QA 노드 오류: {e}", "error": str(e)}
