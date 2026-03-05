import logging
from langchain_core.messages import AIMessage
from app.agent.state import AgentState
from app.agent.prompts.templates import CODE_PROMPT
from app.core.llm import LLMFactory

logger = logging.getLogger(__name__)


async def code_node(state: AgentState) -> dict:
    logger.info("[Code] 시작 | query=%.50s", state.user_query)

    chain = CODE_PROMPT | LLMFactory.get_code_llm()

    try:
        response = await chain.ainvoke({"user_query": state.user_query})
        logger.info("[Code] 완료 | length=%d", len(response.content))

        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content, name="code_expert")],
        }

    except Exception as e:
        logger.error("[Code] 실패: %s", e)
        return {
            "final_answer": f"코드 노드 오류: {e}",
            "error": str(e),
        }
