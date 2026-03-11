import logging

from langchain_core.messages import AIMessage

from app.agent.prompts.templates import CODE_PROMPT
from app.agent.state import AgentState
from app.agent.utils import build_history
from app.core.llm import LLMFactory

logger = logging.getLogger(__name__)


async def code_node(state: AgentState) -> dict:
    logger.info("[Code] 시작 | query=%.50s", state.user_query)

    chain = CODE_PROMPT | LLMFactory.get_code_llm()

    code_context = ""
    if state.code_snapshot:
        code_context = (
            "\n\n## 현재 작업 중인 코드\n"
            "아래는 사용자가 작업 중인 최신 코드입니다. "
            "질문에 이 코드가 관련되면 참고하세요.\n"
            f"```\n{state.code_snapshot}\n```"
        )

    try:
        response = await chain.ainvoke({
            "user_query": state.user_query,
            "history": build_history(state.history, state.summary),
            "code_context": code_context
        })
        logger.info("[Code] 완료 | length=%d", len(response.content))
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content, name="code_expert")],
        }
    except Exception as e:
        logger.error("[Code] 실패: %s", e)
        return {"final_answer": f"코드 노드 오류: {e}", "error": str(e)}
