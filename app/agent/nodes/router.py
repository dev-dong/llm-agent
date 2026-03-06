import logging

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.prompts.templates import ROUTER_PROMPT
from app.agent.state import AgentState, RouterDecision
from app.agent.utils import build_history
from app.core.llm import LLMFactory

logger = logging.getLogger(__name__)


async def router_node(state: AgentState) -> dict:
    logger.info("[Router] 시작 | query=%.50s", state.user_query)

    llm = LLMFactory.get_general_llm()
    chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision)

    try:
        decision: RouterDecision = await chain.ainvoke({
            "user_query": state.router_query or state.user_query,
            "history": build_history(state.history, state.summary)
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
