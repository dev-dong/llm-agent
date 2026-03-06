import logging
from functools import lru_cache

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

from app.agent.nodes.code import code_node
from app.agent.nodes.dev_qa import dev_qa_node
from app.agent.nodes.infra import infra_node
from app.agent.nodes.router import router_node
from app.agent.state import AgentState, NodeType

logger = logging.getLogger(__name__)


def _route_after_router(state: AgentState) -> NodeType:
    """조건부 엣지 함수: 라우터 결과에 따라 다음 노드 결정."""
    return state.route

async def unknown_node(state: AgentState) -> dict:
    from app.agent.prompts.templates import UNKNOWN_PROMPT
    from app.agent.utils import build_history
    from app.core.llm import LLMFactory

    chain = UNKNOWN_PROMPT | LLMFactory.get_general_llm()
    try:
        response = await chain.ainvoke({
            "user_query": state.user_query,
            "history": build_history(state.history, state.summary)
        })
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content, name="unknown")],
        }
    except Exception as e:
        return {"final_answer": f"오류가 발생했습니다: {e}", "error": str(e)}


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(AgentState)

    # 1. 노드 등록
    graph.add_node("router", router_node)
    graph.add_node("code", code_node)
    graph.add_node("infra", infra_node)
    graph.add_node("dev_qa", dev_qa_node)
    graph.add_node("unknown", unknown_node)

    # 2. 엣지 연결
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"code": "code", "infra": "infra", "dev_qa": "dev_qa", "unknown": "unknown"},
    )
    graph.add_edge("code", END)
    graph.add_edge("infra", END)
    graph.add_edge("dev_qa", END)
    graph.add_edge("unknown", END)

    return graph.compile()


def get_graph():
    return build_graph()
