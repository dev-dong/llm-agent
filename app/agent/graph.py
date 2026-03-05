import logging
from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from app.agent.state import AgentState, NodeType
from app.agent.nodes.router import router_node
from app.agent.nodes.code import code_node
from app.agent.nodes.infra import infra_node
from app.agent.nodes.dev_qa import dev_qa_node

logger = logging.getLogger(__name__)


def _route_after_router(state: AgentState) -> NodeType:
    """조건부 엣지 함수: 라우터 결과에 따라 다음 노드 결정."""
    if state.route in ("code", "infra", "dev_qa"):
        return state.route
    logger.warning("[Graph] unknown route → dev_qa 폴백")
    return "dev_qa"


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(AgentState)

    # 1. 노드 등록
    graph.add_node("router", router_node)
    graph.add_node("code", code_node)
    graph.add_node("infra", infra_node)
    graph.add_node("dev_qa", dev_qa_node)

    # 2. 엣지 연결
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {"code": "code", "infra": "infra", "dev_qa": "dev_qa"},
    )
    graph.add_edge("code", END)
    graph.add_edge("infra", END)
    graph.add_edge("dev_qa", END)

    return graph.compile()


def get_graph():
    return build_graph()
