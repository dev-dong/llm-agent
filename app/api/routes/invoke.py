import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.schemas.chat import InvokeRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream", summary="스트리밍 LLM 에이전트 질문")
async def chat_stream(request: InvokeRequest):
    graph = get_graph()
    summary = request.summary
    history = request.history

    async def event_generator():
        try:
            async for event in graph.astream_events(
                    AgentState(
                        user_query=request.query,
                        history=history,
                        summary=summary
                    ),
                    version="v2",
            ):
                name = event.get("event", "")
                node = event.get("metadata", {}).get("langgraph_node", "")

                if name == "on_chain_end" and node == "router":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        route = output.get("route")
                        reason = output.get("routing_reason")
                    else:
                        route = getattr(output, "route", None)
                        reason = getattr(output, "routing_reason", getattr(output, "reason", None))
                    logger.info("[Stream] 라우팅 완료 | route=%s | reason=%s", route, reason)
                    yield _sse({"type": "route", "route": route, "reason": reason})

                elif name == "on_chat_model_stream" and node in ("code", "infra", "dev_qa", "unknown"):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and chunk.content:
                        yield _sse({"type": "token", "content": chunk.content})

            yield _sse({"type": "done"})
            logger.info("[Stream] 완료")

        except Exception as err:
            logger.error("[StreamAPI] 실패: %s", err)
            yield _sse({"type": "error", "message": str(err)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
