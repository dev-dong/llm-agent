import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.chat import ChatRequest, ChatResponse, RouteInfo
from app.agent.graph import get_graph
from app.agent.state import AgentState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="LLM 에이전트 질문")
async def chat(request: ChatRequest) -> ChatResponse:
    graph = get_graph()
    try:
        result: AgentState = await graph.ainvoke(
            AgentState(user_query=request.query)
        )
        return ChatResponse(
            answer=result.final_answer or "답변을 생성하지 못했습니다.",
            route=RouteInfo(selected=result.route, reason=result.routing_reason),
            has_error=result.error is not None,
        )
    except Exception as e:
        logger.error("[ChatAPI] 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream", summary="스트리밍 LLM 에이전트 질문")
async def chat_stream(request: ChatRequest):
    graph = get_graph()

    async def event_generator():
        try:
            async for event in graph.astream_events(
                    AgentState(user_query=request.query), version="v2"
            ):
                name = event.get("event", "")
                node = event.get("metadata", {}).get("langgraph_node", "")

                # 라우팅 완료
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

                # LLM 토큰 스트리밍
                elif name == "on_chat_model_stream" and node in ("code", "infra", "dev_qa"):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and chunk.content:
                        logger.info("[Stream] 토큰: %s", chunk.content)
                        yield _sse({"type": "token", "content": chunk.content})

            yield _sse({"type": "done"})
            logger.info("[Stream] 완료")

        except Exception as e:
            logger.error("[StreamAPI] 실패: %s", e)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
