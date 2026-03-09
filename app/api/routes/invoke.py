import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.core.llm import LLMFactory
from app.schemas.chat import InvokeRequest, SummarizeResponse, SummarizeRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream", summary="스트리밍 LLM 에이전트 질문")
async def chat_stream(request: InvokeRequest):
    graph = get_graph()
    summary = request.summary
    history = [h.model_dump() for h in request.history]

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


@router.post("/summarize", response_model=SummarizeResponse, summary="대화 요약 생성")
async def summarize(request: SummarizeRequest) -> SummarizeResponse:
    """단 하나의 책임: LLM으로 대화 요약 생성 후 반환."""
    llm = LLMFactory.get_general_llm()

    history_text = "\n".join(f"{item.role.upper()}: {item.content[:200]}" for item in request.history)

    prompt = f"""다음 대화를 5문장 이내로 요약해줘.
코드, 에러, 해결책이 있으면 반드시 포함해줘.
기존 요약이 있으면 합쳐서 작성해줘.

기존 요약:
{request.current_summary if request.current_summary else "없음"}

대화:
{history_text}
"""
    response = await llm.ainvoke(prompt)
    return SummarizeResponse(summary=response.content)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
