import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama

from app.core.config import get_settings
from app.schemas.chat import ChatRequest, ChatResponse, RouteInfo
from app.agent.graph import get_graph
from app.agent.state import AgentState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


async def _summarize(history: list[dict], current_summary: str) -> str:
    """히스토리가 MAX_HISTORY """
    setting = get_settings()
    llm = ChatOllama(
        model=setting.general_model,
        base_url=setting.ollama_base_url,
        temperature=0.1,
        num_predict=500  # 요약은 짧게
    )

    history_text = "\n".join(
        f"{item['role'].upper()}: {item['content'][:200]}"  # 너무 길면 자름
        for item in history
    )

    prompt = f"""다음 대화를 5문장 이내로 요약해줘.
코드, 에러, 해결책이 있으면 반드시 포함해줘.
기존 요약이 있으면 합쳐서 작성해줘.

기존 요약:
{current_summary if current_summary else "없음"}

대화:
{history_text}
"""
    response = await llm.ainvoke(prompt)
    return response.content


@router.post("", response_model=ChatResponse, summary="LLM 에이전트 질문")
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    graph = get_graph()

    # MAX_HISTORY 초과 시 요약 트리거
    summary = request.summary
    history = [h.model_dump() for h in request.history]
    if len(history) > settings.max_history:
        summary = await _summarize(history, summary)
        history = history[-settings.max_history:]

    try:
        result: AgentState = await graph.ainvoke(
            AgentState(
                user_query=request.query,
                history=history,
                summary=summary
            )
        )
        return ChatResponse(
            answer=result.final_answer or "답변을 생성하지 못했습니다.",
            route=RouteInfo(selected=result.route, reason=result.routing_reason),
            has_error=result.error is not None,
            summary=result.summary or summary  # <- 요약본 응답에 포함
        )
    except Exception as e:
        logger.error("[ChatAPI] 실패: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream", summary="스트리밍 LLM 에이전트 질문")
async def chat_stream(request: ChatRequest):
    settings = get_settings()
    graph = get_graph()

    summary = request.summary
    history = [h.model_dump() for h in request.history]
    if len(history) > settings.max_history:
        summary = await _summarize(history, summary)
        history = history[-settings.max_history:]

    async def event_generator():
        try:
            async for event in graph.astream_events(
                    AgentState(
                        user_query=request.query,
                        history=history,
                        summary=summary
                    ), version="v2"
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
                        logger.info("[Stream]: %s", chunk.content)
                        yield _sse({"type": "token", "content": chunk.content})

            yield _sse({"type": "done", "summary": summary})
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
