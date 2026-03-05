import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama

from app.core.config import get_settings
from app.schemas.chat import ChatRequest
from app.agent.graph import get_graph
from app.agent.state import AgentState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _read_file_sync(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"파일을 찾을 수 없습니다: {file_path}")
    if not path.is_file():
        raise ValueError(f"파일이 아닙니다: {file_path}")
    if path.stat().st_size > 100_000:
        raise ValueError("파일이 너무 큽니다. (최대 100KB)")
    return path.read_text(encoding="utf-8")


async def _summarize(history: list[dict], current_summary: str) -> str:
    settings = get_settings()
    llm = ChatOllama(
        model=settings.general_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_predict=500,
    )
    history_text = "\n".join(
        f"{item['role'].upper()}: {item['content'][:200]}"
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


@router.post("/stream", summary="스트리밍 LLM 에이전트 질문")
async def chat_stream(request: ChatRequest):
    settings = get_settings()
    graph = get_graph()

    # 파일 경로가 있으면 파일 내용을 query에 합침
    query = request.query
    if request.file_path:
        try:
            file_content = await asyncio.to_thread(_read_file_sync, request.file_path)
            file_name = Path(request.file_path).name
            query = f"{query}\n\n**파일: {file_name}**\n```\n{file_content}\n```"
            logger.info("[Stream] 파일 첨부 | path=%s", request.file_path)
        except ValueError as e:
            async def error_gen():
                yield _sse({"type": "error", "message": str(e)})
            return StreamingResponse(error_gen(), media_type="text/event-stream")

    # MAX_HISTORY 초과 시 요약 트리거
    summary = request.summary
    history = [h.model_dump() for h in request.history]
    if len(history) > settings.max_history:
        summary = await _summarize(history, summary)
        history = history[-settings.max_history:]

    async def event_generator():
        try:
            async for event in graph.astream_events(
                    AgentState(
                        user_query=query,       # ← request.query → query 로 수정
                        history=history,
                        summary=summary,
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

                elif name == "on_chat_model_stream" and node in ("code", "infra", "dev_qa"):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and chunk.content:
                        logger.info("[Stream]: %s", chunk.content)
                        yield _sse({"type": "token", "content": chunk.content})

            yield _sse({"type": "done", "summary": summary})
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