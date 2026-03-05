import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.routes import chat
from app.agent.graph import get_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 서버 시작 시 ──────────────────────────
    settings = get_settings()
    logger.info("🚀 LLM Agent 서버 시작")
    logger.info("   Ollama  : %s", settings.ollama_base_url)
    logger.info("   Code    : %s", settings.code_model)
    logger.info("   General : %s", settings.general_model)

    get_graph()  # 첫 요청 전에 그래프 미리 컴파일
    logger.info("✅ LangGraph 컴파일 완료")

    yield  # ← 서버 실행 중

    # ── 서버 종료 시 ──────────────────────────
    logger.info("👋 서버 종료")


def create_app() -> FastAPI:
    fast_api_app = FastAPI(
        title="내부망 LLM Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    fast_api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fast_api_app.include_router(chat.router, prefix="/api/v1")

    return fast_api_app


app = create_app()


def start():
    """uv run serve 진입점."""
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level="info",
    )
