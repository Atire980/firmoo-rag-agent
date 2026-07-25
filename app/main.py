"""
FastAPI 应用入口：创建应用、注册路由、提供 /health 端点。

不在该文件中编写 RAG 或 Agent 业务逻辑。

启动方式：
    python -m uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。

    Returns:
        配置完成的 FastAPI 实例。
    """
    app = FastAPI(
        title="Firmoo 眼镜电商 RAG Agent",
        description="基于 LangGraph 的跨境电商智能客服 API",
        version="0.1.0",
    )

    # 注册路由
    app.include_router(chat_router)

    # 健康检查
    @app.get("/health")
    async def health_check():
        """健康检查端点。"""
        return {
            "status": "ok",
            "env": settings.APP_ENV,
        }

    # 启动日志
    @app.on_event("startup")
    async def startup():
        logger.info("=" * 40)
        logger.info("Firmoo RAG Agent 启动")
        logger.info("环境: %s", settings.APP_ENV)
        logger.info("地址: %s:%d", settings.APP_HOST, settings.APP_PORT)
        logger.info("=" * 40)

    return app


# 应用实例
app = create_app()
