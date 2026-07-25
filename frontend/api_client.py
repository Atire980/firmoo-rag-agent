"""
封装与 FastAPI 后端的 HTTP 通信。

提供：
- health_check()       → 健康检查
- send_chat_message()  → 发送聊天请求

统一处理超时、连接失败、HTTP 错误和 JSON 解析异常。
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv

# 加载 .env（如果尚未加载）
load_dotenv()

# ---- 配置 ----
API_BASE_URL: str = os.getenv("FRONTEND_API_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT: int = int(os.getenv("FRONTEND_REQUEST_TIMEOUT", "120"))


class ApiClientError(Exception):
    """API 客户端统一异常。"""

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def health_check() -> dict[str, Any]:
    """调用 GET /health，返回后端状态。

    Returns:
        {"status": "ok", "env": "development"} 或类似结构。

    Raises:
        ApiClientError: 连接失败 / 超时 / HTTP 错误。
    """
    try:
        resp = requests.get(
            f"{API_BASE_URL}/health",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise ApiClientError(
            "无法连接到后端服务，请确认 FastAPI 已启动。",
            detail=f"连接地址: {API_BASE_URL}",
        )
    except requests.exceptions.Timeout:
        raise ApiClientError(
            "连接后端超时，请检查网络或后端服务状态。",
            detail=f"请求地址: {API_BASE_URL}/health",
        )
    except requests.exceptions.HTTPError as e:
        raise ApiClientError(
            f"后端返回了异常状态码: {e.response.status_code if e.response else '未知'}",
            detail=str(e),
        )
    except requests.exceptions.JSONDecodeError:
        raise ApiClientError(
            "后端返回的内容不是合法的 JSON，请检查后端是否正常运行。",
            detail="JSON 解析失败",
        )
    except Exception as e:
        raise ApiClientError("健康检查时发生未知错误。", detail=str(e))


def send_chat_message(
    user_id: str,
    session_id: str,
    message: str,
) -> dict[str, Any]:
    """调用 POST /api/chat，发送聊天消息并获取回复。

    Args:
        user_id:    用户唯一标识。
        session_id: 会话唯一标识。
        message:    用户输入消息（将在发送前 strip）。

    Returns:
        {
            "answer": "...",
            "sources": [...],
            "confidence": 0.99,
            "answer_mode": "rag",
            "business_intent": "商品咨询",
        }

    Raises:
        ApiClientError: 各类通信异常。
    """
    payload: dict[str, str] = {
        "user_id": user_id,
        "session_id": session_id,
        "message": message.strip(),
    }

    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/chat",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        # 基础校验：至少要有 answer 字段
        if "answer" not in data:
            raise ApiClientError(
                "后端响应缺少 answer 字段，响应结构异常。",
                detail=str(data)[:300],
            )

        return data

    except requests.exceptions.ConnectionError:
        raise ApiClientError(
            "无法连接到后端服务。请先启动 FastAPI：\n\n"
            "```bash\n"
            "python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000\n"
            "```",
            detail=f"连接地址: {API_BASE_URL}",
        )
    except requests.exceptions.Timeout:
        raise ApiClientError(
            f"请求超时（{REQUEST_TIMEOUT}s），后端处理时间过长，请稍后重试。",
            detail="请求超时",
        )
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "未知"
        raise ApiClientError(
            f"后端返回了异常响应（HTTP {status}），请稍后重试。",
            detail=str(e),
        )
    except requests.exceptions.JSONDecodeError:
        raise ApiClientError(
            "后端返回的内容无法解析为 JSON，请检查后端日志。",
            detail="JSON 解析失败",
        )
    except ApiClientError:
        raise
    except Exception as e:
        raise ApiClientError("发送消息时发生未知错误。", detail=str(e))
