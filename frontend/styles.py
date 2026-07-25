"""
集中存放 Streamlit 页面的自定义 CSS 样式。

不要在 streamlit_app.py 中堆放大量 CSS 字符串，
全部样式统一从这里引用。
"""

# ---- 全局 CSS ----
GLOBAL_CSS = """
<style>
    /* ===== 基础排版 ===== */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* ===== 欢迎区域 ===== */
    .welcome-card {
        text-align: center;
        padding: 2.5rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
        border: 1px solid #d6e9f8;
        margin-bottom: 1rem;
    }
    .welcome-card h2 {
        color: #1a73e8;
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
    }
    .welcome-card p {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    /* ===== 快捷问题按钮 ===== */
    .quick-btn-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin-top: 1.2rem;
    }

    /* ===== 助手回答元信息 ===== */
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.4rem;
    }
    .meta-tag {
        display: inline-block;
        font-size: 0.78rem;
        color: #666;
        background: #f5f5f5;
        border-radius: 6px;
        padding: 2px 10px;
        border: 1px solid #eaeaea;
    }
    .meta-tag.confidence-high {
        color: #1b7a1b;
        background: #eaf7ea;
        border-color: #b6dfb6;
    }
    .meta-tag.confidence-mid {
        color: #b06b00;
        background: #fff8ed;
        border-color: #f5d8a0;
    }
    .meta-tag.confidence-low {
        color: #c0392b;
        background: #fdf0ef;
        border-color: #f1c4c2;
    }

    /* ===== 侧边栏 ===== */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }

    /* ===== 服务状态指示 ===== */
    .status-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-dot.online { background: #27ae60; box-shadow: 0 0 6px #27ae6088; }
    .status-dot.offline { background: #e74c3c; box-shadow: 0 0 6px #e74c3c88; }
    .status-dot.unknown { background: #f39c12; box-shadow: 0 0 6px #f39c1288; }

    /* ===== 错误消息 ===== */
    .error-box {
        background: #fff5f5;
        border-left: 4px solid #e74c3c;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        color: #a8231a;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .error-box code {
        background: #fce4e4;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #c0392b;
    }

    /* ===== 注意事项 ===== */
    .note-box {
        background: #fffdf5;
        border-left: 4px solid #f39c12;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #8b6d1a;
        line-height: 1.6;
        margin-top: 1rem;
    }
</style>
"""
