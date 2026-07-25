"""
八分类意图识别器：使用 Hugging Face Transformers 微调后的 all-MiniLM-L6-v2 模型。

- 单例加载模型和 tokenizer，禁止每次请求重新加载。
- 在异步节点中通过 asyncio.to_thread() 调用同步模型推理。
- 对外暴露 classify(text) -> tuple[str, float]。
- 返回八种业务意图标签之一 + 置信度。

训练脚本：scripts/train_classifier.py
训练数据：data/raw/intent_train.csv（5001 条，八种标签）
"""

import asyncio
from typing import Any

import numpy as np

from app.config import (
    settings,
    BUSINESS_INTENT_LABELS,
    BUSINESS_INTENT_TO_ID,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 模型单例
_classifier_model: Any = None
_classifier_tokenizer: Any = None
_device: str = "cpu"

# 回退模式：正式模型未训练时为 True，使用规则 + 随机占位
_fallback_mode: bool = False


def _load_classifier():
    """惰性加载八分类模型和 tokenizer。"""
    global _classifier_model, _classifier_tokenizer, _device, _fallback_mode

    if _classifier_model is not None:
        return

    model_path = settings.CLASSIFIER_MODEL_PATH
    if not model_path:
        logger.warning(
            "CLASSIFIER_MODEL_PATH 未配置，八分类模型不可用。"
            "请先运行 python -m scripts.train_classifier 训练模型，"
            "然后将模型路径填入 .env。当前使用回退模式。"
        )
        _fallback_mode = True
        return

    logger.info("正在加载八分类模型: %s", model_path)
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"

        _classifier_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _classifier_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _classifier_model.to(_device)
        _classifier_model.eval()

        logger.info(
            "八分类模型加载完成 (device=%s, num_labels=%d, vocab_size=%d)",
            _device,
            _classifier_model.config.num_labels,
            _classifier_tokenizer.vocab_size,
        )
    except Exception as e:
        logger.error("八分类模型加载失败: %s，使用回退模式", e)
        _fallback_mode = True


def _model_inference(text: str) -> tuple[int, float]:
    """用 Transformers 模型推理，返回 (label_index, confidence)。"""
    import torch

    inputs = _classifier_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _classifier_model(**inputs)
        logits = outputs.logits[0].cpu().numpy()

    # softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    label_idx = int(np.argmax(probs))
    confidence = float(probs[label_idx])

    return label_idx, confidence


def classify(text: str) -> tuple[str, float]:
    """对文本进行八种业务意图分类。

    Args:
        text: 用户输入文本。

    Returns:
        (business_intent, confidence)
        business_intent 为八种标签之一。
        回退模式下始终返回 ("商品咨询", 0.50)。
    """
    if _fallback_mode:
        _load_classifier()
        # 加载后再次检查：可能 CLASSIFIER_MODEL_PATH 已配置但加载失败
        if _fallback_mode:
            return "商品咨询", 0.50

    # 确保模型已加载
    if _classifier_model is None:
        _load_classifier()
        if _fallback_mode:
            return "商品咨询", 0.50

    try:
        label_idx, confidence = _model_inference(text)
        if 0 <= label_idx < len(BUSINESS_INTENT_LABELS):
            label = BUSINESS_INTENT_LABELS[label_idx]
        else:
            label = "商品咨询"
            logger.warning("模型输出异常 label_idx=%d，回退为 商品咨询", label_idx)

        logger.debug("八分类结果: text=%.40s → %s (%.3f)", text, label, confidence)
        return label, confidence

    except Exception as e:
        logger.error("八分类推理失败: %s，回退为 商品咨询", e)
        return "商品咨询", 0.50


async def async_classify(text: str) -> tuple[str, float]:
    """异步分类入口，在 executor 中执行同步推理。"""
    return await asyncio.to_thread(classify, text)
