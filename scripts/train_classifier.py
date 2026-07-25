"""
八分类模型训练脚本。

使用 intent_train.csv 微调 all-MiniLM-L6-v2（AutoModelForSequenceClassification）。

数据格式：
    sample_id,query,label
    其中 label 为以下八种之一：
        商品咨询 / 镜片/度数咨询 / 物流查询 / 退换货售后
        / 订单修改/取消 / 投诉/人工客服 / 闲聊/其他 / 优惠/支付

输出：
    models/intent_classifier/   （保存完整模型和 tokenizer）

使用方式：
    python -m scripts.train_classifier

可选环境变量：
    TRAIN_EPOCHS=3          训练轮数
    TRAIN_BATCH_SIZE=16     批次大小
    TRAIN_LEARNING_RATE=2e-5 学习率
    TRAIN_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2  基础模型
"""

import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from app.config import BUSINESS_INTENT_LABELS, BUSINESS_INTENT_TO_ID
from app.utils.logger import get_logger

logger = get_logger(__name__)

# 默认超参数（可通过环境变量覆盖）0
TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "3"))
TRAIN_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "16"))
TRAIN_LEARNING_RATE = float(os.getenv("TRAIN_LEARNING_RATE", "2e-5"))
TRAIN_MODEL_NAME = os.getenv(
    "TRAIN_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
OUTPUT_DIR = "models/intent_classifier"
DATA_PATH = "data/raw/intent_train.csv"


def _load_and_prepare_data(csv_path: str) -> tuple:
    """加载训练数据，划分训练/验证集。

    Args:
        csv_path: intent_train.csv 路径。

    Returns:
        (train_texts, train_labels, val_texts, val_labels)
    """
    import torch
    from torch.utils.data import Dataset

    logger.info("加载训练数据: %s", csv_path)

    df = pd.read_csv(csv_path)
    logger.info("原始数据: %d 条", len(df))

    # 过滤无效标签
    df = df[df["label"].isin(BUSINESS_INTENT_LABELS)]
    logger.info("有效数据: %d 条", len(df))

    # 标签分布
    for label in BUSINESS_INTENT_LABELS:
        count = (df["label"] == label).sum()
        logger.info("  %s: %d", label, count)

    # 转换为 label_id
    df["label_id"] = df["label"].map(BUSINESS_INTENT_TO_ID)

    # 划分训练/验证集 (80/20, 按标签分层)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42,
        stratify=df["label_id"],
    )

    logger.info("训练集: %d, 验证集: %d", len(train_df), len(val_df))

    class IntentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    return train_df, val_df, IntentDataset


def _compute_metrics(pred):
    """计算评估指标。"""
    from sklearn.metrics import accuracy_score, f1_score

    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main():
    """训练主流程。"""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
    import torch

    logger.info("=" * 50)
    logger.info("八分类模型训练")
    logger.info("基础模型: %s", TRAIN_MODEL_NAME)
    logger.info("标签数量: %d", len(BUSINESS_INTENT_LABELS))
    logger.info("输出目录: %s", OUTPUT_DIR)
    logger.info("=" * 50)

    # 1. 加载数据
    train_df, val_df, IntentDataset = _load_and_prepare_data(DATA_PATH)

    # 2. 加载 tokenizer 和模型
    logger.info("加载 tokenizer: %s", TRAIN_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TRAIN_MODEL_NAME)

    logger.info("加载模型: %s (num_labels=%d)", TRAIN_MODEL_NAME, len(BUSINESS_INTENT_LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(
        TRAIN_MODEL_NAME,
        num_labels=len(BUSINESS_INTENT_LABELS),
        problem_type="single_label_classification",
    )

    # 3. 构建 Dataset
    train_dataset = IntentDataset(
        train_df["query"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
    )
    val_dataset = IntentDataset(
        val_df["query"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
    )

    logger.info("训练样本: %d, 验证样本: %d", len(train_dataset), len(val_dataset))

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE * 2,
        learning_rate=TRAIN_LEARNING_RATE,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 6. 训练
    logger.info("开始训练...")
    trainer.train()

    # 7. 评估
    logger.info("最终评估...")
    eval_results = trainer.evaluate()
    logger.info("验证集结果: %s", eval_results)

    # 8. 保存模型
    logger.info("保存模型到 %s", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 9. 保存标签映射
    import json
    label_config = {
        "labels": BUSINESS_INTENT_LABELS,
        "label_to_id": BUSINESS_INTENT_TO_ID,
    }
    with open(f"{OUTPUT_DIR}/label_config.json", "w", encoding="utf-8") as f:
        json.dump(label_config, f, ensure_ascii=False, indent=2)

    logger.info("=" * 50)
    logger.info("训练完成！模型已保存至 %s", OUTPUT_DIR)
    logger.info("请将 .env 中的 CLASSIFIER_MODEL_PATH 设置为: %s",
                os.path.abspath(OUTPUT_DIR))
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
