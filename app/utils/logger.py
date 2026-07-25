"""统一日志配置。禁止在业务代码中使用 print() 作为正式日志。"""

import logging
import sys

# 单一日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """获取一个已配置好的 logger 实例。

    Args:
        name: 通常传入 __name__。

    Returns:
        配置完成的 Logger，输出到 stdout。
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)

    # 默认 INFO 级别，可通过环境变量 APP_LOG_LEVEL 覆盖
    log_level = logging.INFO
    logger.setLevel(log_level)

    # 避免日志向上传播到 root logger 造成重复
    logger.propagate = False

    return logger
