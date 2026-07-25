"""
建库入口脚本。

使用方式：
    python -m scripts.build_index

可选参数（通过环境变量控制）：
    DATA_DIR=        原始数据目录，默认 data/raw
    DROP_EXISTING=1  删除已有 Collection 后重建
"""

import os
import sys

# 确保项目根目录在 sys.path 中（从项目根目录运行时自然满足）
# 禁止使用 sys.path.append()，本项目通过 python -m 运行


def main():
    """建库主流程。"""
    data_dir = os.getenv("DATA_DIR", "data/raw")
    drop_existing = os.getenv("DROP_EXISTING", "0") == "1"

    print(f"数据目录: {data_dir}")
    print(f"重建模式: {drop_existing}")

    # 延迟导入，避免在脚本加载时触发模型加载
    from app.rag.ingest import run_ingest
    run_ingest(data_dir=data_dir, drop_existing=drop_existing)


if __name__ == "__main__":
    main()
