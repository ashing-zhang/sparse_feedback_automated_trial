"""
运行指南：
1) 复制 .env.example 为 .env，并按需修改 AFAC_CONFIG_PATH / AFAC_LOGGING_PATH
2) 确认 data/ 目录包含 A 榜数据（本仓库默认已提供）
3) 运行：
   python -m afac_agent.run

说明：
- 不使用 argparse；配置文件路径从环境变量读取（缺省为 configs/default.yaml）
- 输出目录默认为 outputs/，会生成 A1.csv、A2.csv 与 prediction.zip
"""

from __future__ import annotations

from afac_agent.presentation.main import main


if __name__ == "__main__":
    raise SystemExit(main())

