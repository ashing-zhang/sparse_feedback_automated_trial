## AFAC 2026 赛道三：稀疏反馈自动化实验 Agent（A榜）

本仓库提供一个“配置驱动、DDD 分层、串行实验（禁止并行）”的最小可用实现，用于在 A 榜数据上自动生成提交文件：

- 产品分类任务：生成 `A1.csv`
- 产品推荐任务：生成 `A2.csv`
- 最终输出：`prediction.zip`（内含 `A1.csv` 与 `A2.csv`）

### 运行环境

- Python 3.12+
- 依赖见 `pyproject.toml`

### 快速开始

1. 复制环境变量文件

```bash
cp .env.example .env
```

2. 运行（默认读取 `configs/default.yaml`）

```bash
python -m afac_agent.run
```

3. 查看输出

```text
outputs/
  prediction.zip
  A1.csv
  A2.csv
  experiments.jsonl
  logs/
```

### 配置说明

- 运行入口不使用 argparse；配置文件路径通过环境变量指定：
  - `AFAC_CONFIG_PATH`：默认 `configs/default.yaml`
  - `AFAC_LOGGING_PATH`：默认 `configs/logging.yaml`
- 数据路径默认指向仓库内的 A 榜数据目录：
  - `data/classification_a/A1.npz`
  - `data/recommendation_a/*.csv`

### 设计说明（与 docs 对齐）

- 串行实验控制：每轮实验顺序执行，记录反馈到 `outputs/experiments.jsonl`，并根据验证集指标选择最优方案。
- 配置与代码分离：模型候选与搜索空间写在 YAML，代码只实现通用接口与执行流程。
- DDD 分层：
  - `afac_agent/domain`：领域模型与端口（ports）
  - `afac_agent/application`：用例与编排
  - `afac_agent/infrastructure`：配置、数据读取、模型实现、输出写入
  - `afac_agent/presentation`：组合根（composition root）与入口

