#!/usr/bin/env sh

# 运行指南：
# 1) 在仓库根目录执行：
#    sh scripts/install.sh
# 2) 该脚本会安装项目依赖（基于 pyproject.toml）

set -eu

python -m pip install -U pip
python -m pip install .

