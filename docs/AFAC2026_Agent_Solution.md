# AFAC 2026 赛道三 自动化实验 Agent 解决方案设计方案

## 1. 总体目标

根据 SPEC
要求，本方案设计一个面向"稀疏反馈、有限预算、禁止并行"的自动化实验 Agent
系统。

系统目标：

-   自动读取任务数据与历史实验结果；
-   自主设计模型实验方案；
-   执行单轮实验并获取反馈；
-   基于实验结果进行诊断、反思和策略调整；
-   在预算耗尽前选择最优预测器；
-   输出符合要求的分类预测文件和推荐预测文件。

核心思想：

> 将机器学习实验过程建模为一个由 LLM 驱动的闭环优化系统。

------------------------------------------------------------------------

# 2. 系统总体架构

系统由以下模块组成：

                     用户任务输入
                          |
                          v
                  Experiment Agent
                          |
     ------------------------------------------------
     |              |              |                 |
     v              v              v                 v
    Task      Memory System   Experiment      Evaluator
    Analyzer                  Controller
     |
     v
    Model Zoo
     |
     v
    Graph / Recommendation Predictor
     |
     v
    Prediction Generator

------------------------------------------------------------------------

# 3. Agent 核心设计

## 3.1 Agent 状态设计

Agent 不维护传统强化学习 State Space，而维护实验上下文：

``` json
{
 "task_type":"classification/recommendation",
 "dataset_profile":{},
 "experiment_history":[],
 "best_solution":{},
 "remaining_budget":{},
 "failure_cases":[]
}
```

------------------------------------------------------------------------

## 3.2 Agent 推理循环

每一次实验执行：

    Observe
     |
     v
    Diagnosis
     |
     v
    Experiment Design
     |
     v
    Execute
     |
     v
    Evaluate
     |
     v
    Memory Update
     |
     v
    Next Decision

对应 LLM Agent Prompt：

-   当前实验表现如何？
-   哪些因素可能限制性能？
-   下一轮应该探索还是利用？
-   是否应该提前停止？

------------------------------------------------------------------------

# 4. 实验记忆系统

采用 Hybrid Memory：

## 4.1 Short Term Memory

保存当前数据集实验：

内容：

-   最近实验配置
-   指标变化
-   失败原因
-   当前最佳模型

## 4.2 Long Term Memory

保存跨任务经验：

例如：

    GraphSAGE + feature normalization
    适用于：
    - 稀疏节点分类
    - 小规模金融图

    GRU4Rec
    适用于：
    - 短序列推荐

    LightGCN
    适用于：
    - 用户-item交互预测

采用：

-   Vector Database
-   Qwen text-embedding-v4 embedding

实现经验检索。

------------------------------------------------------------------------

# 5. 产品分类任务方案

## 5.1 Baseline模型池

Candidate Model Zoo：

  模型                用途
  ------------------- ------------------
  MLP                 节点特征baseline
  GCN                 图卷积
  GraphSAGE           稀疏邻居聚合
  GAT                 注意力机制
  Label Propagation   低监督场景

------------------------------------------------------------------------

## 5.2 自动优化空间

Agent 可以调整：

### 模型

-   GCN / GraphSAGE / GAT

### 超参数

-   hidden dimension
-   learning rate
-   dropout
-   weight decay
-   epoch

### 数据处理

-   feature normalization
-   self-loop
-   adjacency normalization

------------------------------------------------------------------------

# 6. 产品推荐任务方案

## 6.1 推荐模型池

包含：

  模型               作用
  ------------------ ----------------
  Popularity Model   冷启动baseline
  ItemCF             序列相似
  GRU4Rec            行为序列
  LightGCN           用户-item图
  Two Tower          特征匹配

------------------------------------------------------------------------

## 6.2 推荐Pipeline

    User History
          |
          v
    Sequence Encoder
          |
          v
    User Embedding

    Item Feature
          |
          v
    Item Embedding

           |
           v

    Similarity Ranking

           |
           v

    Top10 Recommendation

------------------------------------------------------------------------

# 7. Experiment Controller

由于禁止并行实验，因此采用 Sequential Optimization。

每轮：

    budget = total_budget - consumed_time


    if improvement > threshold:

        exploit(best direction)

    else:

        explore(new strategy)

策略：

-   前30%预算探索
-   中间50%预算优化
-   最后20%预算稳定最佳方案

------------------------------------------------------------------------

# 8. LLM Agent设计

## Planner Agent

负责：

-   分析任务
-   制定实验计划

## Scientist Agent

负责：

-   分析实验结果
-   提出优化假设

## Engineer Agent

负责：

-   生成模型配置
-   调整代码参数

## Reviewer Agent

负责：

-   判断是否继续实验

------------------------------------------------------------------------

# 9. RAG设计

采用 Experiment Knowledge RAG：

数据来源：

-   历史实验日志
-   模型结果
-   图学习论文摘要
-   推荐算法经验

Retrieval:

    Current Problem

           |

    Embedding

           |

    Vector Search

           |

    Relevant Experience

           |

    LLM Decision

------------------------------------------------------------------------

# 10. 运行流程

    Start

     |
    Load Dataset

     |
    Task Analyzer

     |
    Initialize Model Candidates

     |
    while budget:

          Agent Diagnosis

          Generate Experiment

          Run Training

          Evaluate

          Update Memory


     |
    Select Best Model

     |
    Generate Submission

     |
    End

------------------------------------------------------------------------

# 11. 工程实现架构

推荐技术栈：

## Agent Framework

-   LangGraph

## LLM

-   Qwen API

## ML

-   PyTorch
-   PyTorch Geometric
-   LightGBM

## Storage

-   SQLite实验记录

-   FAISS向量数据库

## Logging

MLflow

------------------------------------------------------------------------

# 12. 提分关键策略

## 分类任务

优先级：

1.  GraphSAGE
2.  Feature Engineering
3.  Ensemble

## 推荐任务

优先级：

1.  LightGCN
2.  Sequence Model
3.  Hybrid Ranking

## Agent优化重点

不是搜索大量模型，而是：

-   快速定位有效方向
-   避免重复实验
-   利用历史经验

------------------------------------------------------------------------

# 13. 最终提交输出

Agent自动生成：

    prediction.zip

     ├── A1.csv
     |
     └── A2.csv

满足：

-   字段规范
-   排序要求
-   item合法性
-   节点编号一致

------------------------------------------------------------------------

# 14. 总结

本方案将赛题抽象为：

> 一个由 LLM 驱动的自动化机器学习科学家系统。

通过：

-   Agent规划
-   Experiment Memory
-   RAG经验检索
-   Sequential Optimization
-   Graph/Recommender Model Zoo

实现有限预算下实验效率最大化。
