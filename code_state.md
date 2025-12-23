# Codex 工程说明文件（code_state.md）

## 0. 项目目标
实现支持以下实验的科研代码项目：
- 文件级漏洞检测（含多种消融）
- 行级漏洞定位（评分、子图提取、指标评估）

---

## 1. 项目结构（当前/计划）
project/
├── configs/
│   ├── file_level.yaml
│   └── line_level.yaml
├── data/
│   ├── processed/
│   └── splits/
├── dataset/
│   ├── graph_dataset.py
│   └── msr_loader.py
├── models/
│   ├── rgat.py
│   ├── cross_attention.py
│   └── classifier.py
├── explain/
│   ├── risk_score.py
│   ├── weighted_mcs.py
│   └── perturbation.py
├── metrics/
│   ├── hitk.py
│   ├── ndcg.py
│   └── recall_k.py
├── train_file.py
├── eval_file.py
├── eval_line.py
└── utils/
    ├── logger.py
    └── seed.py

---

## 2. 当前实现状态
- Dataset：❌ 待实现 / 重构
- File-level model：❌
- Line-level explainer：❌
- Metrics：❌

---

## 3. 代码风格与约束
- 使用 PyTorch / PyTorch Geometric（如需要）
- 强调模块化与可消融
- 每个模块需有清晰输入输出
- 实验结果需保存为 CSV / JSON

---

## 4. Codex 工作指令
- 严格按照本文件结构写代码
- 不自行扩展实验目标
- 不改论文指标定义
- 若有不确定之处，返回 TODO 注释

---

## 5. 给 Codex 的提示模板
你正在为一个硕士论文实验实现科研代码。
请严格按照本文件描述的项目结构与目标实现模块，
不引入额外假设，不自行修改评价指标。
