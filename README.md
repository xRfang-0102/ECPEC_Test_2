# ECPEC_Test_2 — Emotion-Cause Pair Extraction in Conversations (ECF)

会话级情绪-原因对抽取（ECPEC）实验代码库，数据集为 **ECF**（Emotion-Cause Friends）。

## 最优模型：v3a LoRA-ECPEC

- **架构**：RoBERTa-base（LoRA 微调，r=16，仅 Q/V 投影 + 对话 Transformer + 情绪/原因双头 + 对分类器）
- **可训练参数**：2.79M（编码器本体 110M 冻结）
- **主结果**（seed 42，threshold 0.66，EMA）：

| 划分 | Pair P | Pair R | Pair F1 | AUPRC | Emotion F1 | Cause F1 |
|---|---|---|---|---|---|---|
| dev | 0.5742 | 0.5907 | **0.5824** | 0.5946 | 0.8025 | 0.7507 |
| test | 0.5254 | 0.5717 | **0.5476** | 0.5430 | 0.7656 | 0.7071 |

- **远距盲区修复**：d==2 对召回 17.3%（冻结特征基线 4.0%），d≥3 对 3.5%（基线 0%）
- 检查点：`checkpoints/base_lora_ctrl_best.pt`，配置：`config/config_lora_ctrl.yaml`

## 快速开始

```bash
# 1. 训练最优模型（LoRA 端到端，约 2 小时 / RTX 8GB）
python train_lora.py --config config/config_lora_ctrl.yaml

# 2. 评估 dev + test
python eval_lora.py checkpoints/base_lora_ctrl_best.pt config/config_lora_ctrl.yaml

# 3. 演示：预测单段对话的情绪-原因对
python predict_pairs.py --checkpoint checkpoints/base_lora_ctrl_best.pt \
    --config config/config_lora_ctrl.yaml --split dev --dialogue-id 0
```

冻结特征管线（旧基线）入口：

```bash
python extract_roberta_features.py --config config/config_feature_mean_ctrl.yaml  # 提取特征
python train_feature.py --config config/config_feature_mean_ctrl.yaml            # 冻结特征训练
```

## 目录

| 路径 | 说明 |
|---|---|
| `train_lora.py` | LoRA 端到端训练主脚本（含 bf16、梯度检查点、EMA、warmup） |
| `eval_lora.py` | 检查点评估（dev/test） |
| `models/lora.py` | 手写 LoRA 层（无 peft 依赖） |
| `models/lora_ecpec_model.py` | LoRA 编码器 + 对话头组合模型 |
| `models/feature_base_model.py` | 对话 Transformer + 对分类器 + **A+C 模块**（EventMemoryRetriever 跨说话人历史检索 + LocalityPriorHead 位置捷径干预） |
| `dataset/raw_ecf_dataset.py` | 原始对话 tokenize（批内动态 padding） |
| `train_feature.py` | 冻结特征训练 + 全部损失（BCE/排序/条件排序/检索监督） |
| `v3_analysis.py` / `scan_rank_decoding.py` / `diag_ac2_attention.py` | 远距盲区诊断与决策侧扫描 |
| `RESULTS.md` | 全部实验数据与论文素材 |

## 核心发现（论文素材）

1. **远距因果盲区**：冻结特征模型对 d≥3 的跨说话人对召回为 **0%**，且根因被拆解为「表征缺乏远距证据」与「绝对分数校准失效」两层（可容许集内 top-3 命中 91% 却全部被全局阈值 0.66 杀掉）。
2. **LoRA 端到端微调**是主杠杆：dev F1 0.5534 → 0.5824，近距远距全面增强，且 **LoRA ≥ 全量微调**（全量微调 dev 仅 0.5320，小数据过拟合）。
3. **A+C 机制**（约束式跨说话人历史事件检索 + 位置捷径因果干预）：冻结特征上检索注意力聚焦真因 2.17×、top-1 65%（content 仅 58%），并把 d≥3 召回从 0% 提到 3.5%；微调编码器上 content 信号反超检索（64% vs 62%）——检索的价值随编码器可学习性转移。

详见 [RESULTS.md](RESULTS.md)。
