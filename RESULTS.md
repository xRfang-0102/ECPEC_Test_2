# RESULTS — ECPEC 实验全记录（ECF 数据集）

统一设置：seed 42 · threshold 0.66（dev 扫描最优）· EMA 0.9（epoch 8 起）· warmup 2 · pair_rank 0.3 · pos_weight 2.5 · λ_emotion 0.2 / λ_cause 0.4。Pair F1 为任务主指标。

## 1. 主结果（最优模型：v3a LoRA-ECPEC）

| 划分 | Pair P | Pair R | Pair F1 | AUPRC | Emotion F1 | Cause F1 |
|---|---|---|---|---|---|---|
| dev | 0.5742 | 0.5907 | **0.5824** | 0.5946 | 0.8025 | 0.7507 |
| test | 0.5254 | 0.5717 | **0.5476** | 0.5430 | 0.7656 | 0.7071 |

## 2. 全部实验对比（dev，Pair F1 / AUPRC）

| # | 模型 | F1 | AUPRC | 备注 |
|---|---|---|---|---|
| B1 | 冻结 [CLS] 特征基线 | 0.5534 | 0.5365 | 此前最优冻结配置 |
| R1 | 冻结 + 距离加权 | 0.5086 | 0.4815 | 失败 |
| R2 | 冻结 + 容量 4 层 | 0.5358 | 0.5160 | 失败 |
| R3 | 冻结 + 节点门控 | 0.5491 | 0.5363 | 中性 |
| A1 | A+C v1（冻结） | 0.5013 | 0.4682 | 失败 |
| B2 | 冻结 mean 特征对照 | 0.5446 | 0.5473 | v2 对照组 |
| A2 | A+C v2（冻结 mean） | 0.5184 | 0.5019 | 远距召回↑ |
| **v3a** | **LoRA 微调（ctrl）** | **0.5824** | **0.5946** | **最优** |
| v3b | LoRA + A+C | 0.5516 | 0.5532 | 远距召回翻倍 |
| v4 | 全量微调 | 0.5320 | 0.5336 | 小数据过拟合 |

## 3. 远距盲区诊断（dev，threshold 0.66 下的召回率）

| 模型 | d==0 F1 | d==1 F1 | d==2 R | d≥3 R | d≤-1 R |
|---|---|---|---|---|---|
| 冻结 [CLS] 基线 | 0.664 | 0.542 | 4.0% | **0%** | 0% |
| 冻结 mean 对照 | 0.664 | 0.542 | 4.0% | 0% | 0% |
| A+C v2（冻结） | 0.650 | 0.478 | 1.3% | 3.5% | 0% |
| **v3a LoRA** | **0.703** | **0.574** | **17.3%** | 3.5% | 0% |
| v3b LoRA+A+C | 0.679 | 0.533 | 14.7% | **7.0%** | 4.2% |

- dev 共 161 个 d≥2 对（75 d==2 + 86 d≥3），约占总正例 19%。
- 决策侧天花板（冻结特征）：per-bucket 阈值校准 0.556；强行 top-k 召回远距对时新增预测精度仅 ~9%，F1 崩至 0.42 —— 证明瓶颈在表征而非解码。
- LoRA 后 bucket 校准不再有增益（0.571 < 0.582）：微调同时修复了表征与校准两层。

## 4. A+C 检索注意力诊断

| 骨干 | 注意力聚焦真因 | 均匀基线 | top-1（attention / content） |
|---|---|---|---|
| 冻结 mean（A+C v2） | 0.60（2.17×） | 0.28 | 65% / 58% |
| LoRA（v3b） | 0.60（2.17×） | 0.28 | 62% / **64%** |

结论：检索注意力始终聚焦真因，但「content 排名 vs 检索排名」的对比随编码器可学习性反转 —— 检索是表征不足时的补偿机制（能力互补性转移）。

## 5. 论文叙事（现状）

- **痛点**：ECPEC 的远距跨说话人对（d≥2，约 19% 正例）是全场盲区；根因 = 表征缺乏远距证据 + 绝对校准被近距对主导（locality shortcut 的深层形态）。
- **Novelty**：① A+C 架构（约束式跨说话人历史事件检索 + 位置捷径因果干预，可解释注意力）；② 能力互补性转移的实证发现；③ 远距盲区诊断工具链（可容许集排名 / bucket 校准天花板 / rank-decoding 扫描）；④ LoRA ≥ 全量微调的小数据结论。
- **诚实差距**：test 0.5476 低于 SOTA（SCALE 57.7，口径待核实；MultiCauseNet ~55.12）。当前定位建议 Findings / workshop；冲主会需补「test ≥ 56」+ 3 种子方差报告 + 与现有模型的直接对照复现。

## 6. 投稿前待办

1. 核实 SCALE 57.7 的口径（dev 还是 test）并抄录其完整结果表。
2. seed-2 / seed-3 复跑（v3a），报告 mean±std。
3. 可选：RoBERTa-large + LoRA（预期 +1~2 F1）；LLM 远距难例增强（预期补足 novelty）。
4. test 侧阈值扫描（当前 0.66 为 dev 调出）。

## 7. 复现命令

```bash
python train_lora.py --config config/config_lora_ctrl.yaml
python eval_lora.py checkpoints/base_lora_ctrl_best.pt config/config_lora_ctrl.yaml
python v3_analysis.py checkpoints/base_lora_ctrl_best.pt config/config_lora_ctrl.yaml dev
```
