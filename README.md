# Looped Transformer with Per-Loop LoRA

验证假设：在循环Transformer中为每个循环轮次添加独立的低秩适配器（LoRA），能在少量额外参数下提升模型效果。

## 三种配置

| 配置 | 说明 |
|------|------|
| `baseline` | 传统循环Transformer，所有轮次完全共享参数 |
| `lora` | 共享基础参数 + 每轮独立的低秩适配器 B_t @ A_t |
| `full` | 每轮完全独立的参数（理论上界参考） |

## 环境准备

```bash
pip install torch transformers datasets tqdm
```

## 运行实验

### 方式一：一键运行所有实验

```bash
bash run_all.sh
```

### 方式二：单独运行

```bash
# Baseline
python train.py --mode baseline --epochs 15 --output_dir ./results

# LoRA-per-Loop
python train.py --mode lora --lora_rank 8 --epochs 15 --output_dir ./results

# Full-Loop (独立参数)
python train.py --mode full --epochs 15 --output_dir ./results
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | baseline | baseline / lora / full |
| `--lora_rank` | 8 | LoRA秩 |
| `--embed_dim` | 384 | 嵌入维度 |
| `--num_layers` | 3 | Transformer块数 |
| `--num_loops` | 4 | 每块的循环次数 |
| `--num_heads` | 6 | 注意力头数 |
| `--epochs` | 15 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--seq_len` | 256 | 序列长度 |
| `--lr` | 3e-4 | 学习率 |
| `--device` | cuda | 设备（自动fallback到CPU） |

## 查看结果

```bash
# 概览对比
python analyze.py --results_dir ./results

# 详细的每轮指标
python analyze.py --results_dir ./results --detail
```

## 输出文件

```
results/
├── baseline_results.json    # Baseline 训练指标
├── lora_r8_results.json     # LoRA-per-Loop 训练指标
├── full_results.json        # Full-Loop 训练指标
├── baseline_best.pt         # 最佳模型权重
├── lora_best.pt
└── full_best.pt
```

## 预期结果解读

- **Baseline**: 参数量最少，作为基准
- **LoRA-per-Loop**: 参数量略多于baseline（增加约 `num_layers * num_loops * 6 * rank * 2 * embed_dim`），预期困惑度低于baseline
- **Full-Loop**: 参数量约为baseline的 `num_loops` 倍，预期困惑度最低（上界）
- 如果 LoRA-per-Loop 的困惑度接近 Full-Loop 但参数量远小于 Full-Loop，则假设成立

## 模型结构

```
Input
  → Token Embedding + Position Embedding
  → [Block 1 × num_loops 次]   ← 每轮可注入独立LoRA
  → [Block 2 × num_loops 次]
  → [Block 3 × num_loops 次]
  → LayerNorm → LM Head
  → Loss (cross-entropy)
```

每个Block = Pre-LN Attention + Pre-LN FFN（GPT-2风格）
