# 训练工作流 & 踩坑指南

## 快速启动

### 标准训练

```bash
# tmux 持久化（关电脑不断）
tmux new -s train_name

CUDA_VISIBLE_DEVICES=X python train_delta.py --delta_type ffn \
  --embed_dim 768 --num_heads 12 --num_layers 8 --num_loops 8 \
  --batch_size 8 --dataset wikitext-103-raw-v1 --epochs 5 \
  --tag your_tag --output_dir ./results

# Ctrl+B D 断开
# tmux attach -t train_name  连回来
```

### 后台 nohup

```bash
CUDA_VISIBLE_DEVICES=X nohup python train_delta.py ... \
  > output.log 2> output.err &

tail -f output.log     # epoch 完成
tail -1 output.err     # 当前进度
```

---

## 续训

### 规则：epoch 数必须一致

只有 `--epochs` 和原始训练相同才能精确续训。

```bash
# 从 epoch 5 crash 续练 3 epoch（总共还是 8）
python train_delta.py --epochs 8 --resume results/xxx_resume.pt ...

# 想从 8 扩到 12：epoch 9-12 会用恒定小 lr，和一次性跑不完全一样
python train_delta.py --epochs 12 --resume results/xxx_resume.pt ...
```

### 保存的文件

- `xxx_best.pt`：val PPL 最低时的模型权重
- `xxx_resume.pt`：完整训练状态（模型 + 优化器 + scheduler + epoch）——**最重要的续训文件**
- `xxx_results.json`：每 epoch 的 train/val 指标

### 如果想一次性训够

**直接设足够大的 epoch，不要中途续。**

---

## 数据集

| 参数 | 数据集 | tokens | 每 epoch 时间 (d=768) |
|------|--------|--------|---------------------|
| `wikitext-2-raw-v1` | WT-2 | ~2M | ~3min |
| `wikitext-103-raw-v1` | WT-103 | ~100M | ~1.5h |
| `fineweb` | FineWeb-Edu | 500M | ~4h |

### FineWeb 注意事项

- 流式加载，没有 `len()`，每 epoch 结束后 **重新下载数据** 而非复用
- 验证集取 5M tokens 分片，同分布
- 数据集路径：`HuggingFaceFW/fineweb-edu`
- **不要加** `trust_remote_code=True`（新版 datasets 不支持）

---

## 验证集陷阱

`train_delta.py` 默认 val 是 WT-2。训练时 val 和 train 不是同一个分布会导致 val 不降，checkpoint 不保存。

**规则：val 必须和 train 同一分布。** 当前代码已修复，`dataset=fineweb` 时 val 也取 FineWeb。

---

## GPU & 显存

```
d=384, batch=8, L=8:  ~6GB   (A800 13%)
d=768, batch=8, L=8:  ~30GB  (A800 38%)
d=768, batch=16, L=8: ~50GB  (A800 63%)
```

8 张 A800 用 `CUDA_VISIBLE_DEVICES=X` 选卡，**不需要** DDP 多卡训练（模型太小，通信开销 > 收益）。

---

## 文件命名约定

```
delta_ffn_loop8_wt103_final_best.pt    ← 最佳权重
delta_ffn_loop8_wt103_final_resume.pt  ← 完整训练状态
delta_ffn_loop8_wt103_final_results.json ← 训练指标

--tag your_name    ← 避免覆盖旧文件（必须用）
```

---

## 频繁遇到的 bug

| 问题 | 原因 | 解决 |
|------|------|------|
| `total_steps` 溢出 | OneCycleLR 不能扩 epoch | 续训 epoch 不要改，或者接受恒定小 lr |
| resume 后 lr 不对 | scheduler 恢复了旧 total_steps | 已修复：扩 epoch 时自动切 ConstantLR |
| checkpoint 不保存 | val 下降条件不满足 | 跨分布训练时改保存条件，或每 epoch 存 `_epochN.pt` |
| `FineWebStream has no len()` | 流式数据集 | 已修复：try/except 手动算 total_steps |
| `FileNotFoundError` | 文件名和 `--tag` 不对 | 先 `ls results/` 确认实际文件名 |
| WT-2 val 上升 | val 集和训练集不同分布 | 正常现象，不需要停训 |

---

## 常用脚本

```bash
# 查看所有实验结果
python analyze.py --results_dir ./results --detail

# 生成文本
python generate.py --checkpoint results/xxx_best.pt --prompt "..."

# 跨数据集 PPL 评测
python eval_ppl.py --checkpoint results/xxx_best.pt

# 敏感度分析
python sensitivity.py --checkpoint results/xxx_best.pt
python loop_drift.py --checkpoint results/xxx_best.pt

# 持续学习实验
python cl_experiment.py --checkpoint results/xxx_best.pt --strategy full
python cl_experiment.py --checkpoint results/xxx_best.pt --strategy protect
python cl_experiment.py --checkpoint results/xxx_best.pt --strategy delta_only
```
