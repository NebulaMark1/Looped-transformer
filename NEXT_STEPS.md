# Next Steps — 2026-06-02

## 当前状态

- **最佳模型**: `delta_ffn_fineweb_epoch4.pt` (d=768, 8层, L=8, FineWeb 4 epoch)
  - FineWeb val PPL: 62.9
  - 约 98M 总参数, 59M transformer, 2.4M delta
- **baseline 模型**: `delta_ffn_loop8_wt103_final_best.pt` (d=768, 8层, L=8, WT-103 8 epoch)
  - WT-2 PPL: 47

---

## 完整 CL 实验（含 FineWeb 遗忘）

### 1. 保存 CL 模型副本

```bash
# cl_experiment.py 已改：结束前自动保存模型 pt
# 跑前先确认代码是最新的：
git pull
grep "cl_.*_model.pt" cl_experiment.py  # 应搜到新加的 save 行
```

### 2. 跑两个 CL 策略

```bash
CUDA_VISIBLE_DEVICES=2 nohup python cl_experiment.py \
  --checkpoint results/delta_ffn_fineweb_epoch4.pt \
  --strategy full --epochs 1 --output_dir ./results \
  > cl_fw_full.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python cl_experiment.py \
  --checkpoint results/delta_ffn_fineweb_epoch4.pt \
  --strategy delta_only --epochs 1 --output_dir ./results \
  > cl_fw_delta.log 2>&1 &
```

~20 分钟跑完。

### 3. 测 FineWeb 遗忘

```bash
python -c "
import torch, math, torch.nn.functional as F
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from delta_model import DeltaConfig, DeltaLoopedTransformer

def eval_fineweb(path):
    sd = torch.load(path, map_location='cpu')
    cfg = DeltaConfig(max_seq_len=256, embed_dim=768, num_heads=12, num_layers=8, num_loops=8)
    m = DeltaLoopedTransformer(cfg).cuda().eval()
    m.load_state_dict(sd, strict=True)
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    ds = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
    tokens = []; n = 0
    for item in ds:
        t = item['text'].strip()
        if t: tokens.extend(tok.encode(t))
        n += len(t.split())
        if n >= 3_000_000: break
    total_loss, total_tokens = 0.0, 0
    for i in range(0, len(tokens)-256, 256):
        c = tokens[i:i+257]
        if len(c) < 257: continue
        ids = torch.tensor([c[:256]]).cuda()
        lb = torch.tensor(c[1:257]).cuda()
        o = m(ids)
        total_loss += F.cross_entropy(o['logits'][0,:-1,:].float(), lb[:-1], reduction='sum').item()
        total_tokens += 255
    return math.exp(total_loss / total_tokens)

for label, p in [('base', 'results/delta_ffn_fineweb_epoch4.pt'),
                  ('full', 'results/cl_full_L8_model.pt'),
                  ('delta', 'results/cl_delta_only_L8_model.pt')]:
    print(f'{label}: FineWeb PPL = {eval_fineweb(p):.1f}')
"
```

### 4. 最终结果表

| Strategy | WT-2 Forgetting | TS New | FineWeb Forgetting |
|----------|----------------|--------|-------------------|
| full | +288% | 7.3 | ??? |
| delta_only | +54% | 12.8 | ??? |

---

## 如果需要续训 FineWeb

修复了 streaming 的 total_steps off-by-one bug。操作：

```bash
# 1. 确认代码最新
git pull

# 2. 训练（epochs 一次设够，不要中途改）
CUDA_VISIBLE_DEVICES=6 nohup python train_delta.py --delta_type ffn \
  --embed_dim 768 --num_heads 12 --num_layers 8 --num_loops 8 \
  --batch_size 16 --dataset fineweb --epochs 8 \
  --tag final_v2 --output_dir ./results \
  > fw8.log 2> fw8.err &
```

---

## 常用速查

| 需求 | 命令 |
|------|------|
| 看训练进度 | `tail -1 fw8.err` |
| 看 epoch 完成 | `tail -3 fw8.log` |
| 提取最新权重 | `python -c "import torch; ckpt=torch.load('results/xxx_resume.pt'); torch.save(ckpt['model'], 'results/extracted.pt')"` |
| 生成文本 | `python generate.py --checkpoint results/xxx.pt --prompt "..."` |
| 跨数据集 PPL | `python eval_ppl.py --checkpoint results/xxx.pt` |
| 看所有结果 | `python analyze.py --results_dir ./results --detail` |

---

## 代码变更记录（0602）

- `cl_experiment.py`: 结束前保存 `cl_{strategy}_L{loops}_model.pt`
- `train_delta.py`: streaming total_steps 加安全 margin
- `train_delta.py`: resume 时更新 scheduler.total_steps
- `WORKFLOW.md`: 新增踩坑指南
