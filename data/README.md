# data/ 目录说明（对外发布）

本目录只提交**我们产出的评测结果**；语料与 adapter 体积大/含第三方数据派生物，不入库。

## 提交内容

- `bench/summary_*.json`、`bench/locomo_summary.json`、`bench/longhealth_summary.json`：
  各研究汇总（数字与论文一致）
- `bench/figs/`：论文全部图（pdf+png）
- 合成语料逐题明细（`bench/xinglan__*.json` / `bench/hanhai__*.json`，无第三方权利）
- LongHealth 逐题明细（Apache 2.0，宽松许可，随公开 bundle）
- **LoCoMo 逐题明细（含 gold）不公开**：CC BY-NC 4.0，按业界 "code in, data out"
  惯例留在研究归档（`engram_research_archive/`），on request 提供——与 Mem0
  memory-benchmarks 的处理一致（其 benchmarks/locomo/ 只有代码零数据）

## 第三方基准数据获取（复现时自行下载）

```bash
# LoCoMo（snap-research/locomo，~2.8MB）
mkdir -p data/locomo
curl -sL -o data/locomo/locomo10.json <locomo10.json 的官方地址，见 github.com/snap-research/locomo>

# LongHealth（kbressem/LongHealth）
mkdir -p data/longhealth && cd data/longhealth
curl -sL -o repo.zip https://github.com/kbressem/LongHealth/archive/refs/heads/main.zip
unzip -q repo.zip   # 基准文件在 LongHealth-main/data/benchmark_v5.json

```

两个基准的数据使用条款归各自仓库所有；本仓库只含派生分析结果与脚本。

## adapter（LoRA 权重）

- 演示租户 adapter（星澜科技/瀚海机器人，虚构语料）：发布包内 `adapters_bf16/`
  （~77MB/个，bf16），挂到 Qwen/Qwen2.5-7B-Instruct 即可复现多租户托管演示
- 基准研究 adapter（locomo_* ×10 / lh_* ×20 fp32）：
  仅随完整研究归档分发，**标注"训练自基准数据，仅限复现研究，不得用于基准计分"**
- adapter 为 Qwen2.5-7B-Instruct（Apache 2.0）的派生物，再分发需附 Apache 2.0 声明

## LongMemEval-S (v0.1.1, Study 4)

- Download `longmemeval_s_cleaned.json` from the official repo
  (https://github.com/xiaowu0162/longmemeval, LongMemEval_S set) into
  `data/longmemeval/`
- Then run `bash scripts/run_longmemeval.sh` (prep → bake 50 tenants → bench → judge;
  judge needs `OPENROUTER_API_KEY` and calls deepseek-v3.2, 3-vote majority)
- Aggregate result numbers (no gold answers) are committed at
  `data/bench/lme_summary.json`; per-question rows are NOT redistributed
