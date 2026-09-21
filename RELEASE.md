# 对外发布清单（Release Plan）

发布形态 = **论文（arXiv）+ 代码仓（GitHub）+ 演示 adapter（HF Hub）+ 研究归档包**。

## 1. 代码仓（GitHub: tersemind/mini-engram）

- [x] LICENSE（MIT, Copyright (c) 2026 TerseMind）
- [x] .gitignore：排除 .venv / release/ / 语料 / adapter / 逐题明细（含基准 gold）
- [x] 入库：全部代码（engram/ + scripts/）、docs/paper/、data/bench 汇总+图、data/README.md
- [x] 无内嵌密钥（judge/review 脚本一律走 OPENROUTER_API_KEY 环境变量）
- [ ] git init + 首次 commit（已完成本地，见下）
- [ ] push 到远端：`git remote add origin git@github.com:tersemind/mini-engram.git && git push -u origin main`（需 GitHub 凭据）
- [ ] 打 tag `v0.1.0` + GitHub Release 附 engram_release_bundle.zip

## 2. 演示 adapter（HF Hub: tersemind/mini-engram-xinglan / -hanhai）

- [x] bf16 转换（80.8MB/个，fp32 减半）→ `release/adapters/`
- [x] 模型卡 `release/adapters/README.md`（使用方法、数字、Apache 2.0 派生声明）
- [ ] 上传：`huggingface-cli upload tersemind/mini-engram-xinglan release/adapters/xinglan .`（需 HF token）
- [ ] 验证：上传后 `vllm serve` 冒烟一遍（README 的 curl 示例应答对"算盘"）

## 3. 论文（arXiv）

- [x] main.tex/main.pdf（IEEEtran，11 页 6 图；LME 合入后 4 研究）
- [x] 元数据表 `docs/paper/arxiv_metadata.md`（对齐 WorldCup 8087221 字段）
- [ ] 作者改 Zhanhui Kang（PDF 首页 = metadata，WorldCup 教训）
- [ ] 上传 engram_arxiv_submit.zip（tex + figs + cls，可离线编译）

## 4. 研究归档（不公开，on request）

- [x] 归档目录：`/root/train-study/engram_research_archive/`（含 README 分发规则）
- 内容：LoCoMo 逐题明细（含 gold）+ LoCoMo 派生训练集/adapter（CC BY-NC 4.0，业界
  "code in, data out" 惯例不公开，与 Mem0 memory-benchmarks 同姿势）
- LongHealth（Apache 2.0）派生物随公开 bundle（宽松许可）
- [ ] 可选：Zenodo 受限访问归档全量 85 个研究 adapter（~6.8GB fp32 / ~3.4GB bf16）拿 DOI

## 5. 验收标准

- [ ] 干净环境 clone 后：`bash scripts/setup_env.sh && bash scripts/run_demo.sh` 全链路可跑
- [ ] 复现三基准：data/README.md 的下载命令 + `run_locomo.sh` / `run_longhealth.sh` /
  `run_longmemeval.sh`（结果数字与 data/bench 汇总一致）
- [ ] 发布物中 grep 不到任何密钥；逐题明细只出现在带警示的归档包
- [ ] 论文 PDF 页数/图数与 arXiv Comments 一致；摘要 ≤1920 字符

## 产物索引（本机）

| 物 | 路径 |
|---|---|
| 代码仓（已 init+commit） | `/root/train-study/mini-engram/` |
| 演示 adapter bf16 | `mini-engram/release/adapters/{xinglan,hanhai}/` |
| 论文 | `AgentLaboratory/MATH_research_dir/research_dir_0_lab_1/tex/main.{tex,pdf}` |
| 论文交付包 | `/root/train-study/engram_paper_deliverable.zip` |
| 发布总包 | `/root/train-study/engram_release_bundle.zip` |
| arXiv 元数据 | `mini-engram/docs/paper/arxiv_metadata.md` |
