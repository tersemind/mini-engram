# mini-engram 论文研究素材包（喂给 Agent Laboratory 的 task notes）

## 产物状态（2026-09-21）

- 论文 PDF：`/root/train-study/AgentLaboratory/MATH_research_dir/research_dir_0_lab_1/tex/main.pdf`（11 页，IEEEtran）
- LaTeX 源：同目录 `main.tex`（+ `figs/`、IEEEtran.cls）；原始草稿备份 `temp.tex.bak`
- 打包：`/root/train-study/engram_paper_deliverable.zip`
- 出稿管线：Agent Laboratory（deepseek-v3.2 via OpenRouter）→ 文献 6 篇 + 实验分析 + 草稿；
  review 阶段因 API 超时未走修订，数字审计/补图/引用/编译由人工 agent 按本文件锚点完成
- 补画图脚本：`mini-engram/scripts/plot_fig1_fig2.py`

## Research goal（一段话）

Reproduce and evaluate **Engram-style parametric memory** — baking long-term knowledge into
LoRA adapters via self-study synthetic data — as an alternative to retrieval-augmented memory
(Mem0/RAG) and full-context prompting for long-term conversational memory. We build a minimal
end-to-end pipeline (corpus → self-study QA synthesis → per-tenant LoRA → multi-adapter serving)
and benchmark it under controlled conditions (synthetic company wikis) and on the public
**LoCoMo** very-long-term conversational memory benchmark, comparing closed-book LoRA,
BM25-RAG, and full-context baselines with the official LoCoMo token-F1 protocol.

## 系统管线（method，照此写 Section 3）

1. **Corpus ingestion**: docs/chat histories → timestamped chunks (`engram/ingest.py`, `locomo_prep.py`).
2. **Self-study synthesis**: base model (Qwen2.5-7B-Instruct) generates QA pairs per chunk in two
   passes (general + coverage/drill), dialog-mode templates for chat corpora (`engram/synth.py`).
3. **Per-tenant LoRA baking**: PEFT LoRA (r=16, 7 target modules, QA-SFT, 6–20 epochs)
   (`engram/train_lora.py`).
4. **Multi-tenant serving**: one vLLM process hosts all adapters, routed by OpenAI-compatible
   model name (`scripts/serve.sh`).
5. **Competitor baselines**: closed-book (base), full context (stuff corpus), BM25 top-k RAG
   (`engram/rag.py`, `engram/bench.py`, `engram/locomo_bench.py`).
6. **Scoring**: token-level (multiset) F1 with Porter stemming; LoCoMo uses the official
   per-category protocol incl. adversarial refusal detection (`engram/locomo_score.py`).

## Experiment 1 — 合成公司 wiki（可控对照，已完成）

虚构公司语料（base model 绝不可能知道的事实），2 租户（xinglan 123 QA / hanhai 93 QA 合成训练集）。
条件 × 题集矩阵（token-F1）：

xinglan（n=18/18/13 per cell）:

| 题集 | base | full | rag3 | rag5 | lora:v1 | lora:v2(+拒答训练) |
|---|---|---|---|---|---|---|
| own  | 0.126 | 0.349 | 0.370 | 0.361 | **0.666** | 0.621 |
| para | 0.090 | 0.333 | 0.351 | 0.346 | **0.581** | 0.581 |
| cross| 0.109 | 0.022 | 0.022 | 0.024 | **0.363** | 0.363 |

hanhai（lora 仅 v1）: own 0.113/0.349/0.351/0.348/**0.664**; para 0.101/0.330/0.342/0.325/**0.592**;
cross 0.111/0.042/0.047/0.047/**0.324**。

要点：
- LoRA 烤入把闭卷 F1 从 ~0.12 提到 0.62–0.67，约为 full-context/RAG 上界的 **1.8–2×**。
- para（换问法）只掉 ~0.08 → 参数化记忆对复述鲁棒；rag/full 在 cross（别家题）上≈0.02–0.05，
  天然不会串味，而 LoRA 有"自信幻觉"（0.32–0.36 的 F1 是短答案精度偏好的指标假象 + 通用常识巧合，
  逐题核对无真实参数泄漏）→ 多租户需要拒答/路由校准（v2 尝试：own 略降、cross 未降）。
- 遗忘检查：6 组通用探针（代码/常识/写作/身份）+LoRA 与 base 一致，无人设泄漏。

## Experiment 2 — LoCoMo（公开基准，进行中，数字出来后填这里）

- 数据：locomo10.json，10 段两人对话（8k–16k 词 ≈ 12k–21k tokens），1986 官方 QA
  （single_hop 841 / adversarial 446 / temporal 321 / multi_hop 282 / open_domain 96）。
- 每段对话 = 一个租户：prep → dialog-mode synth（QA 训练集 ~500/对话）→ LoRA(6 epochs)
  → locomo_bench 四条件：base / full(32K 上下文) / rag5(BM25 top-5) / lora(闭卷)。
- 评分：`engram/locomo_score.py`（与 snap-research/locomo 官方 evaluation.py 逐条对齐：
  normalize → Porter stem → token 多重集 F1；multi-hop 逗号拆分子答案 max-then-mean；
  adversarial 命中 "no information available"/"not mentioned" 记 1）。
- 结果（2026-09-21 跑出，locomo_summary.json，n=1986）：
  - overall: base 0.039 / full 0.285 / rag5 0.273 / lora 0.181
  - by_category (base/full/rag5/lora): single_hop 0.059/0.222/0.127/**0.304**;
    multi_hop 0.055/0.161/0.071/**0.193**; temporal 0.012/0.071/0.035/**0.091**;
    open_domain 0.074/0.067/0.058/**0.213**; adversarial 0.002/0.684/**0.895**/0.000
  - 剔除 adversarial（n=1540）：lora **0.234** > full 0.170 > rag5 0.093 > base 0.048
  - 逐对话 F1 范围：lora 0.149-0.240，无 32K 截断（truncated={}）
- 结论：事实类目参数化记忆全面第一；adversarial 上 LoRA 100% 自信编造（字面 0 分）——
  "会的事最好、不会的事不会拒答"的参数化记忆固有代价，与合成 cross 实验互证；
  rag5 靠忠实指令 0.895 正确拒答但 multi_hop 仅 0.071（单 session 检索拼不出跨 session 链）。
- 预期讨论点：temporal 题依赖 chunk 内时间戳；adversarial 题上 LoRA 闭卷必然幻觉
  （参数化记忆的固有代价 → 拒答校准是未来工作）；与 Mem0 论文数字的可比性。

## Experiment 3 — LongHealth（Engram 家族自己的尺子，进行中）

- 数据：kbressem/LongHealth `benchmark_v5.json`：20 个虚构病人 × 20 道五选一医学选择题（400 题），
  每病人 1-3 封转诊/随访信（5-7K 词）；canary-string 防污染。这是 Hazy Research Cartridges
  官方演示同款闭卷基准（其 0.96GB cartridge 闭卷 55.1% vs 全量 ICL）。
- 每病人 = 一个租户：prep（900 词切片）→ synth（company=病人名，qa-per-chunk 8）→
  LoRA（20 epochs，~40M 参数）→ longhealth_bench 四条件：base / full(8K) / rag5 / lora(闭卷)。
- 评分：MC 字母抽取准确率（A-E），抽不出回退 correct 文本包含匹配。
- 结果（2026-09-21 跑出，longhealth_summary.json，n=400 MC）：
  - overall: base 0.398 / full 0.625 / rag5 0.560 / lora 0.355（lora 低于 base！）
  - by_patient 见 JSON；lora null-letter 0/400 → 字母抽取不是败因，错答全是细粒度知识错误
    （剂量数字、方案组成、NOT-part-of 否定题）
- 结论：与 wiki/LoCoMo 相反——细粒度判别型任务上朴素 QA-SFT 烤入会扰动 base 医学推理
  却没换来精确记忆；全文阅读理解（0.625）才是上限。对照 Cartridges 55.1%（0.96GB
  持续预训练 cartridge vs 我们 ~40M 参数最小 SFT）：差距=训练配方深度。
- 论文定位：**任务依赖性**是核心发现——闭卷原子事实回忆参数化记忆占优；
  文档级细粒度理解读原文更好。
- 意义：用"竞品家族自己的尺子"验证 LoRA 烤入在闭卷选择题上的等效性——我们的 adapter 只有
  ~40M 参数，对标的却是对方 0.96GB 的 cartridge。

## 竞品基准全景（2026-09-20 调研结论，支撑 related work 与 future work）

- **Mem0**（arXiv:2504.19413）：论文用 LoCoMo（LLM-as-judge，+26% vs OpenAI 记忆）；官方三件套
  LoCoMo / LongMemEval / BEAM，开源评分框架在 mem0ai/memory-benchmarks。注意 Mem0 的 LoCoMo
  是 judge 打分，与我们官方 token-F1 不可直接比。
- **Zep**：DMR（MemGPT 提出，MSC 500 对话）已被 Zep 自己承认过时（60 条消息全塞 context 就 94-98%）；
  现公开成绩跑 LoCoMo(1540 题)+LongMemEval(500 题)。
- **Letta/MemGPT**：Letta Leaderboard 面向 LLM 选型；学术侧 MemBench 他们并未采用。
- **Engram/Cartridges 家族**：从不刷聊天记忆榜；官方尺子 = LongHealth / MTOB / 合成金融文档闭卷 QA
  （"Stuffing MLPs Full of Facts" 配方文）。
- **LongMemEval**（hf `xiaowu0162/longmemeval-cleaned`，hf-mirror 可达）：haystack 随包发布
  （turn 级含 has_answer），500 题 S/M 两档（S ~115K tokens/样本）。我们计划抽 50 对话做
  mini-LongMemEval（全量 500 租户 × synth 在 A10 上单机不可行）。
- 接入优先级：LongMemEval-S-50 > BEAM-128K 子集 > DMR（已饱和，仅作对照）。

## Key findings（论文卖点）

1. **任务依赖性**（三研究合成后的主结论）：闭卷原子事实回忆（合成 wiki own/para、LoCoMo 事实类目）参数化记忆显著占优（~2× 于检索）；文档级细粒度判别（LongHealth MC）读原文更好，朴素烤入甚至低于 base。
2. 参数化记忆的两大代价：串味/幻觉（wiki cross、LoCoMo adversarial 字面 0 分）与增量更新（语料变只能重烤）。
3. 配方深度决定上限：同一把尺子（LongHealth）上，Cartridges 持续预训练级 cartridge 55.1% vs 我们最小 SFT 0.355——差距不在概念而在训练配方。
4. 工程可复制性：单卡 A10 24GB、7B base、分钟级/租户训练，一个 vLLM 进程托管全部 adapter。

## Limitations / threats to validity

- 7B 单模型、temperature 0、短答案 F1 偏好精确匹配；open_domain 只取 ';' 首段。
- 自产训练集与评测分布同源（self-study），存在"考官即考生"偏差——用 LoCoMo 官方 QA 作
   held-out 评测缓解。
- LoCoMo full 条件 32K 截断策略：保时间序头部 session；adversarial 评分依赖英文拒答短语。

## Related work 锚点（让 AgentLab 的文献 agent 核实并补全）

- LoCoMo: "Evaluating Very Long-Term Conversational Memory of LLM Agents" (arXiv:2402.17753)
- Mem0: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory" (arXiv:2504.19413)
- LongMemEval (arXiv:2410.10813) — 后续工作
- LoRA (arXiv:2106.09685), PEFT (arXiv:2110.04366)
- Qwen2.5 (arXiv:2412.15115); vLLM (arXiv:2309.06180)
- HazyResearch: Engram (产品/技术博客), cartridges (self-study + context distillation, GitHub)
- Sakana AI: doc-to-lora (hypernetwork 直出 LoRA)
- Zep / Letta-MemGPT 的 DMR 基准（视调研结果引用）

## 数据/代码可用性

- 代码：`mini-engram/`（engram/ 包 + scripts/）；数据产物：`mini-engram/data/`
  （corpus/ dataset/ adapters/ bench/ locomo/）；日志：`logs/engram_*.log`。
