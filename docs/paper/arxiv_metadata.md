# arXiv 投稿元数据（对齐 WorldCup Predict submission 8087221 的字段与教训）

## Metadata 各字段

- **Title**：Bake It or Retrieve It: Reproducing Engram-Style Parametric Memory
  and Benchmarking It Against Retrieval and Full-Context Baselines
  （与 PDF 首页一致；勿改标题除非同步改 tex）

- **Author(s)**：Zhanhui Kang（affiliation = TerseMind，邮箱 research@tersemind.ai）
  - ⚠ WorldCup 教训：不能只写机构名（会被 arXiv Support 以"无署名/匿名投稿"退回）；
    **PDF 首页作者必须与 metadata 一致**。当前 main.pdf 首页作者是 "Agent Laboratory"
    ----补 Part 4 时需把 tex 作者改为 Zhanhui Kang / TerseMind / research@tersemind.ai
    并重编译，否则照 WorldCup 的经验会被退回。

- **Abstract**（≤1920 字符，框架 = 背景 → 系统 → 审计设定 → headline findings → 开源收尾）：

  ```
  Retrieval-based memory systems (RAG, Mem0, Zep) pay latency, context-window,
  and leakage costs at serving time; meanwhile "parametric memory" vendors
  (Engram, Cartridges) that bake knowledge into weights publish claims without
  open, comparable evidence. We build mini-engram, a minimal fully-open reproduction of
  the Engram-style pipeline--corpus ingestion, two-pass self-study QA synthesis,
  per-tenant LoRA baking (r=16), single-process multi-adapter serving--on
  Qwen2.5-7B-Instruct--benchmarked against closed-book, full-context, and
  BM25-RAG baselines under one protocol. Controlled synthetic wikis with facts the
  base model cannot know show closed-book LoRA recall at 1.9x the full-context
  upper bound (0.67 vs 0.35 token-F1), robust to paraphrase. On LoCoMo (10
  conversations, 1,986 official QA), LoRA wins every factual category (single-hop
  0.304 vs 0.222 full-context; 0.127 RAG) yet scores exactly 0.000 on adversarial
  questions where retrieval reaches 0.895: baked
  knowledge never refuses. Excluding adversarial items LoRA leads (0.234 vs 0.170
  vs 0.093). On LongHealth (20 fictional patients, 400 five-option
  MC questions, the Cartridges-family yardstick) the ordering reverses: full
  context 0.625 beats LoRA 0.355, below the base model--minimal QA-SFT
  fails to preserve fine-grained clinical detail. Together these
  results delimit where parametric memory wins--stable atomic facts answered
  closed-book--and where reading the document wins--fine-grained document-level
  discrimination--turning vendor claims into a measurable engineering trade-off.
  All code, adapters, and result files are released.
  ```

  （已含 LME 试点句；实测 1914 字符，≤1920 达标 ✅）

- **Comments**：`12 pages, 6 figures. Code and data released under MIT License, Copyright (c) 2026 TerseMind.`
  （v0.1.1 版 PDF 实测 12 页、6 图，已复核）

- **Subjects**：Primary = cs.AI；Secondary = cs.CL（备选讨论过 cs.LG）
- **Keywords**（建议）：parametric memory, LoRA, long-term conversational memory,
  retrieval-augmented generation, LoCoMo, LongHealth, multi-tenant
  serving, reproducibility
- **Report number / Journal reference / External DOI / ACM class / MSC class**：全部留空
  （未正式发表，非期刊）----与 WorldCup 一致

## 文件包（submission 打包清单，对齐 WorldCup 的可编译要求）

- tex 源目录：`AgentLaboratory/MATH_research_dir/research_dir_0_lab_1/tex/`
  （main.tex + figs/*.pdf + IEEEtran.cls；thebibliography 内嵌，无需 .bbl）
- 打包命令：`cd <tex 目录> && zip -r ../engram_arxiv_submit.zip main.tex figs/*.pdf IEEEtran.cls`
- arXiv 要求：源码包须能在无网络环境编译通过（cls 自带 ✓）；图用 PDF ✓
- 提交前 checklist：
  1. PDF 首页作者 = Zhanhui Kang（对齐 metadata）
  2. Abstract ≤ 1920 字符（填入 LME 句后重数）
  3. Comments 里页数/图数与 PDF 实际一致
  
## WorldCup 流程备注（直接复用）

1. arXiv 账号沿用之前注册的（submission 8087221 同一账号可直接提交新文章）
2. submit → Start new submission → 填 metadata → 上传 zip → 系统在线编译验证 →
   Request endorsement 不需要（账号已有授权历史）
3. 提交后 1-2 个工作日出 announcement 编号
