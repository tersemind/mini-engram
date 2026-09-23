# mini-engram

A minimal end-to-end reproduction of "baking knowledge into model parameters" — the product
pipeline behind Engram (the Hazy Research family):
**data source → self-study synthesis of a training set → per-tenant LoRA adapter baking →
one base model hosting many adapters**.

Unlike retrieval-style memory (Mem0/RAG), knowledge lives in LoRA parameters: no retrieval
at inference time, no documents stuffed into the context.

## Pipeline

```
scripts/make_demo_corpus.py   fictional company wiki ("Xinglan Tech") — facts the base model cannot possibly know
engram/ingest.py              directory/git repo → chunks.jsonl
engram/synth.py               self-study: the base model quizzes itself on the corpus → QA training set
engram/train_lora.py          PEFT LoRA fine-tuning → data/adapters/<tenant>/
engram/eval.py                same question set: closed-book (no adapter) vs baked (with adapter) F1
scripts/serve.sh              vLLM multi-LoRA: one base hosts all tenant adapters, OpenAI-compatible API
scripts/chat.py               demo call: ask the same question to base and to a tenant adapter
```

## Quick start (runs on a single A10 24GB)

```bash
bash scripts/setup_env.sh        # venv + dependencies (~10 min first time)
bash scripts/run_demo.sh         # end-to-end demo (auto-downloads Qwen2.5-7B-Instruct on first run)
```

Run on your own data:

```bash
.venv/bin/python -m engram.ingest --tenant myteam --source /path/to/docs     # or a git repo URL
.venv/bin/python -m engram.synth  --tenant myteam
.venv/bin/python -m engram.train_lora --tenant myteam
.venv/bin/python -m engram.eval   --tenant myteam
```

Multi-tenant serving:

```bash
bash scripts/serve.sh                       # auto-mounts every adapter under data/adapters/
.venv/bin/python scripts/chat.py --tenant myteam "What is your reimbursement system called?"
```

## Configuration

- `MINI_ENGRAM_BASE_MODEL`: switch the base model (default `Qwen/Qwen2.5-7B-Instruct`, downloaded via ModelScope)
- `--model` on synth/train/eval selects the generation/training model independently

## Initial experiments (2026-09-20, A10 24GB, Qwen2.5-7B-Instruct)

| Tenant | Synthetic QA | Training (20 epochs) | Closed-book F1 | Baked F1 |
|---|---|---|---|---|
| xinglan (Xinglan Tech) | 123 (two-pass synthesis) | 168s | 0.126 | **0.649** |
| hanhai (Hanhai Robotics) | 93 | 127s | 0.113 | **0.664** |

- A single-pass synthesis control (52 QA) only reaches 0.575, and facts never taught are
  inevitably fabricated → **data coverage is the core recipe ingredient**
- Contamination check (cross-tenant questions): F1 ~0.33, shown by per-question analysis to
  be a metric artifact (short-answer precision bias + generic workplace common sense), with no
  genuine parameter-level leakage; however, adapters do confidently fabricate answers to other
  tenants' questions (e.g., attributing their own facts to the other company) → multi-tenant
  products need refusal/routing calibration
- Forgetting check: 6 general probes (code/commonsense/writing/identity) — adapter matches
  base; no company persona leakage on identity probes
- Serving demo: one vLLM process hosts both tenant adapters, routed by model name over HTTP;
  questions the base answers with "I don't know" are answered accurately per tenant
  ("Abacus" / "Bailu Fresh")

## Competitor benchmark suite (2026-09-21, Qwen2.5-7B-Instruct, A10 24GB)

Uniform conditions: **base** (closed-book) / **full** (entire corpus stuffed into context) /
**rag5** (BM25 top-5, Mem0/RAG-style retrieval memory) / **lora** (per-tenant adapter baked,
then closed-book). Training recipe: two-pass self-study synthesis (dialog templates for chat
corpora; QA language matches corpus language) → LoRA r=16 QA-SFT.

### 1. Synthetic company wikis (controlled comparison, leakage-free by construction)

| Tenant | Quizset | base | full | rag5 | lora:v1 | lora:v2(+refusal) |
|---|---|---|---|---|---|---|
| xinglan | own | 0.126 | 0.349 | 0.361 | **0.666** | 0.621 |
| xinglan | para (paraphrase) | 0.090 | 0.333 | 0.346 | **0.581** | 0.581 |
| xinglan | cross (other tenant) | 0.109 | 0.022 | 0.024 | 0.363 | 0.363 |
| hanhai | own | 0.113 | 0.349 | 0.349 | **0.664** | — |
| hanhai | para | 0.101 | 0.330 | 0.325 | **0.592** | — |
| hanhai | cross | 0.111 | 0.042 | 0.047 | 0.324 | — |

LoRA is **1.8–2× the full/rag5 upper bound** on own/para and robust to paraphrase
(-0.06..0.09 absolute); the cost is confident fabrication on cross (0.32–0.36, shown by
per-question inspection to be a metric artifact plus generic common sense, no real leakage —
but multi-tenant products still need routing/refusal calibration).

### 2. LoCoMo public benchmark (10 real long conversations of 8k–16k words, 1,986 official questions, official token-F1)

| Condition | overall | single_hop | multi_hop | temporal | open_domain | adversarial |
|---|---|---|---|---|---|---|
| base | 0.039 | 0.059 | 0.055 | 0.012 | 0.074 | 0.002 |
| full (32K) | 0.285 | 0.222 | 0.161 | 0.071 | 0.067 | 0.684 |
| rag5 | 0.273 | 0.127 | 0.071 | 0.035 | 0.058 | **0.895** |
| lora | 0.181 | **0.304** | **0.193** | **0.091** | **0.213** | 0.000 |

- **What it knows: parametric memory wins.** LoRA leads all four factual categories
  (single_hop 0.304 vs full 0.222 vs rag5 0.127); on multi_hop, rag5 only reaches 0.071
  (BM25 fetches single sessions and cannot assemble cross-session chains).
- **What it doesn't know: parametric memory never refuses.** On adversarial questions
  (about things that never happened; 446/1986 = 22%), LoRA scores literally 0 — it
  fabricates 100% of the time, while rag5 with faithful instructions refuses correctly
  at 0.895. The overall gap (lora < full/rag5) is caused entirely by this column.
- **Excluding adversarial** (n=1540): lora **0.234** > full 0.170 > rag5 0.093 > base
  0.048 — parametric memory has by far the highest "known-knowledge density".
- Temporal is low for everyone (0.03–0.09): a 7B model struggles with date formats
  ("7 May 2023") under strict token-F1; lora still leads.
- Comparability note: Mem0's published LoCoMo numbers use LLM-as-judge scoring on a
  different base model — order-of-magnitude reference only, never a direct comparison;
  both Zep and Mem0 now report on this benchmark.

### 3. LongHealth (the Engram-family's own ruler: 20 fictional patients, 400 five-option MC questions, MC accuracy)

The same closed-book benchmark used in the official Cartridges demo (their 0.96GB
cartridge scores 55.1% closed-book vs full-context ICL). Our adapters are only ~40M
parameters, ~5 minutes per patient.

| Condition | overall | Note |
|---|---|---|
| base | 0.398 | fictional patients, but real medical common sense transfers |
| full (8K) | **0.625** | reading the letters — the comprehension ceiling |
| rag5 | 0.560 | 5 retrieved chunks help, but less than full text |
| lora | **0.355** | **below base** — fine-grained clinical detail was not baked in |

- **The reversed result is precisely the paper's central discussion point**: when the
  evaluation is fine-grained discrimination (dosage numbers, regimen composition,
  NOT-part-of negation), naive QA-SFT baking (no curriculum, no anti-forgetting mix,
  ~100 QA per patient) perturbs the base model's medical reasoning without buying precise
  recall; letter extraction is not the cause (lora null-letter 0/400 — every error is a
  knowledge error).
- Against Cartridges' 55.1%: the gap comes from training-recipe depth (continued-pretraining
  scale cartridge vs our minimal SFT) — showing the bake-in ceiling depends on recipe depth,
  an empirical footnote to the moat section below.
- Task-dependence conclusion: parametric memory clearly wins **closed-book atomic-fact
  recall** (wiki/LoCoMo short-answer F1); **document-level fine-grained understanding**
  (LongHealth MC) favors reading the original. Parametric memory is not a universal
  replacement for retrieval.

### 4. LongMemEval-S (50 questions, ~110K-token personal histories, official judge, v0.1.1)

The benchmark Mem0/Zep report official numbers on. Each question = one tenant with a
~110K-token user–assistant history — 4× our 24GB card's context, so "full" is a 32K
chronological head and the official oracle (evidence-only) marks the upper reference.
v0.1.1 adds **six-pass taxonomy-targeted synthesis** (cross-session stitching, temporal
normalization, preference extraction, update consolidation, numeric aggregation,
elapsed-time drills — all from the user's own history, never from eval questions) and
an optional **shared-skill adapter + exact additive LoRA merge**
(`engram/train_skill.py`, `engram/merge_adapters.py`).

| Condition | overall | knowledge-update | SS-user | SS-assistant |
|---|---|---|---|---|
| base | 0.120 | 0.000 | 0.286 | 0.286 |
| full (32K head) | 0.160 | 0.100 | 0.571 | 0.429 |
| rag5 | 0.200 | 0.400 | 0.143 | **0.714** |
| lora (generic synth, pilot n=15) | 0.133 | 0.333 | — | — |
| **lora (six-pass synth)** | **0.300** | **0.600** | **0.571** | 0.429 |
| lora + skill merge (λ=0.5) | 0.280 | 0.500 | 0.571 | 0.429 |
| *oracle (evidence-only, n=15)* | *0.467* | *0.833* | — | — |

- **0.300 = 2.5× base, +50% over rag5, +87.5% over truncated full-context** — same
  7B base, closed-book, 0 retrieval tokens. Recipe was designed on a 15-question pilot
  and frozen before the full run: the 35 held-out questions score 0.314.
- The gain concentrates on **knowledge-update** (0.600 vs rag5 0.400): consolidation-style
  QA teaches the adapter the *latest* value of superseded facts.
- **Negative result worth keeping**: additively merging a shared skill adapter
  (ΔW = ΔW_content + 0.5·ΔW_skill, exact LoRA concatenation) gives no reliable gain
  (0.280) — weight-space addition cannot substitute for Engram's architectural
  content/skill separation.
- Retrieval's last bastion: single-session-assistant questions (0.714) — verbatim
  assistant statements favor lexical matching (a routing point).
- Judge note: deepseek-v3.2, 3-vote majority, official templates; it declines to apply
  the official off-by-one exemption, so temporal numbers are conservative.

## What this reproduction deliberately simplifies (= Engram's real moat)

- **Training recipe**: here it is the most naive QA SFT; no curriculum learning, no
  general-data mixing against forgetting
- **Forgetting control**: no monitoring of general-capability or old-knowledge regression
  after baking
- **Incremental updates**: a changed corpus requires a full re-bake; no incremental-adapter
  mechanism
- **Evaluation depth**: factual recall only — no generalization probes (paraphrase,
  cross-document reasoning) or negative controls (near-miss factually-wrong options)

## Release

- License: MIT (code and produced data); demo adapters are derivatives of Qwen2.5-7B
  (Apache 2.0) with notices included
- Demo adapters (bf16, ~81MB each): `release/adapters/` — mount back onto Qwen2.5-7B to
  reproduce multi-tenant serving
- Third-party benchmarks (LoCoMo / LongHealth) are downloaded by you, see `data/README.md`
- Release checklist and acceptance criteria: `RELEASE.md`; arXiv submission metadata:
  `docs/paper/arxiv_metadata.md`

## Suggested learning path (to reproduce)

1. [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges) — self-study +
   context distillation, the source of this project's synth.py ideas
2. [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora) — a hypernetwork that
   emits LoRA directly, skipping per-tenant training
3. [Continual-Intelligence/SEAL](https://github.com/Continual-Intelligence/SEAL) — the RL
   loop in which the model produces its own training data
4. This project = step 4: assembling those academic building blocks into an engineering product
