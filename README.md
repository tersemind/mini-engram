# mini-engram

一个"把知识烤进模型参数"的最小端到端工程仿制——对应 Engram（Hazy Research 系）的产品链路：
**数据源 → self-study 合成训练集 → 按租户烤 LoRA adapter → 一个 base model 挂多 adapter 托管**。

与检索式记忆（Mem0/RAG）的区别：知识以 LoRA 参数形式存在，推理时不需要检索、不需要把文档塞进上下文。

## 链路

```
scripts/make_demo_corpus.py   虚构公司"星澜科技"wiki（base model 绝不可能知道的事实）
engram/ingest.py              目录/git 仓库 → chunks.jsonl
engram/synth.py               self-study：base model 围绕语料自问自答 → QA 训练集
engram/train_lora.py          PEFT LoRA 微调 → data/adapters/<租户>/
engram/eval.py                同一套题：闭卷(无adapter) vs 烤入后(带adapter) 的 F1 对比
scripts/serve.sh              vLLM multi-LoRA：一个 base 挂全部租户 adapter，OpenAI 兼容 API
scripts/chat.py               调用演示：同一问题分别问 base 和租户 adapter
```

## 快速开始（A10 24GB 可跑）

```bash
bash scripts/setup_env.sh        # venv + 依赖（首次约 10 分钟）
bash scripts/run_demo.sh         # 端到端 demo（首次会自动下载 Qwen2.5-7B-Instruct）
```

跑自己的数据：

```bash
.venv/bin/python -m engram.ingest --tenant myteam --source /path/to/docs     # 或 git 仓库 URL
.venv/bin/python -m engram.synth  --tenant myteam
.venv/bin/python -m engram.train_lora --tenant myteam
.venv/bin/python -m engram.eval   --tenant myteam
```

多租户托管：

```bash
bash scripts/serve.sh                       # 自动挂载 data/adapters/ 下所有 adapter
.venv/bin/python scripts/chat.py --tenant myteam "你们的报销系统叫什么？"
```

## 配置

- `MINI_ENGRAM_BASE_MODEL`：换 base model（默认 `Qwen/Qwen2.5-7B-Instruct`，ModelScope 下载）
- synth/train/eval 的 `--model` 参数可单独指定生成/训练用的模型

## 实验结果（2026-09-20，A10 24GB，Qwen2.5-7B-Instruct）

| 租户 | 合成 QA | 训练（20 epochs） | 闭卷 F1 | 烤入后 F1 |
|---|---|---|---|---|
| xinglan（星澜科技） | 123（双通道合成） | 168s | 0.126 | **0.649** |
| hanhai（瀚海机器人） | 93 | 127s | 0.113 | **0.664** |

- 单通道合成（52 QA）对照组只有 0.575，且漏教的事实模型必然瞎编 → **数据覆盖度是核心配方**
- 串味检查（交叉问答）：F1 ~0.33，逐题分析为指标假象（短答案精度偏好 + 通用职场常识巧合），无真实参数泄漏；但 adapter 对非本租户问题会自信瞎编（如把自家事实安到别家问题上）→ 多租户产品需要拒答/路由校准
- 遗忘检查：6 组通用探针（代码/常识/写作/身份），+LoRA 与 base 表现一致，身份探针无公司人设泄漏
- 托管演示：一个 vLLM 进程挂两个租户 adapter，HTTP 按 model 名路由，base 答"不知道"的问题，各租户准确回答（"算盘"/"白鹿生鲜"）

## 竞品对比 bench（2026-09-21，Qwen2.5-7B-Instruct，A10 24GB）

统一四条件：**base**（闭卷）/ **full**（语料全文塞上下文）/ **rag5**（BM25 top-5，Mem0/RAG 式检索记忆）/ **lora**（每租户烤 adapter 后闭卷）。训练配方：self-study 双通道合成（对话语料用 dialog 模板，问答语言与语料一致）→ LoRA r=16 QA-SFT。

### 1. 合成公司 wiki（可控对照，防泄漏 by construction）

| 租户 | 题集 | base | full | rag5 | lora:v1 | lora:v2(+拒答) |
|---|---|---|---|---|---|---|
| xinglan | own | 0.126 | 0.349 | 0.361 | **0.666** | 0.621 |
| xinglan | para(换问法) | 0.090 | 0.333 | 0.346 | **0.581** | 0.581 |
| xinglan | cross(别家题) | 0.109 | 0.022 | 0.024 | 0.363 | 0.363 |
| hanhai | own | 0.113 | 0.349 | 0.349 | **0.664** | — |
| hanhai | para | 0.101 | 0.330 | 0.325 | **0.592** | — |
| hanhai | cross | 0.111 | 0.042 | 0.047 | 0.324 | — |

lora 在 own/para 上约为 full/rag5 上界的 **1.8-2×** 且对换问法鲁棒（-0.06~0.09）；代价是 cross 上自信瞎编（0.32-0.36 经逐题核对为指标假象+常识巧合，无真实参数泄漏，但多租户产品必须配路由/拒答校准）。

### 2. LoCoMo 公开基准（10 段真实长对话 8k-16k 词，1986 官方题，官方 token-F1 口径）

| 条件 | overall | single_hop | multi_hop | temporal | open_domain | adversarial |
|---|---|---|---|---|---|---|
| base | 0.039 | 0.059 | 0.055 | 0.012 | 0.074 | 0.002 |
| full(32K) | 0.285 | 0.222 | 0.161 | 0.071 | 0.067 | 0.684 |
| rag5 | 0.273 | 0.127 | 0.071 | 0.035 | 0.058 | **0.895** |
| lora | 0.181 | **0.304** | **0.193** | **0.091** | **0.213** | 0.000 |

- **会的事：参数化记忆最好**。四个事实类目 lora 全面第一（single_hop 0.304 vs full 0.222 vs rag5 0.127）；multi_hop 上 rag5 仅 0.071（BM25 只捞单 session，跨 session 拼不起来）。
- **不会的事：参数化记忆不会拒答**。adversarial 题（问的谈话里根本没发生，446/1986=22%）lora 得分字面 0——它 100% 自信编造，而 rag5 靠忠实指令 0.895 正确拒答。overall 的 lora < full/rag5 完全由这一列造成。
- **剔除 adversarial 后**（n=1540）：lora **0.234** > full 0.170 > rag5 0.093 > base 0.048——参数化记忆的"已知知识密度"显著最高。
- temporal 全体都低（0.03-0.09）：7B 对日期格式（"7 May 2023"）(token-F1 惩罚严格），lora 仍第一。
- 对照提示：Mem0 论文的 LoCoMo 数字是 LLM-as-judge 口径且 base 模型不同，只可作量级参照，不可直接比较；Zep/Mem0 现均已转向此基准。

### 3. LongHealth（Engram 家族官方尺子，20 虚构病人 400 五选一，MC accuracy）

Cartridges 官方演示同款闭卷基准（其 0.96GB cartridge 闭卷 55.1% vs 全量 ICL）；我们的 adapter 仅 ~40M 参数、~5 分钟/病人。

| 条件 | overall | 说明 |
|---|---|---|
| base | 0.398 | 虚构病人，但医学常识可迁移 |
| full(8K) | **0.625** | 读着信答，阅读理解上限 |
| rag5 | 0.560 | 检索 5 块够用但不如全文 |
| lora | **0.355** | **低于 base**——细粒度临床细节没烤住 |

- **与前两张表相反的结果，恰好是论文最重要的讨论点**：当评测是"细粒度判别"（剂量数字、方案组成、否定式提问 NOT-part-of）时，朴素 QA-SFT 烤入（无课程学习、无防遗忘混合、~100 题/病人）会扰动 base 的医学推理却没换来精确记忆；
  字母抽取不是原因（lora null-letter 0/400，错答全是知识性错误）。
- 对照 Cartridges 55.1%：差距来自训练配方（持续预训练级 cartridge vs 我们的最小 SFT），
  说明"烤入"的上限取决于配方深度——这正是护城河章节的实证注脚。
- 任务依赖结论：**闭卷原子事实回忆**（wiki/LoCoMo 短答 F1）参数化记忆显著占优；
  **文档级细粒度理解**（LongHealth MC）读原文更好。参数化记忆不是检索的万能替代品。

## 这条仿制链路刻意简化了什么（= Engram 真正的护城河）

- **训练配方**：这里是最朴素的 QA SFT；没有课程学习、没有混合通用数据防遗忘
- **遗忘控制**：烤入新知识后通用能力/旧知识退化多少，没有监控
- **增量更新**：语料变了只能全量重烤，没有"增量 adapter"机制
- **评测深度**：只测事实回忆，没测泛化（换问法、跨文档推理）和负面对照（相似但事实错误的选项）

## 对外发布

- License：MIT（代码与产出数据）；演示 adapter 为 Qwen2.5-7B（Apache 2.0）派生物，随附声明
- 演示 adapter（bf16，~81MB/个）：`release/adapters/`，挂回 Qwen2.5-7B 即复现多租户托管
- 第三方基准（LoCoMo/LongHealth）自行下载，见 `data/README.md`
- 发布清单与验收标准：`RELEASE.md`；论文投稿元数据：`docs/paper/arxiv_metadata.md`

## 对应的学习路径（复现建议）

1. [HazyResearch/cartridges](https://github.com/HazyResearch/cartridges) — self-study + context distillation，本项目 synth.py 的思想来源
2. [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora) — hypernetwork 直接生成 LoRA，免去逐租户训练
3. [Continual-Intelligence/SEAL](https://github.com/Continual-Intelligence/SEAL) — 模型自产训练数据的 RL 闭环
4. 本项目 = 第 4 步：把上述学术积木拼成工程产品
