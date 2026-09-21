# NOTICE（第三方数据与派生物许可）

本仓库代码按 LICENSE（MIT）发布；但以下第三方基准数据及其派生物另有条款，再分发时必须遵守：

| 基准 | 原始许可 | 我们的派生物 | 再分发要求 |
|---|---|---|---|
| **LoCoMo**（snap-research/locomo） | **CC BY-NC 4.0** | `data/locomo/`（若有）、LoCoMo 题集派生 `data/bench/locomo_*`、训练产物 `data/adapters/locomo_*`、`data/dataset/locomo_*` | **非商业使用**；保留署名（引用 LoCoMo 论文与仓库）；派生物同许可（BY-NC-SA 兼容性遵循 CC 官方指引：ShareAlike 并非强制，但 BY-NC 署名+非商业是） |
| **LongHealth**（kbressem/LongHealth） | Apache 2.0 | `data/longhealth/`、`data/bench/longhealth_*`、`data/adapters/lh_*` | 保留 Apache 2.0 许可与 NOTICE；派生物可并入任何许可但需附原许可文本 |
| Qwen2.5-7B-Instruct（基座） | Apache 2.0 | 全部 LoRA adapter（其派生物） | 保留 Apache 2.0 声明；adapter 目录内已随附 |

仓库不直接再分发 LoCoMo/LongHealth 的原始数据文件（见 `data/README.md` 的
下载指引）；逐题明细（bench rows）与基准训练 adapter 属派生物，**LoCoMo 相关部分禁止商用**，
已在 RELEASE.md 的发布物分级中标注。

合成语料（星澜/瀚海）与对应 adapter 为程序生成，无第三方权利，按 MIT 随仓库发布。
