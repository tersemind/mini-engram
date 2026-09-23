"""把两个 LoRA adapter 精确相加合并（skill + content 分层的纯 LoRA 实现）。

Engram 论文（arXiv:2606.19172）把"推理技能"放进一个所有租户共享的 adapter、
"事实内容"放进每租户的局部存储。本模块在纯 LoRA 框架下实现同一分层：
两个独立训练的 LoRA 可以精确相加 ΔW = s1·B1A1 + s2·B2A2，
用 concat 表示：A=[A1|A2], B=[s1·B1|s2·B2]，合并 rank = r1+r2，alpha=rank（scale=1）。
数学上与分别挂载两个等价，但保持"每租户单 adapter"的 vLLM 服务形态。

用法: python -m engram.merge_adapters --skill <skill adapter 目录> \
        --content <content adapter 目录> --out <输出目录>
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _cfg(d: Path) -> dict:
    return json.load(open(d / "adapter_config.json", encoding="utf-8"))


def merge(skill: Path, content: Path, out: Path, skill_scale: float = 1.0) -> None:
    cs, cc = _cfg(skill), _cfg(content)
    for k in ("target_modules", "task_type"):
        if set(cs.get(k, [])) != set(cc.get(k, [])):
            raise SystemExit(f"两个 adapter 的 {k} 不一致，无法合并: "
                             f"{cs.get(k)} vs {cc.get(k)}")
    ws = load_file(str(skill / "adapter_model.safetensors"))
    wc = load_file(str(content / "adapter_model.safetensors"))
    if set(ws) != set(wc):
        raise SystemExit("两个 adapter 的权重键不一致（模块集合相同仍可能有"
                         "层数差异），无法合并")

    ss = skill_scale * cs["lora_alpha"] / cs["r"]
    sc = cc["lora_alpha"] / cc["r"]
    merged = {}
    for key in ws:
        if key.endswith("lora_A.weight"):
            merged[key] = torch.cat([ws[key], wc[key]], dim=0)
        elif key.endswith("lora_B.weight"):
            merged[key] = torch.cat([ss * ws[key], sc * wc[key]], dim=1)
        else:
            merged[key] = ws[key]
    r_total = cs["r"] + cc["r"]

    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"))
    cfg = dict(cs)
    cfg["r"] = r_total
    cfg["lora_alpha"] = r_total  # scale=1，缩放已折进 B
    (out / "adapter_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8")
    for f in ("README.md",):
        if (content / f).exists():
            shutil.copy(content / f, out / f)
    print(f"[merge] rank {cs['r']}({ss:.3f}x) + {cc['r']}({sc:.3f}x) -> "
          f"{r_total}(1.0x), {len(merged)} tensors -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True, help="共享技能 adapter 目录")
    ap.add_argument("--content", required=True, help="租户内容 adapter 目录")
    ap.add_argument("--out", required=True, help="合并输出目录")
    ap.add_argument("--skill-scale", type=float, default=1.0,
                    help="skill 分支缩放 λ（干涉过大时 <1）")
    args = ap.parse_args()
    merge(Path(args.skill), Path(args.content), Path(args.out), args.skill_scale)


if __name__ == "__main__":
    main()
