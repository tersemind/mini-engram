"""把 fp32 adapter 转成 bf16 发布版（体积减半，~154MB → ~77MB）。

用法: .venv/bin/python scripts/export_adapter_bf16.py \
        data/adapters/xinglan release/adapters/xinglan
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not (src / "adapter_model.safetensors").exists():
        raise SystemExit(f"不是 adapter 目录: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    sd = {k: v.to(torch.bfloat16)
          for k, v in load_file(str(src / "adapter_model.safetensors")).items()}
    save_file(sd, str(dst / "adapter_model.safetensors"),
              metadata={"format": "pt"})
    for f in ("adapter_config.json", "chat_template.jinja",
              "tokenizer_config.json", "README.md"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    n = sum(v.numel() for v in sd.values())
    mb = (dst / "adapter_model.safetensors").stat().st_size / 1e6
    print(f"[export] {dst.name}: {n:,} 参数, bf16, {mb:.1f}MB")


if __name__ == "__main__":
    main()
