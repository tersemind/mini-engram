"""补画论文 fig1_synthetic_matrix 与 fig2_robustness。

样式借用 AgentLaboratory run_experiments.py（seaborn-v0_8-colorblind + Set2 配色，
pdf+png 双输出）。数据直接读 bench/summary_{xinglan,hanhai}.json（已验证锚点）。
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH = pathlib.Path("/root/train-study/mini-engram/data/bench")
FIGS = BENCH / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-colorblind")
matplotlib.rcParams.update({"font.size": 10, "figure.dpi": 1315})

TENANTS = ["xinglan", "hanhai"]
PROBES = ["own", "para", "cross"]
CONDITIONS = ["base", "full", "rag5", "lora:v1", "lora:v2"]  # v2 仅 xinglan

F1 = {t: json.load(open(BENCH / f"summary_{t}.json")) for t in TENANTS}


def cell(tenant, probe, cond):
    if cond == "lora:v2" and tenant != "xinglan":
        return None
    return F1[tenant][probe][cond]


# ---------------- fig1: grouped bar matrix ----------------
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(TENANTS) * len(PROBES))
offsets = np.linspace(-0.3, 0.3, len(CONDITIONS))
width = 0.15

colors = plt.cm.Set2(np.linspace(0, 1, len(CONDITIONS)))
for i, cond in enumerate(CONDITIONS):
    vals = []
    for tenant in TENANTS:
        for probe in PROBES:
            v = cell(tenant, probe, cond)
            vals.append(v if v is not None else 0.0)
    ax.bar(x + offsets[i], vals, width, label=cond, color=colors[i])

tick_labels = [f"{t}\n{p}" for t in TENANTS for p in PROBES]
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, rotation=0)
ax.set_ylabel("Token F1")
ax.set_ylim(0, 0.75)
ax.set_title("Synthetic Wiki: token F1 by tenant, probe type, and condition")
ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig1_synthetic_matrix.{ext}", format=ext, bbox_inches="tight")
plt.close(fig)

# ---------------- fig2: robustness deltas ----------------
# delta_para = own - para; delta_cross = own - cross（越大表示掉得越狠）
rows = []
for tenant in TENANTS:
    for cond in CONDITIONS:
        own = cell(tenant, "own", cond)
        para = cell(tenant, "para", cond)
        cross = cell(tenant, "cross", cond)
        if own is None or para is None or cross is None:
            continue
        rows.append(
            {
                "tenant": tenant,
                "condition": cond,
                "para_drop": own - para,
                "cross_drop": own - cross,
            }
        )

conds_present = [c for c in CONDITIONS if any(r["condition"] == c for r in rows)]
tenants_present = TENANTS

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (drop_key, drop_name) in zip(
    axes, [("para_drop", "Own - Para F1 drop"), ("cross_drop", "Own - Cross F1 drop")]
):
    data = np.array(
        [
            [next((r[drop_key] for r in rows if r["condition"] == c and r["tenant"] == t), 0.0)
             for c in conds_present]
            for t in tenants_present
        ]
    )
    xpos = np.arange(len(conds_present))
    w = 0.35
    for j, t in enumerate(tenants_present):
        ax.bar(xpos + (j - 0.5) * w, data[j], w, label=t, color=plt.cm.Set2([j]))
    ax.set_xticks(xpos)
    ax.set_xticklabels(conds_present, rotation=45, ha="right")
    ax.set_ylabel("F1 drop")
    ax.set_title(drop_name)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Tenant")

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGS / f"fig2_robustness.{ext}", format=ext, bbox_inches="tight")
plt.close(fig)

print("fig1/fig2 saved to", FIGS)
for r in rows:
    print(r)
