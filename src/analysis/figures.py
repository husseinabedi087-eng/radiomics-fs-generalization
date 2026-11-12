"""Figures (PDF and PNG), byte-reproducible: Agg backend, fixed metadata, fixed size and resolution."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PDF_META = {"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None}
PNG_META = {"Software": None}
DPI = 150


def _style():
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "figure.dpi": DPI, "savefig.dpi": DPI,
                         "pdf.fonttype": 42, "svg.hashsalt": "analysis"})


def _save(fig, stem):
    """Save stem.pdf and stem.png; return the two paths."""
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf", metadata=PDF_META)
    fig.savefig(png, format="png", metadata=PNG_META)
    plt.close(fig)
    return [pdf, png]


def anytime(table, header, stem):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for cfg, g in table.groupby("config", sort=True):
        g = g.sort_values("checkpoint")
        style = "--" if cfg.endswith("__C4") else "-"
        axes[0].plot(g["checkpoint"], g["mean_outer_auc"], style, marker="o", ms=2, lw=0.8, label=cfg)
        axes[1].plot(g["checkpoint"], g["mean_optimism_gap"], style, marker="o", ms=2, lw=0.8, label=cfg)
    for ax, lab in zip(axes, ["mean outer AUC", "mean optimism gap (standard inner AUC - outer AUC)"]):
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([250, 500, 1000, 2000])
        ax.set_xticklabels(["250", "500", "1000", "2000"])
        ax.set_xlabel("evaluations")
        ax.set_ylabel(lab)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5, ncol=1, loc="center left", bbox_to_anchor=(0.80, 0.5), frameon=False)
    fig.suptitle(f"{header}\nAnytime curves: mean over datasets of per-dataset means (dashed: C4)")
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    return _save(fig, stem)


def gap_vs_logpn(d, log_pn, header, stem):
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.array([log_pn[k] for k in d.index])
    ax.scatter(x, d.to_numpy(), s=12)
    for k, xi, yi in zip(d.index, x, d.to_numpy()):
        ax.annotate(str(k), (xi, yi), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("log(p / n)")
    ax.set_ylabel("per-dataset mean optimism gap (C0 configurations)")
    ax.set_title(f"{header}\nOptimism gap against log(p/n)")
    fig.tight_layout()
    return _save(fig, stem)


def differences(d, title, ylabel, margin, header, stem):
    _style()
    d = d.sort_values(kind="stable")
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = np.arange(len(d))
    ax.scatter(pos, d.to_numpy(), s=12, zorder=3)
    ax.axhline(0, color="grey", lw=0.6)
    for m in (-margin, margin):
        ax.axhline(m, color="tab:red", lw=0.8, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels([str(k) for k in d.index], rotation=90, fontsize=5)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{header}\n{title} (dashed: +/- {margin})")
    fig.tight_layout()
    return _save(fig, stem)
