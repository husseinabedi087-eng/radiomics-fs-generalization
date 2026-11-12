"""Per-dataset scores, confirmatory tests (section 8, amendment 1) and secondary analyses (section 11).

Run-level outer AUC at a checkpoint: the outer AUC of the subset the run returns at that checkpoint
(checkpoints[c]); for C4, the knee record c4[c] of the C0 run; for a baseline, its returned subset
(checkpoint 2000 only). Optimism gap: standard inner AUC of that subset minus its outer AUC.
Per-dataset score: mean over all runs of the dataset and configuration (outer folds x search seeds;
random forest: outer folds x model seeds). Primary checkpoint 2000.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats as st
from .load import CHECKPOINTS, Refused

PRIMARY = 2000
C0_SEARCHERS = ["random", "hill_climb", "one_plus_one", "ga", "bpso_S", "bpso_V", "de_S", "de_V", "gwo_S", "gwo_V",
                "woa_S", "woa_V"]
CTRL_SEARCHERS = ["ga", "bpso_S", "de_S", "gwo_S"]
CONTROLS = ["C0", "C1", "C2", "C3", "C4"]
METAPHOR = ["gwo_S", "gwo_V", "woa_S", "woa_V"]
CLASSICAL = ["ga", "bpso_S", "bpso_V", "de_S", "de_V"]
BASELINES = ["all_k", "l1", "enet", "ftest", "mrmr", "rf"]
H3_MARGIN, H5_MARGIN = 0.02, 0.01
EXPECTED = ([f"{s}__C0" for s in C0_SEARCHERS] + [f"{s}__{c}" for c in ("C1", "C2", "C3") for s in CTRL_SEARCHERS]
            + [f"{s}__C4" for s in C0_SEARCHERS]  # section 7: C4 is applied to every searcher
            + [f"baseline__{b}" for b in BASELINES])


# ---------------- tables ----------------

def run_table(records):
    rows = []
    for r in records:
        for c, v in r["checkpoints"].items():
            rows.append({"plan": r["plan"], "dataset": r["dataset"], "config": r["config"], "family": r["family"],
                         "control": r["control"], "control_value": r["control_value"], "kind": r["kind"],
                         "repeat": r["repeat"], "fold": r["fold"], "seed_index": r["seed_index"], "checkpoint": c,
                         "auc": v["auc"], "std_auc": v["std_auc"], "gap": v["std_auc"] - v["auc"],
                         "brier": v["brier"], "size": v["size"], "p": r["p"], "n": r["n_train"] + r["n_test"],
                         "K": r["K"]})
    df = pd.DataFrame(rows)
    df["log_pn"] = np.log(df["p"] / df["n"])
    return df


def select_control_values(df, control_values):
    """Keep C1-C3 rows at the configured values; returns (df, excluded configuration labels)."""
    excluded = []
    keep = np.ones(len(df), dtype=bool)
    for ctrl in ("C1", "C2", "C3"):
        m = (df["control"] == ctrl).to_numpy()
        if not m.any():
            continue
        values = sorted(df.loc[m, "control_value"].unique())
        if ctrl not in control_values:
            if len(values) > 1:
                raise Refused(f"{ctrl} has values {values}; set control_values.{ctrl} in the config")
            continue
        chosen = float(control_values[ctrl])
        if chosen not in values:
            raise Refused(f"{ctrl} = {chosen} not present (values {values})")
        drop = m & (df["control_value"] != chosen).to_numpy()
        excluded += sorted({f"{c} ({ctrl} = {v:g})" for c, v in
                            zip(df.loc[drop, "config"], df.loc[drop, "control_value"])})
        keep &= ~drop
    return df[keep].reset_index(drop=True), excluded


def per_dataset(df, metric, checkpoint=PRIMARY):
    sub = df[df["checkpoint"] == checkpoint]
    return sub.groupby(["dataset", "config"])[metric].mean().unstack("config").sort_index(axis=0).sort_index(axis=1)


def _avail(table, names):
    return [c for c in names if c in table.columns], [c for c in names if c not in table.columns]


def _series(s):
    return {k: st._f(v) for k, v in s.items()}


# ---------------- confirmatory family ----------------

def _wording(h, res):
    if res.get("skipped"):
        return "not tested (configurations missing)"
    rej = res["reject_holm"]
    if h == "H1":
        return "optimism gap greater than zero" if rej else "no detectable difference"
    if h == "H2":
        return "primary control raises outer AUC" if rej else "no detectable difference"
    if h == "H3":
        if rej:
            return f"equivalent (margin {H3_MARGIN})"
        two = res["two_sided"]["p"]
        if two is not None and two <= st.ALPHA:
            return "ga + C4 higher than L1" if res["effect"]["median"] > 0 else "ga + C4 lower than L1"
        return "no detectable difference"
    if h == "H4":
        if not res["outer_auc"].get("estimable", True):
            return "not estimable"
        return "searcher-by-control interaction detected" if rej else "no detectable interaction"
    if h == "H5":
        if rej:
            full = res["tost_full"]["p_tost"]
            if full is not None and full <= st.ALPHA:
                return f"equivalent (margin {H5_MARGIN})"
            return f"not better than classical searchers by more than {H5_MARGIN} AUC"
        two = res["two_sided"]["p"]
        if two is not None and two <= st.ALPHA:
            return "metaphor-based higher than classical" if res["effect"]["median"] > 0 else \
                "metaphor-based lower than classical"
        return "no detectable difference"
    raise ValueError(h)


def h3_test(d):
    """H3 on per-dataset d = ga + C4 minus L1: TOST (margin 0.02) and a two-sided Wilcoxon."""
    t = st.tost(d, H3_MARGIN)
    return {"tost": t, "two_sided": st.wilcoxon(d, "two-sided"), "effect": st.effect(d), "p_raw": t["p_tost"]}


def h5_test(d):
    """H5 (amendment 1) on per-dataset d = metaphor minus classical: wilcoxon(d - 0.01, 'less');
    secondary: full TOST (margin 0.01) and a two-sided Wilcoxon."""
    t = st.wilcoxon(np.asarray(d, dtype=float) - H5_MARGIN, "less")
    return {"test": t, "tost_full": st.tost(d, H5_MARGIN), "two_sided": st.wilcoxon(d, "two-sided"),
            "effect": st.effect(d), "p_raw": t["p"]}


def h4_frame(df):
    sub = df[df["family"].isin(CTRL_SEARCHERS) & df["control"].isin(CONTROLS) & (df["kind"] == "search")].copy()
    sub["searcher"] = sub["family"]
    sub["partition"] = sub["repeat"].astype(str) + "_" + sub["fold"].astype(str)
    sub["log_budget"] = np.log(sub["checkpoint"].astype(float))
    return sub


def confirmatory(df, metric_auc="auc"):
    """H1-H5 on the given run table (already filtered to the chosen control values and datasets)."""
    auc, gap = per_dataset(df, metric_auc), per_dataset(df, "gap")
    out = {}

    used, missing = _avail(gap, [f"{s}__C0" for s in C0_SEARCHERS])
    if used:
        d = gap[used].dropna().mean(axis=1)
        out["H1"] = {"configurations_used": used, "configurations_missing": missing, "d_per_dataset": _series(d),
                     "test": st.wilcoxon(d, "greater"), "effect": st.effect(d)}
        out["H1"]["p_raw"] = out["H1"]["test"]["p"]
    else:
        out["H1"] = {"skipped": True, "configurations_missing": missing}

    pairs = [s for s in CTRL_SEARCHERS if f"{s}__C4" in auc.columns and f"{s}__C0" in auc.columns]
    if pairs:
        diffs = pd.DataFrame({s: auc[f"{s}__C4"] - auc[f"{s}__C0"] for s in pairs}).dropna()
        d = diffs.mean(axis=1)
        out["H2"] = {"searchers_used": pairs, "searchers_missing": [s for s in CTRL_SEARCHERS if s not in pairs],
                     "d_per_dataset": _series(d), "test": st.wilcoxon(d, "greater"), "effect": st.effect(d)}
        out["H2"]["p_raw"] = out["H2"]["test"]["p"]
    else:
        out["H2"] = {"skipped": True, "searchers_missing": CTRL_SEARCHERS}

    if {"ga__C4", "baseline__l1"} <= set(auc.columns):
        d = (auc["ga__C4"] - auc["baseline__l1"]).dropna()
        out["H3"] = {"d_per_dataset": _series(d), **h3_test(d)}
    else:
        out["H3"] = {"skipped": True, "configurations_missing": [c for c in ("ga__C4", "baseline__l1")
                                                                 if c not in auc.columns]}

    h4 = h4_frame(df)
    if h4["searcher"].nunique() >= 2 and h4["control"].nunique() >= 2:
        out["H4"] = {"searchers": sorted(h4["searcher"].unique()), "controls": sorted(h4["control"].unique()),
                     "outer_auc": st.mixed_lrt(h4, metric_auc), "optimism_gap_descriptive": st.mixed_lrt(h4, "gap")}
        out["H4"]["p_raw"] = out["H4"]["outer_auc"]["p"]
    else:
        out["H4"] = {"skipped": True}

    m_used, m_miss = _avail(auc, [f"{s}__C0" for s in METAPHOR])
    c_used, c_miss = _avail(auc, [f"{s}__C0" for s in CLASSICAL])
    if m_used and c_used:
        sub = auc[m_used + c_used].dropna()
        d = sub[m_used].mean(axis=1) - sub[c_used].mean(axis=1)
        out["H5"] = {"metaphor_used": m_used, "classical_used": c_used, "configurations_missing": m_miss + c_miss,
                     "d_per_dataset": _series(d), **h5_test(d)}
    else:
        out["H5"] = {"skipped": True, "configurations_missing": m_miss + c_miss}

    names = ["H1", "H2", "H3", "H4", "H5"]
    adj, rej = st.holm([out[h].get("p_raw") for h in names])
    for h, a, r in zip(names, adj, rej):
        out[h]["p_holm"], out[h]["reject_holm"] = a, r
        out[h]["wording"] = _wording(h, out[h]) if not out[h].get("skipped") and a is not None else \
            "not tested (configurations missing)"
    out["holm"] = {"alpha": st.ALPHA, "family": names, "p_raw": [out[h].get("p_raw") for h in names],
                   "p_adjusted": adj, "reject": rej,
                   "note": "H3 enters with p_TOST; H4 with the likelihood-ratio p; missing p-values are left out"}
    return out


# ---------------- secondary ----------------

def brier_paired(df):
    """Brier-score versions of H2, H3, H5 (two-sided, descriptive; lower Brier is better) and of H4."""
    b = per_dataset(df, "brier")
    out = {"H1": {"not_applicable": "no inner Brier score exists"}}
    pairs = [s for s in CTRL_SEARCHERS if f"{s}__C4" in b.columns and f"{s}__C0" in b.columns]
    if pairs:
        d = pd.DataFrame({s: b[f"{s}__C4"] - b[f"{s}__C0"] for s in pairs}).dropna().mean(axis=1)
        out["H2"] = {"d_per_dataset": _series(d), "two_sided": st.wilcoxon(d, "two-sided"), "effect": st.effect(d)}
    if {"ga__C4", "baseline__l1"} <= set(b.columns):
        d = (b["ga__C4"] - b["baseline__l1"]).dropna()
        out["H3"] = {"d_per_dataset": _series(d), "two_sided": st.wilcoxon(d, "two-sided"), "effect": st.effect(d)}
    m_used, _ = _avail(b, [f"{s}__C0" for s in METAPHOR])
    c_used, _ = _avail(b, [f"{s}__C0" for s in CLASSICAL])
    if m_used and c_used:
        sub = b[m_used + c_used].dropna()
        d = sub[m_used].mean(axis=1) - sub[c_used].mean(axis=1)
        out["H5"] = {"d_per_dataset": _series(d), "two_sided": st.wilcoxon(d, "two-sided"), "effect": st.effect(d)}
    h4 = h4_frame(df)
    if h4["searcher"].nunique() >= 2 and h4["control"].nunique() >= 2:
        out["H4"] = st.mixed_lrt(h4, "brier")
    return out


def anytime(df):
    sub = df[df["kind"] == "search"]
    per_ds = sub.groupby(["checkpoint", "config", "dataset"])[["auc", "gap"]].mean().reset_index()
    t = per_ds.groupby(["checkpoint", "config"]).agg(mean_outer_auc=("auc", "mean"), mean_optimism_gap=("gap", "mean"),
                                                     n_datasets=("dataset", "nunique")).reset_index()
    return t.sort_values(["config", "checkpoint"], kind="stable").reset_index(drop=True)


def random_vs_guided(df):
    auc = per_dataset(df, "auc")
    if "random__C0" not in auc.columns:
        return {"skipped": True, "reason": "random__C0 not in the plans",
                "configurations_missing": ["random__C0"]}, pd.DataFrame()
    guided, missing = _avail(auc, [f"{s}__C0" for s in C0_SEARCHERS if s != "random"])
    rows = []
    for g in guided:
        d = (auc[g] - auc["random__C0"]).dropna()
        t = st.wilcoxon(d, "two-sided")
        rows.append({"guided": g, "n_datasets": int(d.size), "mean_d": st._f(d.mean()), "median_d": st._f(d.median()),
                     "p": t["p"], "method": t["method"], "rank_biserial": st.rank_biserial(d)})
    adj, rej = st.holm([r["p"] for r in rows])
    for r, a, j in zip(rows, adj, rej):
        r["p_holm"], r["reject_holm"] = a, j
    return {"comparisons": rows, "configurations_missing": missing, "holm_family": "random vs each guided C0"}, \
        pd.DataFrame(rows)


def size_stability(records, configs):
    """Subset size and Nogueira stability across outer folds, per dataset x configuration x seed index,
    then averaged over seed indices. Features are original column names; d = p after the data-handling rules."""
    groups = {}
    for r in records:
        if r["config"] not in configs or PRIMARY not in r["checkpoints"]:
            continue
        key = (r["dataset"], r["config"], r["seed_index"])
        g = groups.setdefault(key, {"p": r["p"], "subsets": [], "sizes": []})
        c = r["checkpoints"][PRIMARY]
        g["subsets"].append(c["subset"])
        g["sizes"].append(c["size"])
    rows = []
    for (ds, cfg, seed), g in sorted(groups.items()):
        rows.append({"dataset": ds, "config": cfg, "seed_index": seed, "n_folds": len(g["subsets"]), "d": g["p"],
                     "mean_size": float(np.mean(g["sizes"])), "stability": st.nogueira_stability(g["subsets"], g["p"])})
    per_seed = pd.DataFrame(rows)
    per_ds = per_seed.groupby(["dataset", "config"]).agg(
        n_seeds=("seed_index", "nunique"), n_folds=("n_folds", "min"), d=("d", "first"),
        mean_size=("mean_size", "mean"), stability=("stability", lambda s: s.dropna().mean() if s.notna().any()
                                                    else np.nan)).reset_index()
    summary = per_ds.groupby("config").agg(n_datasets=("dataset", "nunique"), mean_size=("mean_size", "mean"),
                                           mean_stability=("stability", "mean"),
                                           n_undefined_stability=("stability", lambda s: int(s.isna().sum()))
                                           ).reset_index()
    return per_ds, summary


def friedman(df):
    auc = per_dataset(df, "auc").dropna()
    cols = list(auc.columns)
    out = {"n_datasets": int(len(auc)), "selectors": cols}
    if len(cols) < 3 or len(auc) < 2:
        out["skipped"] = True
        return out, pd.DataFrame()
    stat, p = sps.friedmanchisquare(*[auc[c].to_numpy() for c in cols])
    out.update({"statistic": st._f(stat), "p": st._f(p), "context_only": True})
    rows = []
    for a, b in itertools.combinations(cols, 2):
        d = auc[a] - auc[b]
        t = st.wilcoxon(d, "two-sided")
        rows.append({"a": a, "b": b, "mean_d": st._f(d.mean()), "p": t["p"], "method": t["method"]})
    adj, rej = st.holm([r["p"] for r in rows])
    for r, x, j in zip(rows, adj, rej):
        r["p_holm"], r["reject_holm"] = x, j
    out["n_pairs"] = len(rows)
    out["n_pairs_rejected_holm"] = int(sum(bool(r["reject_holm"]) for r in rows))
    return out, pd.DataFrame(rows)


def missing_report(df):
    present = sorted(df["config"].unique())
    return {"expected_full_design": EXPECTED, "present": present,
            "missing": [c for c in EXPECTED if c not in present],
            "note": "every analysis runs on the configurations present and lists the missing ones"}


__all__ = ["run_table", "select_control_values", "per_dataset", "confirmatory", "brier_paired", "anytime",
           "random_vs_guided", "size_stability", "friedman", "missing_report", "CHECKPOINTS"]
