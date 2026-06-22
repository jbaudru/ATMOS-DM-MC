#!/usr/bin/env python3
"""
Standalone plotting script for GraphIDyOM results.

Reads the JSON files saved by train_and_evaluate.py and regenerates all figures
without re-running any training.

Usage
-----
    python plot_results.py --results-dir results_unified
    python plot_results.py --results-dir results_unified --output-dir figures
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

# UTF-8 on Windows
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_name(name: str) -> str:
    """Shorten model name for plot labels."""
    n = name
    # Categorical-id DM models are the proposed "DM-MC<order>" family.
    n = n.replace("GraphIDyOM_categorical_id", "DM-MC")
    n = n.replace("GraphIDyOM_", "DM_")
    n = n.replace("categorical_id", "CAT")
    n = n.replace("debruijn",       "DEBR")
    n = n.replace("graph_embedding","N2V")
    n = n.replace("_ord", "")
    # Pure-backoff Markov (mc2p, mc5p, mc10p): keep order, add lowercase 'p'
    if re.fullmatch(r"MC(\d+)P", n, re.IGNORECASE):
        order = re.fullmatch(r"MC(\d+)P", n, re.IGNORECASE).group(1)
        n = f"MC{order}"
    # Entropy-weighted / legacy Markov (mc2, mc5, mc10)
    elif re.fullmatch(r"MC\d+", n, re.IGNORECASE):
        n = "HW-" + n.upper()
    # Graph-embedding MC variant (mc2_ge, mc5_ge, mc10_ge)
    if re.fullmatch(r"MC\d+_GE", n, re.IGNORECASE):
        n = "HW-MC" + "_N2V" + re.fullmatch(r"MC(\d+)_GE", n, re.IGNORECASE).group(1)
    return n


def _markov_order_key(name: str) -> int:
    """Extract trailing Markov order for sorting (e.g. _ord2 → 2, MC5 → 5, MC5P → 5)."""
    m = re.search(r'(?:ord|MC)(\d+)P?$', name, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _sort_group(names: List[str]) -> List[str]:
    """Sort a group by (Markov order, then name) so ord2 < ord5 < ord10."""
    return sorted(names, key=lambda n: (_markov_order_key(n), n))


def _simple_mc_type_key(name: str) -> int:
    """Type bucket for simple Markov chains so the same type stays together.

    0 = entropy-weighted (HW-MCk: mc2/mc5/mc10)
    1 = pure-backoff      (MCkP:   mc2p/mc5p/mc10p)
    2 = graph-embedding   (MCk_GE: mc2_ge/mc5_ge/mc10_ge)
    """
    up = name.upper()
    if up.endswith("_GE"):
        return 2
    if re.fullmatch(r"MC\d+P", up):
        return 1
    return 0


def _sort_simple_mc(names: List[str]) -> List[str]:
    """Group simple Markov chains by type, then Markov order, then name."""
    return sorted(names, key=lambda n: (_simple_mc_type_key(n),
                                        _markov_order_key(n), n))


def _model_encoding_and_order(name: str) -> tuple[str, str]:
    """Infer encoding and Markov order from model name."""
    up = name.upper()
    if name.startswith("GraphIDyOM_"):
        if "categorical_id" in name:
            enc = "Categorical ID"
        elif "debruijn" in name:
            enc = "De Bruijn"
        elif "graph_embedding" in name:
            enc = "Graph Embedding"
        else:
            enc = "-"
        order = str(_markov_order_key(name)) if _markov_order_key(name) > 0 else "-"
        return enc, order

    _all_mc = {"MC2", "MC5", "MC10", "MC2P", "MC5P", "MC10P",
               "MC2_GE", "MC5_GE", "MC10_GE"}
    if up in _all_mc:
        enc = "Graph Embedding" if up in ("MC2_GE", "MC5_GE", "MC10_GE") else "Categorical ID"
        return enc, str(_markov_order_key(name)) if _markov_order_key(name) > 0 else "-"

    # All other baselines (AKOM, CPT, MOGEN, IOHMM) operate on raw node IDs
    return "Categorical ID", "-"


def _latex_escape(text: str) -> str:
    """Escape LaTeX special chars in plain text fields."""
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def _is_graph_embedding(name: str) -> bool:
    """True for graph-embedding / node2vec encoded models (DM_*_graph_embedding,
    and the *_GE Markov baselines).  These are excluded from the comparison
    table and figures, which report only the categorical-ID encoded models."""
    return "graph_embedding" in name or name.upper().endswith("_GE")


def _drop_graph_embedding(results):
    """Return ``results`` without any graph-embedding encoded model."""
    return {k: v for k, v in results.items() if not _is_graph_embedding(k)}


def _ordered_model_names(results: Dict[str, Dict]) -> List[str]:
    """Order models: DM-MC first, then baselines, then simple Markov chain."""
    _mc = {"MC2", "MC5", "MC10", "MC2P", "MC5P", "MC10P",
           "MC2_GE", "MC5_GE", "MC10_GE"}
    _other_b = {"AKOM", "CPT", "MOGEN", "IOHMM"}

    proposed = _sort_group([
        n for n in results
        if n.startswith("GraphIDyOM_")
    ])
    baselines = _sort_group([n for n in results if n.upper() in _other_b])
    simple_mc = _sort_simple_mc([n for n in results if n.upper() in _mc])

    # Keep any unknown entries between baselines and simple MC.
    known = set(proposed) | set(baselines) | set(simple_mc)
    unknown = _sort_group([n for n in results if n not in known])
    return proposed + baselines + unknown + simple_mc


def _filter_best_per_family(results: Dict[str, Dict]) -> List[str]:
    """
    Return a filtered list of model names keeping only the best-order
    representative per model family (selection criterion: accuracy_top1).

    Rules
    -----
    - GraphIDyOM: one per encoding (categorical_id / debruijn / graph_embedding),
      choosing the Markov order with the highest Acc@1.
    - Basic baselines (AKOM, CPT, MOGEN, IOHMM): always kept.
    - Entropy-weighted MC (mc5, mc10): keep the best-Acc@1 one.
    - Pure-backoff MC (mc2, mc5p, mc10p): keep the best-Acc@1 one.
    - GE MC baseline (mc2_ge, mc5_ge, mc10_ge): kept (at most one expected).
    """
    def _acc1(name: str) -> float:
        return float(results.get(name, {}).get("accuracy_top1", -1.0))

    kept: List[str] = []

    # GraphIDyOM: one per encoding, best order
    for enc in ("categorical_id", "debruijn", "graph_embedding"):
        prefix = f"GraphIDyOM_{enc}_ord"
        cands = [n for n in results if n.startswith(prefix)]
        if cands:
            kept.append(max(cands, key=_acc1))

    # Basic baselines: always all
    for bname in ("AKOM", "CPT", "MOGEN", "IOHMM"):
        kept.extend(n for n in results if n.upper() == bname)

    # Entropy-weighted MC: best of mc5, mc10
    ew = [n for n in results if n.upper() in ("MC2", "MC5", "MC10")]
    if ew:
        kept.append(max(ew, key=_acc1))

    # Pure-backoff MC: best of mc2, mc2p, mc5p, mc10p
    pb = [n for n in results if n.upper() in ("MC2P", "MC5P", "MC10P")]
    if pb:
        kept.append(max(pb, key=_acc1))

    # GE MC baseline
    ge = [n for n in results if n.upper() in ("MC2_GE", "MC5_GE", "MC10_GE")]
    if ge:
        kept.append(max(ge, key=_acc1))

    return kept


def write_comparison_tables(results: Dict[str, Dict], output_dir: str) -> None:
    """Write a single combined LaTeX table with next-node and AR metrics side by side.

    Unlike the figures, the table shows *every* model (all encodings and Markov
    orders); no best-per-family filtering is applied here.
    """
    results = _drop_graph_embedding(results)
    model_names = _ordered_model_names(results)
    baseline_set = {"AKOM", "CPT", "MOGEN", "IOHMM"}
    simple_mc_set = {"MC2", "MC5", "MC10", "MC2P", "MC5P", "MC10P",
                    "MC2_GE", "MC5_GE", "MC10_GE"}

    def model_label(name: str) -> str:
        if name.startswith("GraphIDyOM_"):
            order = _markov_order_key(name)
            return f"DM-MC{order}" if order > 0 else "DM-MC"
        up = name.upper()
        if re.fullmatch(r"MC\d+P", up):
            return up           # MC5P, MC10P as-is
        if up in {"MC2", "MC5", "MC10"}:
            return f"HW-{up}"
        if up in {"MC2_GE", "MC5_GE", "MC10_GE"}:
            return f"HW-{up.replace('_GE', '')}"
        return up

    def enc_abbrev(enc: str) -> str:
        return {
            "Categorical ID":  r"Cat.\ ID",
            "De Bruijn":       "De Bruijn",
            "Graph Embedding": "Graph Emb.",
            "-":               "--",
        }.get(enc, enc)

    def model_group(name: str) -> str:
        up = name.upper()
        if name.startswith("GraphIDyOM_"):
            return "proposed"
        if up in baseline_set:
            return "baseline"
        if up in simple_mc_set:
            return "simple_mc"
        return "other"

    def _max_metric(names: List[str], key: str) -> float:
        vals = [results.get(n, {}).get(key, float("nan")) for n in names]
        vals = [v for v in vals if v == v]
        return max(vals) if vals else float("nan")

    def _fmt_metric(value: float, best: float, std: float = float("nan")) -> str:
        if value != value:  # NaN
            return "--"
        std_str = f" ({std:.4f})" if std == std else ""
        if value == value and best == best and np.isclose(value, best, rtol=0.0, atol=1e-12):
            return rf"\textbf{{{value:.4f}}}{std_str}"
        return f"{value:.4f}{std_str}"

    # Acceptable Node Horizon Prediction (ar_horizon_acc75_mean): the mean number
    # of consecutive future nodes the model predicts in closed-loop (auto-
    # regressive) rollout while its cumulative top-1 accuracy stays >= 75% — i.e.
    # how far ahead the trajectory can be trusted. Higher is better (more steps).
    ar_names = [n for n in model_names if "ar_horizon_acc75_mean" in results.get(n, {})]
    has_ar = bool(ar_names)

    # Endpoint metrics are optional — only emit columns when at least one model
    # has them in the JSON.
    ep_names = [n for n in model_names if "ar_endpoint_match" in results.get(n, {})]
    hops_names = [n for n in model_names if "ar_endpoint_hops_median" in results.get(n, {})]
    dist_names = [n for n in model_names if "ar_endpoint_dist_km_median" in results.get(n, {})]
    # Dist@1 (endpoint), Acc@3/MRR and the AR@3/AR-MRR columns are intentionally
    # excluded: the slimmed table reports only Acc@1, certainty (IC), AR@1 and
    # endpoint distance.
    has_endpoint = False
    has_hops = bool(hops_names)
    # AR endpoint physical distance (km, lower is better).
    has_dist = bool(dist_names)
    # Information content (mean NLL of the realised next node, reported in bits)
    # is the IDyOM-native certainty metric: lower = the model places more
    # probability mass on the true node, i.e. it is *more certain and correct*.
    ic_names = [n for n in model_names if "nll" in results.get(n, {})]
    has_ic = bool(ic_names)
    # Auto-regressive NLL: same certainty metric, but measured along the
    # closed-loop rollout (−log P of the true next node at every AR step).
    # It is still captured/saved in the results, but intentionally NOT shown as
    # a table column.
    ar_nll_names = [n for n in model_names if "ar_nll" in results.get(n, {})]
    has_ar_nll = False
    _LN2 = float(np.log(2.0))

    def _min_metric(names: List[str], key: str) -> float:
        vals = [results.get(n, {}).get(key, float("nan")) for n in names]
        vals = [v for v in vals if v == v]
        return min(vals) if vals else float("nan")

    def _fmt_min_metric(value: float, best: float, std: float = float("nan")) -> str:
        """Bold when value equals the minimum (lower-is-better metrics)."""
        if value != value:
            return "--"
        std_str = f" ({std:.4f})" if std == std else ""
        if best == best and np.isclose(value, best, rtol=0.0, atol=1e-12):
            return rf"\textbf{{{value:.4f}}}{std_str}"
        return f"{value:.4f}{std_str}"

    best_acc1  = _max_metric(model_names, "accuracy_top1")
    best_ar1   = _max_metric(ar_names, "ar_horizon_acc75_mean") if has_ar else float("nan")
    best_dist  = (_min_metric(dist_names, "ar_endpoint_dist_km_median")
                  if has_dist else float("nan"))

    def _ic_bits(name: str) -> float:
        v = results.get(name, {}).get("nll", float("nan"))
        return v / _LN2 if v == v else float("nan")

    _ic_vals = [v for v in (_ic_bits(n) for n in ic_names) if v == v]
    best_ic  = min(_ic_vals) if (_ic_vals and has_ic) else float("nan")

    def _ar_nll_bits(name: str) -> float:
        v = results.get(name, {}).get("ar_nll", float("nan"))
        return v / _LN2 if v == v else float("nan")

    _ar_nll_vals = [v for v in (_ar_nll_bits(n) for n in ar_nll_names) if v == v]
    best_ar_nll  = min(_ar_nll_vals) if (_ar_nll_vals and has_ar_nll) else float("nan")

    def _fmt_lower(value: float, best: float) -> str:
        if value == value and best == best and np.isclose(value, best, rtol=0.0, atol=1e-12):
            return rf"\textbf{{{value:.2f}}}"
        if value != value:
            return "--"
        return f"{value:.2f}"

    def _fmt_horizon(value: float, best: float, std: float = float("nan")) -> str:
        """Format the node-horizon (step count); bold the maximum (higher = better)."""
        if value != value:
            return "--"
        std_str = f" ({std:.2f})" if std == std else ""
        if best == best and np.isclose(value, best, rtol=0.0, atol=1e-12):
            return rf"\textbf{{{value:.2f}}}{std_str}"
        return f"{value:.2f}{std_str}"

    # Two-block layout: next-node metrics (Acc@1, NLL) on the left, then a
    # vertical bar, then the auto-regressive metrics (Node horizon, Endpoint
    # dist, NLL) on the right. The encoding column is dropped because every
    # reported model now shares the same (categorical-id) encoding.
    #   Model || Acc@1 [NLL] | Acc. Node Horizon [Endpoint dist] [NLL]
    nn_cols = "r" + ("r" if has_ic else "")          # Acc@1 (+ NLL)
    ar_cols = (("r" if has_ar else "")
               + ("r" if has_dist else "")
               + ("r" if has_ar_nll else ""))        # horizon (+ dist) (+ NLL)
    col_spec = "l" + nn_cols + ("|" + ar_cols if ar_cols else "")

    n_nn = len(nn_cols)
    n_ar = len(ar_cols)

    # Group header row spanning each task block.
    group_cells = [r"\multicolumn{1}{c}{}"]
    group_cells.append(
        rf"\multicolumn{{{n_nn}}}{{c{'|' if n_ar else ''}}}{{\textbf{{Next-node prediction}}}}")
    if n_ar:
        group_cells.append(
            rf"\multicolumn{{{n_ar}}}{{c}}{{\textbf{{Auto-regressive rollout}}}}")
    group_header = " & ".join(group_cells) + r" \\"

    # Per-metric header row.
    nn_hdr_cells = [r"\textbf{Acc@1}$\uparrow$"]
    if has_ic:
        nn_hdr_cells.append(r"\textbf{NLL (bits)}$\downarrow$")
    ar_hdr_cells = []
    if has_ar:
        ar_hdr_cells.append(r"\textbf{Acc. Node Horizon}$\uparrow$")
    if has_dist:
        ar_hdr_cells.append(r"\textbf{Endpoint dist (km)}$\downarrow$")
    if has_ar_nll:
        ar_hdr_cells.append(r"\textbf{NLL (bits)}$\downarrow$")
    header = (
        " & ".join([r"\textbf{Model}", *nn_hdr_cells, *ar_hdr_cells])
        + r" \\"
    )
    caption = (
        r"\caption{Prediction accuracy and certainty for the next-node and "
        r"auto-regressive tasks, averaged over 4 datasets. The Acceptable Node "
        r"Horizon is the mean number of consecutive future nodes predicted in "
        r"auto-regressive rollout while cumulative top-1 accuracy stays "
        r"$\geq 75\%$ (higher $=$ longer reliable prediction). NLL is the mean "
        r"negative log-likelihood of the realised next node (bits, lower $=$ "
        r"more certain).}"
    )

    lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        caption,
        r"\label{tab:comparison_all}",
        rf"\resizebox{{\linewidth}}{{!}}{{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        group_header,
        header,
        r"\midrule",
    ]

    prev_group = None
    for name in model_names:
        row = results.get(name, {})
        grp = model_group(name)
        if prev_group is not None and grp != prev_group:
            lines.append(r"\hline")

        nn_part = _fmt_metric(
            row.get('accuracy_top1', float('nan')), best_acc1,
            row.get('accuracy_top1_std', float('nan')))

        ar_part = ""
        if has_ar:
            ar_part = " & " + _fmt_horizon(
                row.get('ar_horizon_acc75_mean', float('nan')), best_ar1,
                row.get('ar_horizon_acc75_mean_std', float('nan')))

        dist_part = ""
        if has_dist:
            dist_part = " & " + _fmt_min_metric(
                row.get('ar_endpoint_dist_km_median', float('nan')), best_dist,
                row.get('ar_endpoint_dist_km_median_std', float('nan')))

        ic_part = ""
        if has_ic:
            _nll = row.get('nll', float('nan'))
            _ic  = _nll / _LN2 if _nll == _nll else float('nan')
            _ic_std = row.get('nll_std', float('nan'))
            _ic_std = _ic_std / _LN2 if _ic_std == _ic_std else float('nan')
            ic_part = " & " + _fmt_min_metric(_ic, best_ic, _ic_std)

        ar_nll_part = ""
        if has_ar_nll:
            _arn = row.get('ar_nll', float('nan'))
            _arn_bits = _arn / _LN2 if _arn == _arn else float('nan')
            _arn_std = row.get('ar_nll_std', float('nan'))
            _arn_std = _arn_std / _LN2 if _arn_std == _arn_std else float('nan')
            ar_nll_part = " & " + _fmt_min_metric(_arn_bits, best_ar_nll, _arn_std)

        lines.append(
            f"{_latex_escape(model_label(name))} & "
            f"{nn_part}{ic_part}{ar_part}{dist_part}{ar_nll_part} \\\\"
        )
        prev_group = grp

    lines.extend([r"\bottomrule", r"\end{tabular}", r"}%  end resizebox", r"\end{table}"])

    combined_path = os.path.join(output_dir, "comparison_table_next_node.tex")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {combined_path}")

    # Keep the AR file as a stub so downstream includes don't break
    ar_path = os.path.join(output_dir, "comparison_table_autoregressive.tex")
    with open(ar_path, "w", encoding="utf-8") as f:
        f.write("% Merged into comparison_table_next_node.tex (Table~\\ref{tab:comparison_all}).\n")
    print(f"  Saved → {ar_path}")


# ---------------------------------------------------------------------------
# Box-plot figure builder
# ---------------------------------------------------------------------------

def _make_comparison_boxplot(all_results: Dict[str, List[Dict]],
                             metric: str, label: str,
                             names: List[str],
                             n_proposed: int, n_baselines: int):
    """
    Build a single-metric box-plot figure with per-group colours and separators.

    Layout (left → right):
        [GraphIDyOM variants]  |dotted|  [other baselines]  |dotted|  [MC1 MC5 MC10]

    Groups get distinct fill colours; within each group models are sorted by
    Markov order first, then alphabetically.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    GROUP_COLORS = ["#4ECDC4", "#45B7D1", "#FF6B6B"]
    n_mc = len(names) - n_proposed - n_baselines

    def _group_color(i: int) -> str:
        if i < n_proposed:
            return GROUP_COLORS[0]
        elif i < n_proposed + n_baselines:
            return GROUP_COLORS[1]
        else:
            return GROUP_COLORS[2]

    data = []
    for n in names:
        splits = all_results.get(n, [{}])
        vals = [s.get(metric, float("nan")) for s in splits]
        vals = [v for v in vals if v == v]   # drop NaN
        data.append(vals if vals else [0.0])

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.85), 4.5))

    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_group_color(i))
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    if n_proposed > 0 and len(names) > n_proposed:
        ax.axvline(n_proposed + 0.5, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    if n_baselines > 0 and n_mc > 0:
        ax.axvline(n_proposed + n_baselines + 0.5, color="black",
                   linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([_display_name(n) for n in names], rotation=35, ha="right")
    ax.set_ylabel(label, fontsize=13)
    all_vals = [v for d in data for v in d]
    if metric == "ic_bits":
        # Requested scaling for IC: start at 0 and end at the observed maximum.
        y_min = 0.0
        y_max = max(all_vals) if all_vals else 1.0
        if y_max <= 0.0:
            y_max = 1.0
    elif metric in ("ar_endpoint_hops_median", "dist_km",
                    "ar_endpoint_dist_km_median",
                    "ar_horizon_acc75_mean"):
        # Hops/kilometres/step counts can exceed 1; allow free upper bound.
        vmin, vmax = min(all_vals), max(all_vals)
        pad = max(0.5, 0.05 * (vmax - vmin + 1))
        y_min = max(0.0, vmin - pad)
        y_max = vmax + pad
    else:
        y_min = max(0.0, min(all_vals) - 0.03)
        y_max = min(1.0, max(all_vals) + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

    legend_items = []
    if n_proposed > 0:
        legend_items.append(mpatches.Patch(facecolor=GROUP_COLORS[0], label="Proposed (GraphIDyOM)"))
    if n_baselines > 0:
        legend_items.append(mpatches.Patch(facecolor=GROUP_COLORS[1], label="Baselines"))
    if n_mc > 0:
        legend_items.append(mpatches.Patch(facecolor=GROUP_COLORS[2], label="Simple Markov"))
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", framealpha=0.85, fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main plot functions (called by train_and_evaluate.py and standalone)
# ---------------------------------------------------------------------------

def plot_comparison(results: Dict[str, Dict], output_dir: str,
                    all_split_results: Optional[Dict[str, List[Dict]]] = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        print("  matplotlib not available – skipping plot.")
        return

    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10, "figure.dpi": 150,
    })

    results = _drop_graph_embedding(results)
    # Show every model present in the comparison table (all encodings/orders
    # except graph-embedding); no best-per-family reduction.
    keep = set(results.keys())
    results = {k: v for k, v in results.items() if k in keep}
    if all_split_results is not None:
        all_split_results = {k: v for k, v in all_split_results.items() if k in keep}

    _mc      = {"MC2", "MC5", "MC10", "MC2P", "MC5P", "MC10P",
                "MC2_GE", "MC5_GE", "MC10_GE"}
    _other_b = {"AKOM", "CPT", "MOGEN", "IOHMM"}
    _all_b   = _mc | _other_b

    proposed   = _sort_group([n for n in results if n.upper() not in _all_b
                               and not any(n.upper().startswith(b) for b in _all_b)])
    other_base = _sort_group([n for n in results if n.upper() in _other_b])
    mc_base    = _sort_group([n for n in results if n.upper() in _mc])
    names      = proposed + other_base + mc_base

    n_proposed  = len(proposed)
    n_baselines = len(other_base)

    metrics_cfg = [
        ("accuracy_top1", "Acc@1",  "#4ECDC4"),
        ("accuracy_top3", "Acc@3",  "#45B7D1"),
        ("accuracy_top5", "Acc@5",  "#FFA07A"),
        ("mrr",           "MRR",    "#FF6B6B"),
    ]

    # --- combined multi-metric bar figure ---
    x     = np.arange(len(names))
    width = 0.18
    fig_all, ax_all = plt.subplots(figsize=(max(7, len(names) * 0.9), 4))
    for j, (metric, label, color) in enumerate(metrics_cfg):
        vals   = [results[n].get(metric, 0.0) for n in names]
        offset = (j - 1.5) * width
        bars   = ax_all.bar(x + offset, vals, width, label=label,
                            color=color, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax_all.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.004,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=5, rotation=90)

    if n_proposed > 0 and len(names) > n_proposed:
        ax_all.axvline(n_proposed - 0.5, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    if n_baselines > 0 and len(mc_base) > 0:
        ax_all.axvline(n_proposed + n_baselines - 0.5, color="black",
                       linestyle=":", linewidth=1.2, alpha=0.7)

    ax_all.set_xticks(x)
    ax_all.set_xticklabels([_display_name(n) for n in names], rotation=30, ha="right")
    ax_all.set_ylabel("Score")
    ax_all.legend(loc="upper right", framealpha=0.85)
    ax_all.set_ylim(0, min(1.08, max(
        r.get("accuracy_top5", 0) for r in results.values()) + 0.15))
    ax_all.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
    ax_all.set_axisbelow(True)
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(output_dir, "comparison.png"), bbox_inches="tight", dpi=200)
    fig_all.savefig(os.path.join(output_dir, "comparison.pdf"), bbox_inches="tight")
    print(f"  Saved → {os.path.join(output_dir, 'comparison.png')}")
    print(f"  Saved → {os.path.join(output_dir, 'comparison.pdf')}")
    plt.close(fig_all)

    # --- per-metric box-plots (if split data available) ---
    if all_split_results:
        # Derive IC in bits from NLL so it can be box-plotted directly.
        all_split_results_plot: Dict[str, List[Dict]] = {}
        for _name, _splits in all_split_results.items():
            _rows: List[Dict] = []
            for _row in _splits:
                _row2 = dict(_row)
                _nll = _row2.get("nll", float("nan"))
                if _nll == _nll:
                    _row2["ic_bits"] = float(_nll / np.log(2.0))
                _rows.append(_row2)
            all_split_results_plot[_name] = _rows

        has_ar_split = any(
            any("ar_accuracy_top1" in r for r in splits)
            for splits in all_split_results_plot.values()
        )
        has_endpoint_split = any(
            any("ar_endpoint_match" in r for r in splits)
            for splits in all_split_results_plot.values()
        )
        has_hops_split = any(
            any("ar_endpoint_hops_median" in r for r in splits)
            for splits in all_split_results_plot.values()
        )
        has_dist_next_split = any(
            any("dist_km" in r for r in splits)
            for splits in all_split_results_plot.values()
        )
        has_dist_ar_split = any(
            any("ar_endpoint_dist_km_median" in r for r in splits)
            for splits in all_split_results_plot.values()
        )
        has_ic_split = any(
            any("ic_bits" in r for r in splits)
            for splits in all_split_results_plot.values()
        )
        box_metrics = [("accuracy_top1", "Acc@1"), ("mrr", "MRR")]
        if has_ar_split:
            box_metrics.append(("ar_horizon_acc75_mean",
                                "Acceptable Node Horizon (steps, higher is better)"))
        if has_ic_split:
            box_metrics.append(("ic_bits", "NLL (bits, lower is better)"))
        if has_endpoint_split:
            box_metrics.append(("ar_endpoint_match", "Dist@1"))
            box_metrics.append(("ar_path_jaccard",   "Path Jaccard"))
        if has_hops_split:
            box_metrics.append(("ar_endpoint_hops_median",
                                "Endpoint hops (median, lower is better)"))
        if has_dist_next_split:
            box_metrics.append(("dist_km",
                                "Next-node distance (km, lower is better)"))
        if has_dist_ar_split:
            box_metrics.append(("ar_endpoint_dist_km_median",
                                "Endpoint distance (km, lower is better)"))
        slug_extra = {
            "ar_endpoint_match":        "endpoint",
            "ar_path_jaccard":          "jaccard",
            "ar_endpoint_hops_median":  "hops",
            "dist_km":                  "dist_next",
            "ar_endpoint_dist_km_median": "dist_ar",
            "ic_bits":                  "ic_bits",
        }
        for metric, label in box_metrics:
            fig = _make_comparison_boxplot(all_split_results_plot, metric, label,
                                           names, n_proposed, n_baselines)
            slug = {"accuracy_top1": "acc1", "mrr": "mrr",
                    "ar_horizon_acc75_mean": "ar_horizon"}.get(metric,
                                                  slug_extra.get(metric, metric))
            for ext in ("png", "pdf"):
                path = os.path.join(output_dir, f"comparison_{slug}.{ext}")
                fig.savefig(path, bbox_inches="tight", dpi=200)
                print(f"  Saved → {path}")
            plt.close(fig)


def plot_boxplots(all_results: Dict[str, List[Dict]], output_dir: str) -> None:
    """Box plots of Acc@1, MRR (and AR@1 if present) distributions over multiple random splits."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        print("  matplotlib not available – skipping box plots.")
        return

    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    _means = {n: {"accuracy_top1": float(np.mean([r.get("accuracy_top1", 0.0) for r in splits]))}
              for n, splits in all_results.items() if splits}
    _means = _drop_graph_embedding(_means)
    # Keep all models shown in the comparison table (no best-per-family filter).
    keep = set(_means.keys())
    all_results = {k: v for k, v in all_results.items() if k in keep}

    _baselines = {"AKOM", "CPT", "MOGEN", "IOHMM", 
                  "MC2", "MC5", "MC10", "MC2P", "MC5P", "MC10P",
                  "MC2_GE", "MC5_GE", "MC10_GE"}
    names = (sorted(n for n in all_results if n not in _baselines) +
             sorted(n for n in all_results if n in _baselines))
    n_splits = len(next(iter(all_results.values())))
    # Derive IC in bits from NLL for each split row.
    all_results_plot: Dict[str, List[Dict]] = {}
    for _name, _splits in all_results.items():
        _rows: List[Dict] = []
        for _row in _splits:
            _row2 = dict(_row)
            _nll = _row2.get("nll", float("nan"))
            if _nll == _nll:
                _row2["ic_bits"] = float(_nll / np.log(2.0))
            _rows.append(_row2)
        all_results_plot[_name] = _rows

    metrics  = [("accuracy_top1", "Acc@1"), ("mrr", "MRR")]
    has_ar = any(
        any("ar_accuracy_top1" in r for r in splits)
        for splits in all_results_plot.values()
    )
    has_ic = any(
        any("ic_bits" in r for r in splits)
        for splits in all_results_plot.values()
    )
    has_endpoint = any(
        any("ar_endpoint_match" in r for r in splits)
        for splits in all_results_plot.values()
    )
    has_hops = any(
        any("ar_endpoint_hops_median" in r for r in splits)
        for splits in all_results_plot.values()
    )
    has_dist_next = any(
        any("dist_km" in r for r in splits)
        for splits in all_results_plot.values()
    )
    has_dist_ar = any(
        any("ar_endpoint_dist_km_median" in r for r in splits)
        for splits in all_results_plot.values()
    )
    if has_ar:
        metrics.append(("ar_horizon_acc75_mean",
                        "Acceptable Node Horizon (steps, higher is better)"))
    if has_ic:
        metrics.append(("ic_bits", "NLL (bits, lower is better)"))
    if has_endpoint:
        metrics.append(("ar_endpoint_match", "Dist@1"))
        metrics.append(("ar_path_jaccard",  "Path Jaccard"))
    if has_hops:
        metrics.append(("ar_endpoint_hops_median",
                        "Endpoint hops (median, lower is better)"))
    if has_dist_next:
        metrics.append(("dist_km",
                        "Next-node distance (km, lower is better)"))
    if has_dist_ar:
        metrics.append(("ar_endpoint_dist_km_median",
                        "Endpoint distance (km, lower is better)"))

    cmap = plt.cm.tab20
    colors = [cmap(i / max(len(names), 1)) for i in range(len(names))]

    _slug_map = {
        "accuracy_top1":           "boxplot_acc1",
        "mrr":                     "boxplot_mrr",
        "ar_accuracy_top1":        "boxplot_ar",
        "ar_horizon_acc75_mean":   "boxplot_ar_horizon",
        "ic_bits":                 "boxplot_ic_bits",
        "ar_endpoint_match":       "boxplot_endpoint",
        "ar_path_jaccard":         "boxplot_jaccard",
        "ar_endpoint_hops_median": "boxplot_hops",
        "dist_km":                 "boxplot_dist_next",
        "ar_endpoint_dist_km_median": "boxplot_dist_ar",
    }
    saved_paths = []
    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.1), 5))
        data = [[r.get(metric, float("nan")) for r in all_results_plot[n]] for n in names]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([_display_name(n) for n in names], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        slug = _slug_map.get(metric, f"boxplot_{metric}")
        for ext in ("png", "pdf"):
            path = os.path.join(output_dir, f"{slug}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=200)
            saved_paths.append(path)
        plt.close(fig)
    for path in saved_paths:
        print(f"  Saved → {path}")


def plot_all_datasets_boxplots(
    combined_split_results: Dict[str, List[Dict]],
    output_dir: str,
) -> None:
    """
    One box per model, distribution over all datasets × splits (e.g. 4 × 5 = 20 points).

    Two figures are produced:
      boxplot_cross_acc1.{png,pdf}  – Acc@1  (next-node)
      boxplot_cross_ar1.{png,pdf}   – AR@1   (auto-regressive, only if data present)

    Vertical dotted separators distinguish proposed / baseline / simple-MC groups.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available – skipping cross-dataset box plots.")
        return

    mpl.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": 10, "figure.dpi": 150,
    })

    # Use the canonical ordering, restricted to models we actually have data for
    _means = {n: {"accuracy_top1": float(np.mean([r.get("accuracy_top1", 0.0) for r in splits]))}
              for n, splits in combined_split_results.items() if splits}
    _means = _drop_graph_embedding(_means)
    # Keep all models shown in the comparison table (no best-per-family filter).
    keep = set(_means.keys())
    combined_split_results = {k: v for k, v in combined_split_results.items() if k in keep}
    model_names = _ordered_model_names({n: {} for n in combined_split_results})
    model_names = [n for n in model_names if n in combined_split_results]
    if not model_names:
        return

    _mc_set       = {"MC2", "MC5", "MC10", "MC2P", "MC5P", "MC10P",
                     "MC2_GE", "MC5_GE", "MC10_GE"}
    _baseline_set = {"AKOM", "CPT", "MOGEN", "IOHMM"}
    n_proposed  = sum(1 for n in model_names if n.startswith("GraphIDyOM_"))
    n_baselines = sum(1 for n in model_names if n.upper() in _baseline_set)
    n_mc        = sum(1 for n in model_names if n.upper() in _mc_set)

    GROUP_COLORS = ["#4ECDC4", "#45B7D1", "#FF6B6B"]

    def _group_color(i: int) -> str:
        if i < n_proposed:
            return GROUP_COLORS[0]
        elif i < n_proposed + n_baselines:
            return GROUP_COLORS[1]
        return GROUP_COLORS[2]

    has_ar = any(
        any("ar_accuracy_top1" in r for r in splits)
        for splits in combined_split_results.values()
    )
    # Derive IC in bits from NLL for each split row.
    combined_plot: Dict[str, List[Dict]] = {}
    for _name, _splits in combined_split_results.items():
        _rows: List[Dict] = []
        for _row in _splits:
            _row2 = dict(_row)
            _nll = _row2.get("nll", float("nan"))
            if _nll == _nll:
                _row2["ic_bits"] = float(_nll / np.log(2.0))
            _rows.append(_row2)
        combined_plot[_name] = _rows

    has_ic = any(
        any("ic_bits" in r for r in splits)
        for splits in combined_plot.values()
    )
    has_endpoint = any(
        any("ar_endpoint_match" in r for r in splits)
        for splits in combined_plot.values()
    )
    has_hops = any(
        any("ar_endpoint_hops_median" in r for r in splits)
        for splits in combined_plot.values()
    )
    has_dist_next = any(
        any("dist_km" in r for r in splits)
        for splits in combined_plot.values()
    )
    has_dist_ar = any(
        any("ar_endpoint_dist_km_median" in r for r in splits)
        for splits in combined_plot.values()
    )
    metrics_to_plot = [("accuracy_top1", "Acc@1", "boxplot_cross_acc1")]
    if has_ar:
        metrics_to_plot.append(("ar_horizon_acc75_mean",
                                "Acceptable Node Horizon (steps, higher is better)",
                                "boxplot_cross_ar_horizon"))
    if has_ic:
        metrics_to_plot.append(("ic_bits", "NLL (bits, lower is better)",
                                "boxplot_cross_ic_bits"))
    if has_endpoint:
        metrics_to_plot.append(("ar_endpoint_match", "Dist@1",
                                "boxplot_cross_endpoint"))
        metrics_to_plot.append(("ar_path_jaccard",  "Path Jaccard",
                                "boxplot_cross_jaccard"))
    if has_hops:
        metrics_to_plot.append(("ar_endpoint_hops_median",
                                "Endpoint hops (median, lower is better)",
                                "boxplot_cross_hops"))
    if has_dist_next:
        metrics_to_plot.append(("dist_km",
                                "Next-node distance (km, lower is better)",
                                "boxplot_cross_dist_next"))
    if has_dist_ar:
        metrics_to_plot.append(("ar_endpoint_dist_km_median",
                                "Endpoint distance (km, lower is better)",
                                "boxplot_cross_dist_ar"))

    for metric, ylabel, slug in metrics_to_plot:
        data = []
        for name in model_names:
            vals = [s.get(metric, float("nan"))
                for s in combined_plot.get(name, [])]
            vals = [v for v in vals if v == v]  # drop NaN
            data.append(vals if vals else [0.0])

        fig, ax = plt.subplots(figsize=(max(7, len(model_names) * 0.85), 4.5))
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))

        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(_group_color(i))
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        # Vertical separators between groups
        if n_proposed > 0 and len(model_names) > n_proposed:
            ax.axvline(n_proposed + 0.5, color="black", linestyle=":",
                       linewidth=1.2, alpha=0.7)
        if n_baselines > 0 and n_mc > 0:
            ax.axvline(n_proposed + n_baselines + 0.5, color="black",
                       linestyle=":", linewidth=1.2, alpha=0.7)

        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels([_display_name(n) for n in model_names],
                           rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        all_vals = [v for d in data for v in d]
        if all_vals:
            if metric == "ic_bits":
                ax.set_ylim(0.0, max(all_vals) if max(all_vals) > 0 else 1.0)
            elif metric in ("ar_endpoint_hops_median", "dist_km",
                            "ar_endpoint_dist_km_median",
                            "ar_horizon_acc75_mean"):
                # Non-negative, can exceed 1 (hops, kilometres or step counts).
                # Don't clamp to [0,1]; pad the upper bound by ~5%% of the range.
                vmin, vmax = min(all_vals), max(all_vals)
                pad = max(0.5, 0.05 * (vmax - vmin + 1))
                ax.set_ylim(max(0.0, vmin - pad), vmax + pad)
            else:
                ax.set_ylim(max(0.0, min(all_vals) - 0.03),
                            min(1.0, max(all_vals) + 0.05))
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.6)
        ax.set_axisbelow(True)

        legend_items = []
        if n_proposed  > 0:
            legend_items.append(mpatches.Patch(facecolor=GROUP_COLORS[0], label="Proposed (DM-MC)"))
        if n_baselines > 0:
            legend_items.append(mpatches.Patch(facecolor=GROUP_COLORS[1], label="Baselines"))
        if n_mc > 0:
            legend_items.append(mpatches.Patch(facecolor=GROUP_COLORS[2], label="Controls"))
        if legend_items:
            if metric == "accuracy_top1":
                # Left side, centered on the y-axis for the Acc@1 figure.
                ax.legend(
                    handles=legend_items,
                    loc="center left",
                    bbox_to_anchor=(0.02, 0.5),
                    framealpha=0.85,
                    fontsize=9,
                )
            else:
                ax.legend(handles=legend_items, loc="upper right", framealpha=0.85, fontsize=9)

        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(output_dir, f"{slug}.{ext}")
            fig.savefig(path, bbox_inches="tight", dpi=200)
            print(f"  Saved → {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate plots from saved GraphIDyOM results JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default="results_unified",
                   help="Directory containing unified_results.json and/or "
                        "all_splits_results.json (produced by train_and_evaluate.py).")
    p.add_argument("--output-dir", default=None,
                   help="Where to write figures. Defaults to --results-dir.")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir  = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    # Prefer cross-dataset aggregate, then richer all_splits file; fall back to unified_results
    all_datasets_path = os.path.join(results_dir, "all_splits_results_all_datasets.json")
    all_splits_path  = os.path.join(results_dir, "all_splits_results.json")
    unified_path     = os.path.join(results_dir, "unified_results.json")

    all_split_results: Optional[Dict[str, List[Dict]]] = None
    results: Dict[str, Dict] = {}

    if os.path.exists(all_datasets_path):
        print(f"Loading {all_datasets_path} …")
        with open(all_datasets_path) as f:
            data = json.load(f)
        all_split_results = data.get("per_split_results", {})
        results           = data.get("mean_results", {})
        if not results and all_split_results:
            # Recompute means if missing
            for model_name, splits in all_split_results.items():
                float_keys = [k for k in splits[0] if isinstance(splits[0][k], (int, float))]
                results[model_name] = {
                    k: float(np.mean([s[k] for s in splits])) for k in float_keys
                }
    elif os.path.exists(all_splits_path):
        print(f"Loading {all_splits_path} …")
        with open(all_splits_path) as f:
            data = json.load(f)
        all_split_results = data.get("per_split_results", {})
        results           = data.get("mean_results", {})
        if not results and all_split_results:
            # Recompute means if missing
            for model_name, splits in all_split_results.items():
                float_keys = [k for k in splits[0] if isinstance(splits[0][k], (int, float))]
                results[model_name] = {
                    k: float(np.mean([s[k] for s in splits])) for k in float_keys
                }
    elif os.path.exists(unified_path):
        print(f"Loading {unified_path} …")
        with open(unified_path) as f:
            data = json.load(f)
        results = data.get("results", {})
    else:
        print(f"Error: no results JSON found in '{results_dir}'.")
        print("  Expected: all_splits_results_all_datasets.json, all_splits_results.json, or unified_results.json")
        return 1

    print(f"Found {len(results)} model entries: {', '.join(sorted(results))}")

    write_comparison_tables(results, output_dir)
    plot_comparison(results, output_dir, all_split_results=all_split_results)
    if all_split_results and len(next(iter(all_split_results.values()))) > 1:
        plot_boxplots(all_split_results, output_dir)

    # Cross-dataset box plots: scan sub-directories for per-dataset split results
    # and concatenate them so each box covers all datasets × splits.
    combined_split_results: Dict[str, List[Dict]] = {}
    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        ds_json = os.path.join(entry.path, "all_splits_results.json")
        if not os.path.exists(ds_json):
            continue
        with open(ds_json) as _f:
            ds_data = json.load(_f)
        for model_name, splits in ds_data.get("per_split_results", {}).items():
            if isinstance(splits, list) and splits:
                combined_split_results.setdefault(model_name, []).extend(splits)

    if combined_split_results:
        n_ds = len({entry.name
                    for entry in os.scandir(results_dir)
                    if entry.is_dir()
                    and os.path.exists(os.path.join(entry.path, "all_splits_results.json"))})
        print(f"  Cross-dataset box plots: {len(combined_split_results)} models, "
              f"{n_ds} dataset(s), up to "
              f"{max(len(v) for v in combined_split_results.values())} points/model")
        plot_all_datasets_boxplots(combined_split_results, output_dir)

    print(f"\n✓  All figures written to: {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
