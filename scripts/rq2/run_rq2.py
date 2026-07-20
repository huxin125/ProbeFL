"""RQ2: Ablation study — compare FULL vs NO_CONTEXT_BRIDGE vs NO_REFLECTION vs NO_DYNAMIC."""
import json
import math
import argparse
from pathlib import Path
import pandas as pd


PROJECT_ORDER = ["Chart", "Closure", "Lang", "Math", "Time"]
MODE_ORDER = ["ProbeFL_54_FULL", "ProbeFL_54_NO_CONTEXT_BRIDGE", "ProbeFL_54_NO_REFLECTION", "ProbeFL_54_NO_DYNAMIC"]
MODE_LABELS = {
    "ProbeFL_54_FULL": "FULL",
    "ProbeFL_54_NO_CONTEXT_BRIDGE": "w/o Context Bridge",
    "ProbeFL_54_NO_REFLECTION": "w/o Reflection",
    "ProbeFL_54_NO_DYNAMIC": "w/o Dynamic",
}


def load_scores(score_dir, mode_name, projects):
    """Load per-project scores and return flat per-bug rows."""
    score_dir = Path(score_dir)
    rows = []
    for proj in projects:
        path = score_dir / f"{proj}_{mode_name}_scores.json"
        if not path.exists():
            print(f"  [WARN] Missing: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        buggy_methods = data.get("buggy_methods", {})
        for bug_name, method_dict in buggy_methods.items():
            ranks = []
            for _, info in method_dict.items():
                if not isinstance(info, dict):
                    continue
                r = info.get("autofl_rank")
                if r is not None and r > 0:
                    ranks.append(r)

            if ranks:
                best_rank = min(ranks)
                top1 = int(best_rank <= 1)
                top3 = int(best_rank <= 3)
                top5 = int(best_rank <= 5)
                top10 = int(best_rank <= 10)
                rr = 1.0 / best_rank
            else:
                best_rank = None
                top1 = top3 = top5 = top10 = 0
                rr = 0.0

            rows.append({
                "mode": MODE_LABELS.get(mode_name, mode_name),
                "project": proj,
                "bug_name": bug_name,
                "rank": best_rank,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "top10": top10,
                "RR": rr,
            })
    return pd.DataFrame(rows)


def summarize_ablation(df):
    """Compute per-mode x per-project summary with deltas vs FULL."""
    rows = []
    modes = df["mode"].unique().tolist()

    # Get FULL baseline for delta computation
    full_df = df[df["mode"] == "FULL"]

    for mode in MODE_LABELS.values():
        if mode not in df["mode"].unique():
            continue
        mode_df = df[df["mode"] == mode]

        for proj in PROJECT_ORDER + ["Overall"]:
            if proj == "Overall":
                g = mode_df
                g_full = full_df
            else:
                g = mode_df[mode_df["project"] == proj]
                g_full = full_df[full_df["project"] == proj]

            if g.empty:
                continue

            total = len(g)
            top1 = int(g["top1"].sum())
            top3 = int(g["top3"].sum())
            top5 = int(g["top5"].sum())
            top10 = int(g["top10"].sum())
            mrr = float(g["RR"].fillna(0.0).mean())

            ranked = g["rank"].dropna()
            mfr = float(ranked.mean()) if len(ranked) > 0 else None

            # Delta vs FULL
            d_top1 = d_top3 = d_top5 = d_top10 = d_mfr = None
            if mode != "FULL" and not g_full.empty:
                g_f = g_full if proj != "Overall" else df[df["mode"] == "FULL"]
                if proj != "Overall":
                    g_f = g_f[g_f["project"] == proj]
                if not g_f.empty:
                    d_top1 = top1 - int(g_f["top1"].sum())
                    d_top3 = top3 - int(g_f["top3"].sum())
                    d_top5 = top5 - int(g_f["top5"].sum())
                    d_top10 = top10 - int(g_f["top10"].sum())
                    f_ranked = g_f["rank"].dropna()
                    f_mfr = float(f_ranked.mean()) if len(f_ranked) > 0 else None
                    if mfr is not None and f_mfr is not None:
                        d_mfr = mfr - f_mfr

            rows.append({
                "mode": mode,
                "project": proj,
                "total": total,
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "top10": top10,
                "MRR": mrr,
                "MFR": mfr,
                "d_top1": d_top1,
                "d_top3": d_top3,
                "d_top5": d_top5,
                "d_top10": d_top10,
                "d_mfr": d_mfr,
            })

    return pd.DataFrame(rows)


def generate_latex_table(summary_df):
    """Print LaTeX-formatted RQ2 ablation table."""
    modes = [m for m in MODE_LABELS.values() if m in summary_df["mode"].unique()]

    header = r"""\begin{table*}[t]
\caption{Ablation study results. $\Delta$ values show improvement of FULL over each variant.}
\label{tab:rq2_ablation}
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lccccccccccc@{}}
\toprule
\textbf{Mode} & \textbf{Defects} & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} & \textbf{Top-10} & \textbf{MFR}
& \textbf{$\Delta$Top-1} & \textbf{$\Delta$Top-3} & \textbf{$\Delta$Top-5} & \textbf{$\Delta$Top-10} & \textbf{$\Delta$MFR} \\
\midrule
"""
    rows_text = []

    for proj in PROJECT_ORDER + ["Overall"]:
        # FULL row
        full_row = summary_df[(summary_df["project"] == proj) & (summary_df["mode"] == "FULL")]
        if not full_row.empty:
            r = full_row.iloc[0]
            proj_label = f"\\textbf{{{proj}}}" if proj == "Overall" else proj
            if proj != "Overall":
                rows_text.append("\\addlinespace")
            rows_text.append(
                f"\\textbf{{{r['mode']}}} & {int(r['total'])} & {int(r['top1'])} & {int(r['top3'])} & {int(r['top5'])} & {int(r['top10'])} & {r['MFR']:.2f}"
                f" & -- & -- & -- & -- & -- \\\\"
            )

        # Ablation rows
        for mode in modes:
            if mode == "FULL":
                continue
            r = summary_df[(summary_df["project"] == proj) & (summary_df["mode"] == mode)]
            if r.empty:
                continue
            r = r.iloc[0]
            d1 = f"+{int(r['d_top1'])}" if r['d_top1'] and r['d_top1'] > 0 else str(int(r['d_top1'])) if r['d_top1'] is not None else "--"
            d3 = f"+{int(r['d_top3'])}" if r['d_top3'] and r['d_top3'] > 0 else str(int(r['d_top3'])) if r['d_top3'] is not None else "--"
            d5 = f"+{int(r['d_top5'])}" if r['d_top5'] and r['d_top5'] > 0 else str(int(r['d_top5'])) if r['d_top5'] is not None else "--"
            d10 = f"+{int(r['d_top10'])}" if r['d_top10'] and r['d_top10'] > 0 else str(int(r['d_top10'])) if r['d_top10'] is not None else "--"
            dm = f"+{r['d_mfr']:.2f}" if r['d_mfr'] is not None and r['d_mfr'] > 0 else f"{r['d_mfr']:.2f}" if r['d_mfr'] is not None else "--"

            rows_text.append(
                f"{r['mode']} & {int(r['total'])} & {int(r['top1'])} & {int(r['top3'])} & {int(r['top5'])} & {int(r['top10'])} & {r['MFR']:.2f}"
                f" & {d1} & {d3} & {d5} & {d10} & {dm} \\\\"
            )

        if proj == "Time":
            rows_text.append("\\midrule")

    footer = r"""\bottomrule
\end{tabular}%
}
\end{table*}
"""
    latex = header + "\n".join(rows_text) + "\n" + footer
    print(latex)
    return latex


def main():
    parser = argparse.ArgumentParser(description="RQ2 ablation study analysis.")
    parser.add_argument("--score-dir", default="scripts/rq1/outputs")
    parser.add_argument("--out-dir", default="scripts/rq2/outputs")
    parser.add_argument("--projects", nargs="+", default=PROJECT_ORDER)
    parser.add_argument("--modes", nargs="+", default=MODE_ORDER,
                        help="Mode names matching score files: {proj}_{mode}_scores.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for mode in args.modes:
        label = MODE_LABELS.get(mode, mode)
        print(f"Loading {label} ...")
        df = load_scores(args.score_dir, mode, args.projects)
        print(f"  {len(df)} bugs")
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)

    summary = summarize_ablation(full_df)

    # Print summary
    print("\n" + "=" * 80)
    print("RQ2 Ablation Summary")
    print("=" * 80)
    cols = ["mode", "project", "total", "top1", "top3", "top5", "top10", "MFR",
            "d_top1", "d_top3", "d_top5", "d_top10", "d_mfr"]
    print(summary[cols].to_string(index=False))

    # Save
    summary_path = out_dir / "rq2_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {summary_path}")

    # LaTeX
    latex = generate_latex_table(summary)
    latex_path = out_dir / "rq2_table.tex"
    latex_path.write_text(latex, encoding="utf-8")
    print(f"[Saved] {latex_path}")

    # Bar chart
    plot_ablation_bars(summary, out_dir)


def plot_ablation_bars(summary_df, out_dir):
    """Generate a grouped bar chart comparing 4 modes on Top-1/3/5/10."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    overall = summary_df[summary_df["project"] == "Overall"].copy()
    modes = ["FULL", "w/o Context Bridge", "w/o Reflection", "w/o Dynamic"]
    mode_labels = ["ProbeFL\n(FULL)", "w/o\nContext Bridge", "w/o\nReflection", "w/o\nDynamic"]
    metrics = ["top1", "top3", "top5", "top10"]
    metric_labels = ["Top-1", "Top-3", "Top-5", "Top-10"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(modes))
    width = 0.2
    colors = ["#1a3a5c", "#2a6aac", "#5a9ad4", "#a8c8e8"]

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        values = []
        for mode in modes:
            row = overall[overall["mode"] == mode]
            if not row.empty:
                values.append(int(row.iloc[0][metric]))
            else:
                values.append(0)
        bars = ax.bar(x + i * width, values, width, label=mlabel, color=colors[i],
                      edgecolor="white", linewidth=0.5)

        # Annotate values on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    str(val), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Number of Bugs (out of 353)", fontsize=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(mode_labels, fontsize=11)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.set_ylim(250, 325)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "rq2_ablation_bars.pdf"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {fig_path}")


if __name__ == "__main__":
    main()
