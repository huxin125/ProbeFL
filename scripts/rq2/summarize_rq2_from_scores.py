import os
import json
import math
import argparse
from pathlib import Path

import pandas as pd


MODE_ORDER = [
    "ProbeFL_FULL",
    "ProbeFL_NO_CONTEXT_BRIDGE",
    "ProbeFL_NO_RUNTIME_FUSION",
    "ProbeFL_NO_REFLECTION",
]

MODE_DISPLAY = {
    "ProbeFL_FULL": "Full ProbeFL",
    "ProbeFL_NO_CONTEXT_BRIDGE": "w/o Context Bridge",
    "ProbeFL_NO_RUNTIME_FUSION": "w/o Runtime Fusion",
    "ProbeFL_NO_REFLECTION": "w/o Reflection",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_rank(x):
    if x is None:
        return None
    try:
        v = int(float(x))
        if v <= 0:
            return None
        return v
    except Exception:
        return None


def extract_project_from_bug_id(bug_id):
    bug_id = str(bug_id)
    if "_" in bug_id:
        return bug_id.split("_", 1)[0]
    return ""


def extract_per_bug(score_json, project, mode):
    """
    Extract best faulty-method rank for each bug.

    Expected structure from compute_score_verifl.py:
        data["buggy_methods"][bug_id][faulty_method]["autofl_rank"]
    """
    buggy_methods = score_json.get("buggy_methods", {})
    rows = []

    for bug_id, method_dict in buggy_methods.items():
        ranks = []

        if isinstance(method_dict, dict):
            for method, info in method_dict.items():
                if not isinstance(info, dict):
                    continue
                r = safe_rank(info.get("autofl_rank"))
                if r is not None:
                    ranks.append(r)

        if ranks:
            best_rank = min(ranks)
            avg_rank = sum(ranks) / len(ranks)
            mrr = 1.0 / best_rank
        else:
            best_rank = None
            avg_rank = None
            mrr = 0.0

        row = {
            "project": project,
            "bug_id": bug_id,
            "mode": mode,
            "mode_display": MODE_DISPLAY.get(mode, mode),
            "num_faulty_methods": len(ranks),
            "faulty_method_ranks": ";".join(str(x) for x in sorted(ranks)),
            "best_rank": best_rank,
            "avg_faulty_rank": avg_rank,
            "mrr": mrr,
        }

        for n in [1, 3, 5, 10]:
            row[f"top{n}"] = int(best_rank is not None and best_rank <= n)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_effectiveness(per_bug_df):
    rows = []

    for mode, g in per_bug_df.groupby("mode"):
        valid = g[g["best_rank"].notna()].copy()
        bugs = len(g)

        rows.append({
            "Mode": mode,
            "Display Mode": MODE_DISPLAY.get(mode, mode),
            "Bugs": bugs,

            "Top-1": int(g["top1"].sum()),
            "Top-3": int(g["top3"].sum()),
            "Top-5": int(g["top5"].sum()),
            "Top-10": int(g["top10"].sum()),

            "Top-1 Rate": g["top1"].mean() * 100 if bugs > 0 else 0.0,
            "Top-3 Rate": g["top3"].mean() * 100 if bugs > 0 else 0.0,
            "Top-5 Rate": g["top5"].mean() * 100 if bugs > 0 else 0.0,
            "Top-10 Rate": g["top10"].mean() * 100 if bugs > 0 else 0.0,

            "MFR": valid["best_rank"].mean() if len(valid) > 0 else math.nan,
            "Median FR": valid["best_rank"].median() if len(valid) > 0 else math.nan,
            "MAR": valid["avg_faulty_rank"].mean() if len(valid) > 0 else math.nan,
            "MRR": g["mrr"].mean() if bugs > 0 else 0.0,
        })

    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODE_ORDER)}
    df["_order"] = df["Mode"].map(lambda x: order.get(x, 999))
    df = df.sort_values(["_order", "Mode"]).drop(columns=["_order"])
    return df


def build_paired(per_bug_df, full_mode):
    """
    Compare Full ProbeFL with each ablation on the same bugs.

    rank_gain = ablation_rank - full_rank
      > 0: Full is better
      = 0: tie
      < 0: ablation is better
    """
    full = per_bug_df[per_bug_df["mode"] == full_mode].copy()
    if full.empty:
        raise ValueError(f"No rows found for full_mode={full_mode}")

    full = full[[
        "project", "bug_id", "best_rank", "top1", "top3", "top5", "top10", "mrr"
    ]].rename(columns={
        "best_rank": "full_rank",
        "top1": "full_top1",
        "top3": "full_top3",
        "top5": "full_top5",
        "top10": "full_top10",
        "mrr": "full_mrr",
    })

    paired_rows = []
    summary_rows = []

    modes = [m for m in MODE_ORDER if m != full_mode and m in per_bug_df["mode"].unique()]

    for mode in modes:
        ab = per_bug_df[per_bug_df["mode"] == mode].copy()

        ab = ab[[
            "project", "bug_id", "best_rank", "top1", "top3", "top5", "top10", "mrr"
        ]].rename(columns={
            "best_rank": "ablation_rank",
            "top1": "ablation_top1",
            "top3": "ablation_top3",
            "top5": "ablation_top5",
            "top10": "ablation_top10",
            "mrr": "ablation_mrr",
        })

        merged = pd.merge(
            full,
            ab,
            on=["project", "bug_id"],
            how="inner"
        )

        merged["ablation"] = mode
        merged["ablation_display"] = MODE_DISPLAY.get(mode, mode)

        cmp_df = merged[
            merged["full_rank"].notna() & merged["ablation_rank"].notna()
        ].copy()

        cmp_df["rank_gain"] = cmp_df["ablation_rank"] - cmp_df["full_rank"]
        cmp_df["full_better"] = (cmp_df["rank_gain"] > 0).astype(int)
        cmp_df["tie"] = (cmp_df["rank_gain"] == 0).astype(int)
        cmp_df["ablation_better"] = (cmp_df["rank_gain"] < 0).astype(int)

        cmp_df["top1_gain"] = cmp_df["full_top1"] - cmp_df["ablation_top1"]
        cmp_df["top3_gain"] = cmp_df["full_top3"] - cmp_df["ablation_top3"]
        cmp_df["top5_gain"] = cmp_df["full_top5"] - cmp_df["ablation_top5"]
        cmp_df["top10_gain"] = cmp_df["full_top10"] - cmp_df["ablation_top10"]
        cmp_df["mrr_gain"] = cmp_df["full_mrr"] - cmp_df["ablation_mrr"]

        paired_rows.append(cmp_df)

        summary_rows.append({
            "Ablation": mode,
            "Display Ablation": MODE_DISPLAY.get(mode, mode),
            "Comparable Bugs": len(cmp_df),

            "Full Better": int(cmp_df["full_better"].sum()),
            "Tie": int(cmp_df["tie"].sum()),
            "Ablation Better": int(cmp_df["ablation_better"].sum()),

            "Avg Rank Gain": cmp_df["rank_gain"].mean() if len(cmp_df) > 0 else math.nan,
            "Median Rank Gain": cmp_df["rank_gain"].median() if len(cmp_df) > 0 else math.nan,

            "Top-1 Gain": int(cmp_df["top1_gain"].sum()) if len(cmp_df) > 0 else 0,
            "Top-3 Gain": int(cmp_df["top3_gain"].sum()) if len(cmp_df) > 0 else 0,
            "Top-5 Gain": int(cmp_df["top5_gain"].sum()) if len(cmp_df) > 0 else 0,
            "Top-10 Gain": int(cmp_df["top10_gain"].sum()) if len(cmp_df) > 0 else 0,

            "Avg MRR Gain": cmp_df["mrr_gain"].mean() if len(cmp_df) > 0 else math.nan,
        })

    paired_per_bug = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    paired_summary = pd.DataFrame(summary_rows)

    return paired_summary, paired_per_bug


def fmt_int(x):
    return f"{int(round(float(x)))}"


def fmt_float(x, digits=2):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_pct(x, digits=1):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}\\%"


def latex_escape(s):
    s = str(s)
    return (
        s.replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
         .replace("#", "\\#")
    )


def make_effectiveness_table(df, use_rates=False):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Ablation results of ProbeFL.}")
    lines.append("\\label{tab:rq2_ablation_effectiveness}")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Mode} & \\textbf{Bugs} & \\textbf{Top-1} & "
        "\\textbf{Top-3} & \\textbf{Top-5} & \\textbf{Top-10} & "
        "\\textbf{MFR} \\\\"
    )
    lines.append("\\midrule")

    for _, r in df.iterrows():
        mode = latex_escape(r["Display Mode"])
        bugs = fmt_int(r["Bugs"])

        if use_rates:
            top1 = fmt_pct(r["Top-1 Rate"])
            top3 = fmt_pct(r["Top-3 Rate"])
            top5 = fmt_pct(r["Top-5 Rate"])
            top10 = fmt_pct(r["Top-10 Rate"])
        else:
            top1 = fmt_int(r["Top-1"])
            top3 = fmt_int(r["Top-3"])
            top5 = fmt_int(r["Top-5"])
            top10 = fmt_int(r["Top-10"])

        mfr = fmt_float(r["MFR"], 2)

        lines.append(
            f"{mode} & {bugs} & {top1} & {top3} & {top5} & {top10} & {mfr} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def make_paired_table(df):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\caption{Paired comparison between full ProbeFL and ablated variants.}")
    lines.append("\\label{tab:rq2_paired_ablation}")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lrrrrr@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Ablation} & \\textbf{Bugs} & \\textbf{Full Better} & "
        "\\textbf{Tie} & \\textbf{Abl. Better} & \\textbf{Avg. Gain} \\\\"
    )
    lines.append("\\midrule")

    for _, r in df.iterrows():
        ab = latex_escape(r["Display Ablation"])
        lines.append(
            f"{ab} & "
            f"{fmt_int(r['Comparable Bugs'])} & "
            f"{fmt_int(r['Full Better'])} & "
            f"{fmt_int(r['Tie'])} & "
            f"{fmt_int(r['Ablation Better'])} & "
            f"{fmt_float(r['Avg Rank Gain'], 2)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize RQ2 ablation results from score JSON files."
    )

    parser.add_argument(
        "--projects",
        nargs="+",
        required=True,
        help="Projects to summarize, e.g., Chart Closure Lang Math Time."
    )

    parser.add_argument(
        "--score-dir",
        default="./scripts/rq2/scores",
        help="Directory containing *_scores.json files."
    )

    parser.add_argument(
        "--out-dir",
        default="./scripts/rq2",
        help="Output directory."
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODE_ORDER,
        help="Modes to load."
    )

    parser.add_argument(
        "--full-mode",
        default="ProbeFL_FULL",
        help="Full mode used for paired comparison."
    )

    parser.add_argument(
        "--use-rate-table",
        action="store_true",
        help="Use percentages instead of counts in the LaTeX effectiveness table."
    )

    args = parser.parse_args()

    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    csv_dir = out_dir / "csv"
    table_dir = out_dir / "tables"

    csv_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for project in args.projects:
        for mode in args.modes:
            path = score_dir / f"{project}_{mode}_scores.json"

            if not path.exists():
                print(f"[WARN] Missing score file: {path}")
                continue

            data = load_json(path)
            df = extract_per_bug(data, project=project, mode=mode)
            all_rows.append(df)

            print(f"[Loaded] {path}: {len(df)} bugs")

    if not all_rows:
        raise ValueError("No score files loaded.")

    per_bug_df = pd.concat(all_rows, ignore_index=True)

    effectiveness = summarize_effectiveness(per_bug_df)
    paired_summary, paired_per_bug = build_paired(per_bug_df, full_mode=args.full_mode)

    # Save CSV files.
    per_bug_path = csv_dir / "rq2_per_bug_ranks.csv"
    effectiveness_path = csv_dir / "rq2_effectiveness_by_mode.csv"
    paired_summary_path = csv_dir / "rq2_paired_summary.csv"
    paired_per_bug_path = csv_dir / "rq2_paired_per_bug.csv"

    per_bug_df.to_csv(per_bug_path, index=False, encoding="utf-8-sig")
    effectiveness.to_csv(effectiveness_path, index=False, encoding="utf-8-sig")
    paired_summary.to_csv(paired_summary_path, index=False, encoding="utf-8-sig")
    paired_per_bug.to_csv(paired_per_bug_path, index=False, encoding="utf-8-sig")

    # Save LaTeX tables.
    effectiveness_tex = make_effectiveness_table(
        effectiveness,
        use_rates=args.use_rate_table
    )
    paired_tex = make_paired_table(paired_summary)

    effectiveness_tex_path = table_dir / "rq2_effectiveness_table.tex"
    paired_tex_path = table_dir / "rq2_paired_table.tex"

    with open(effectiveness_tex_path, "w", encoding="utf-8") as f:
        f.write(effectiveness_tex)

    with open(paired_tex_path, "w", encoding="utf-8") as f:
        f.write(paired_tex)

    print("\n[Saved CSV]")
    print(per_bug_path)
    print(effectiveness_path)
    print(paired_summary_path)
    print(paired_per_bug_path)

    print("\n[Saved LaTeX]")
    print(effectiveness_tex_path)
    print(paired_tex_path)

    print("\n[Effectiveness]")
    print(effectiveness[[
        "Display Mode", "Bugs", "Top-1", "Top-3", "Top-5", "Top-10", "MFR", "MRR"
    ]].to_string(index=False))

    print("\n[Paired Summary]")
    print(paired_summary[[
        "Display Ablation", "Comparable Bugs", "Full Better", "Tie",
        "Ablation Better", "Avg Rank Gain", "Top-1 Gain", "Top-5 Gain"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
