import os
import argparse

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


def setup_matplotlib():
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["text.usetex"] = False

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "DejaVu Serif"
    ]

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 8.5


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(df.columns)
        )


def get_num(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def filter_valid_rows(df, drop_error_rows=False):
    """
    By default, keep all rows. If --drop-error-rows is used, rows with
    error_occurred=True or error_occurred=1 are removed.
    """
    if not drop_error_rows:
        return df

    if "error_occurred" not in df.columns:
        return df

    s = df["error_occurred"]

    mask_error = (
        (s == True) |
        (s == 1) |
        (s.astype(str).str.lower().isin(["true", "1", "yes"]))
    )

    return df[~mask_error].copy()


def non_outlier_upper(values):
    """
    Compute the conventional upper whisker:
    max value within Q3 + 1.5 * IQR.

    This is used only for setting a readable x-axis range when long tails exist.
    """
    values = pd.Series(values).dropna()

    if values.empty:
        return 1.0

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return max(float(values.max()), 1.0)

    upper_fence = q3 + 1.5 * iqr
    non_outliers = values[values <= upper_fence]

    if non_outliers.empty:
        return max(float(values.max()), 1.0)

    return max(float(non_outliers.max()), 1.0)


def percentile_upper(values, q=0.95):
    values = pd.Series(values).dropna()

    if values.empty:
        return 1.0

    return max(float(values.quantile(q)), 1.0)


def build_dynamic_tool_data(df, merge_rare=False, debugger_only=False):
    """
    Build per-bug dynamic tool-call distributions.

    Default behavior:
        Use exact CSV column names as plot labels.

    If debugger_only=False, include:
        jdb_get_locals
        jdb_print_variable
        jdb_run_to_breakpoint
        jdb_where
        jdb_step
        jdb_next
        jdb_eval
        jdb_list
        submit_verification_report

    If debugger_only=True, include only jdb_* debugger actions.

    If merge_rare=True:
        Merge low-frequency JDB actions into:
            other_jdb_actions =
                jdb_where + jdb_step + jdb_next + jdb_eval + jdb_list

        The kept categories are:
            jdb_get_locals
            jdb_print_variable
            jdb_run_to_breakpoint
            other_jdb_actions
            submit_verification_report  # only if debugger_only=False
    """

    jdb_core_cols = [
        "jdb_get_locals",
        "jdb_print_variable",
        "jdb_run_to_breakpoint",
    ]

    jdb_rare_cols = [
        "jdb_where",
        "jdb_step",
        "jdb_next",
        "jdb_eval",
        "jdb_list",
    ]

    submit_col = "submit_verification_report"

    required_cols = jdb_core_cols + jdb_rare_cols
    if not debugger_only:
        required_cols = required_cols + [submit_col]

    require_columns(df, required_cols)

    data = []

    if merge_rare:
        # Put major actions first.
        ordered_core_cols = [
            "jdb_print_variable",
            "jdb_get_locals",
            "jdb_run_to_breakpoint",
        ]

        for col in ordered_core_cols:
            data.append((col, get_num(df, col)))

        # Put report submission before the residual category.
        if not debugger_only:
            data.append((submit_col, get_num(df, submit_col)))

        # Put the merged residual category at the end.
        other_jdb_actions = pd.Series([0.0] * len(df), index=df.index)
        for col in jdb_rare_cols:
            other_jdb_actions = other_jdb_actions + get_num(df, col)

        data.append(("other_jdb_actions", other_jdb_actions))


    else:
        # Use exact CSV column names as labels.
        all_cols = jdb_core_cols + jdb_rare_cols
        if not debugger_only:
            all_cols = all_cols + [submit_col]

        for col in all_cols:
            data.append((col, get_num(df, col)))

    labels = [x[0] for x in data]
    values = [x[1] for x in data]

    summary_rows = []
    for label, s in data:
        summary_rows.append({
            "tool": label,
            "total": float(s.sum()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "nonzero_bugs": int((s > 0).sum()),
            "nonzero_rate": float((s > 0).mean() * 100.0)
        })

    summary_df = pd.DataFrame(summary_rows)

    return labels, values, summary_df

def pretty_tool_label(label):
    mapping = {
        "jdb_print_variable": "Print variable",
        "jdb_get_locals": "Get locals",
        "jdb_run_to_breakpoint": "Run to breakpoint",
        "submit_verification_report": "Submit report",
        "other_jdb_actions": "Other JDB\nactions",
        "jdb_where": "Where",
        "jdb_step": "Step",
        "jdb_next": "Next",
        "jdb_eval": "Eval",
        "jdb_list": "List",
    }
    return mapping.get(label, label)


def annotate_stat(ax, y_pos, mean_value, median_value):
    """
    Annotate mean and median on the right side of the axis without changing
    the x-axis scale.
    """
    ax.text(
        0.985,
        y_pos,
        f"avg={mean_value:.1f}, med={median_value:.1f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8.5,
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            edgecolor="none",
            alpha=0.75
        )
    )


def plot_violin(labels, values, out_pdf, out_png, xlim_mode="whisker"):
    """
    Horizontal violin plot with an inner boxplot.

    xlim_mode:
        whisker: set x-axis by non-outlier whisker range.
        p95    : set x-axis by 95th percentile.
        full   : show full range.
    """

    if len(labels) == 0:
        raise ValueError("No data to plot.")

    positions = list(range(1, len(labels) + 1))

    SINGLE_COL_WIDTH = 3.45
    fig_height = max(3.0, 0.52 * len(labels) + 1.20)
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, fig_height))


    # Violin plot.
    parts = ax.violinplot(
        values,
        positions=positions,
        vert=False,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    colors = [
        "#4C78A8",
        "#72B7B2",
        "#F58518",
        "#B279A2",
        "#BAB0AC",
        "#54A24B",
        "#E45756",
        "#9D755D",
        "#EECA3B",
    ]

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_edgecolor("black")
        body.set_linewidth(0.7)
        body.set_alpha(0.62)

    # Inner boxplot. Do not use labels/tick_labels to avoid Matplotlib
    # version compatibility warnings.
    ax.boxplot(
        values,
        positions=positions,
        vert=False,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.75
        ),
        whiskerprops=dict(color="black", linewidth=0.8),
        capprops=dict(color="black", linewidth=0.8)
    )

    # Use exact labels.
    ax.set_yticks(positions)
    ax.set_yticklabels([pretty_tool_label(x) for x in labels])
    ax.invert_yaxis()

    # X-axis range.
    if xlim_mode == "whisker":
        upper = max(non_outlier_upper(v) for v in values)
        ax.set_xlim(left=0, right=max(upper * 1.20, 1.0))
    elif xlim_mode == "p95":
        upper = max(percentile_upper(v, q=0.95) for v in values)
        ax.set_xlim(left=0, right=max(upper * 1.20, 1.0))
    elif xlim_mode == "full":
        upper = max(float(pd.Series(v).max()) for v in values)
        ax.set_xlim(left=0, right=max(upper * 1.08, 1.0))
    else:
        raise ValueError(f"Unknown xlim_mode: {xlim_mode}")

    # Annotate average and median.
    for pos, s in zip(positions, values):
        s = pd.Series(s)
        annotate_stat(
            ax,
            pos,
            mean_value=float(s.mean()),
            median_value=float(s.median())
        )

    ax.set_xlabel("Tool calls per bug")
    ax.set_ylabel("Dynamic tool")

    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_pdf}")
    print(f"[Saved] {out_png}")


def check_debugger_total_consistency(df):
    """
    Optional sanity check:
    compare debugger_action_total with the sum of jdb_* columns.
    """
    if "debugger_action_total" not in df.columns:
        return

    jdb_cols = [
        "jdb_get_locals",
        "jdb_print_variable",
        "jdb_run_to_breakpoint",
        "jdb_where",
        "jdb_step",
        "jdb_next",
        "jdb_eval",
        "jdb_list",
    ]

    require_columns(df, jdb_cols)

    jdb_sum = pd.Series([0.0] * len(df), index=df.index)
    for col in jdb_cols:
        jdb_sum = jdb_sum + get_num(df, col)

    total = get_num(df, "debugger_action_total")
    diff = total - jdb_sum

    print("\n[Debugger total consistency check]")
    print(f"debugger_action_total mean: {total.mean():.2f}")
    print(f"sum(jdb_*) mean: {jdb_sum.mean():.2f}")
    print(f"mean difference: {diff.mean():.2f}")
    print(f"max absolute difference: {diff.abs().max():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RQ3 dynamic tool-call violin plot for ProbeFL."
    )

    parser.add_argument(
        "--per-bug",
        default="./rq3/probefl_rq3_full_per_bug_details.csv",
        help="Input per-bug RQ3 CSV."
    )

    parser.add_argument(
        "--out-dir",
        default="./rq3/figures",
        help="Output directory."
    )

    parser.add_argument(
        "--drop-error-rows",
        action="store_true",
        help="Drop rows where error_occurred is true."
    )

    parser.add_argument(
        "--debugger-only",
        action="store_true",
        help=(
            "Only plot jdb_* debugger actions. "
            "By default, submit_verification_report is also included."
        )
    )

    parser.add_argument(
        "--merge-rare",
        action="store_true",
        help=(
            "Merge jdb_where, jdb_step, jdb_next, jdb_eval, and jdb_list "
            "into other_jdb_actions."
        )
    )

    parser.add_argument(
        "--xlim-mode",
        choices=["whisker", "p95", "full"],
        default="whisker",
        help=(
            "How to set x-axis range. "
            "'whisker' uses non-outlier whisker range; "
            "'p95' uses 95th percentile; "
            "'full' shows full range."
        )
    )

    args = parser.parse_args()

    setup_matplotlib()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_bug)
    df = filter_valid_rows(df, drop_error_rows=args.drop_error_rows)

    if df.empty:
        raise ValueError("No rows left after filtering.")

    labels, values, summary_df = build_dynamic_tool_data(
        df,
        merge_rare=args.merge_rare,
        debugger_only=args.debugger_only
    )

    if args.debugger_only:
        base_name = "rq3_jdb_tool_violin"
    else:
        base_name = "rq3_dynamic_tool_violin"

    if args.merge_rare:
        base_name += "_merged"

    out_pdf = os.path.join(args.out_dir, f"{base_name}.pdf")
    out_png = os.path.join(args.out_dir, f"{base_name}.png")
    out_csv = os.path.join(args.out_dir, f"{base_name}_summary.csv")

    plot_violin(
        labels=labels,
        values=values,
        out_pdf=out_pdf,
        out_png=out_png,
        xlim_mode=args.xlim_mode
    )

    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[Saved] {out_csv}")

    print("\n[Plotted tools]")
    print(summary_df.to_string(index=False))

    print("\n[Settings]")
    print(f"Rows: {len(df)}")
    print(f"debugger_only: {args.debugger_only}")
    print(f"merge_rare: {args.merge_rare}")
    print(f"xlim_mode: {args.xlim_mode}")

    check_debugger_total_consistency(df)


if __name__ == "__main__":
    main()
