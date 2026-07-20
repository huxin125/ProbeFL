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

    plt.rcParams["font.size"] = 9.5
    plt.rcParams["axes.labelsize"] = 9.5
    plt.rcParams["axes.titlesize"] = 9.5
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise ValueError(
            "Cannot find any of the following columns: "
            + ", ".join(candidates)
            + "\nAvailable columns:\n"
            + ", ".join(df.columns)
        )

    return None


def to_numeric(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


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
    Compute the conventional upper whisker value used by boxplots:
    the largest value within Q3 + 1.5 * IQR.

    This is used only to set a reasonable x-axis range when fliers are hidden.
    """
    values = pd.Series(values).dropna()

    if values.empty:
        return 1.0

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return max(values.max(), 1.0)

    upper_fence = q3 + 1.5 * iqr
    non_outliers = values[values <= upper_fence]

    if non_outliers.empty:
        return max(values.max(), 1.0)

    return max(non_outliers.max(), 1.0)


def annotate_mean(ax, y_pos, mean_value, label_prefix="Avg"):
    """
    Annotate the mean value at the right side of the axis without changing
    the x-axis scale.

    x is in axis coordinates, y is in data coordinates.
    """
    ax.text(
        0.98,
        y_pos,
        f"{label_prefix}: {mean_value:.1f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        fontsize=8.8,
        color="#333333",
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
            alpha=0.75
        )
    )


def style_boxplot(bp, colors):
    """
    Apply consistent colors and alpha to boxplot patches.
    """
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)


def set_horizontal_boxplot_labels(ax, labels):
    """
    Set labels for a horizontal boxplot without using the deprecated
    'labels' parameter of matplotlib.axes.Axes.boxplot.

    For horizontal boxplots, categories are placed on the y-axis.
    """
    positions = list(range(1, len(labels) + 1))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)


def plot_overhead(df, out_pdf, out_png, show_fliers=False):
    """
    PingFL Fig.9-style overhead figure.

    Left subfigure:
        Static search time vs runtime probing time.

    Right subfigure:
        Main agent tokens vs runtime probing agent tokens.

    The default setting hides outliers to avoid extreme values compressing
    the boxes. Mean values are shown as text annotations instead of markers,
    so they do not affect the axis range.
    """

    static_time_col = find_col(df, [
        "static_search_time",
        "static_time",
        "static_time_seconds",
        "main_time",
        "main_time_seconds",
        "search_time",
        "search_time_seconds"
    ])

    probing_time_col = find_col(df, [
        "dynamic_verify_time",
        "probing_time",
        "probing_time_seconds",
        "runtime_probing_time",
        "runtime_probing_time_seconds",
        "dynamic_time",
        "dynamic_time_seconds",
        "verify_time_total"
    ])

    main_token_col = find_col(df, [
        "main_total_tokens",
        "main_tokens",
        "main_agent_tokens",
        "static_tokens",
        "search_tokens"
    ])

    probing_token_col = find_col(df, [
        "sub_total_tokens",
        "probing_tokens",
        "probing_agent_tokens",
        "runtime_probing_tokens",
        "dynamic_tokens"
    ])

    static_time = to_numeric(df, static_time_col)
    probing_time = to_numeric(df, probing_time_col)

    # Convert tokens to thousands.
    main_tokens = to_numeric(df, main_token_col) / 1000.0
    probing_tokens = to_numeric(df, probing_token_col) / 1000.0

    SINGLE_COL_WIDTH = 3.45
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL_WIDTH, 3.65))

    box_style = dict(
        patch_artist=True,
        showfliers=show_fliers,
        showmeans=False,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(
            marker="o",
            markersize=2.2,
            markerfacecolor="#777777",
            markeredgecolor="#777777",
            alpha=0.35
        )
    )

    colors = ["#4C78A8", "#F58518"]

    # -------------------------
    # (a) Time overhead
    # -------------------------
    ax = axes[0]

    bp = ax.boxplot(
        [static_time, probing_time],
        vert=False,
        **box_style
    )

    style_boxplot(bp, colors)
    set_horizontal_boxplot_labels(ax, ["Static", "Probing"])

    # If fliers are hidden, set x-axis by non-outlier whiskers.
    if not show_fliers:
        upper = max(
            non_outlier_upper(static_time),
            non_outlier_upper(probing_time)
        )
        ax.set_xlim(left=0, right=upper * 1.15)

    ax.set_xlabel("Time (s)")
    ax.set_title("(a) Time overhead")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    annotate_mean(ax, 1, static_time.mean(), label_prefix="Avg")
    annotate_mean(ax, 2, probing_time.mean(), label_prefix="Avg")

    # -------------------------
    # (b) Token overhead
    # -------------------------
    ax = axes[1]

    bp = ax.boxplot(
        [main_tokens, probing_tokens],
        vert=False,
        **box_style
    )

    style_boxplot(bp, colors)
    set_horizontal_boxplot_labels(ax, ["Main", "Probing"])

    # If fliers are hidden, set x-axis by non-outlier whiskers.
    if not show_fliers:
        upper = max(
            non_outlier_upper(main_tokens),
            non_outlier_upper(probing_tokens)
        )
        ax.set_xlim(left=0, right=upper * 1.15)

    ax.set_xlabel("Tokens (K)")
    ax.set_title("(b) Token overhead")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    annotate_mean(ax, 1, main_tokens.mean(), label_prefix="Avg")
    annotate_mean(ax, 2, probing_tokens.mean(), label_prefix="Avg")

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {out_pdf}")
    print(f"[Saved] {out_png}")

    print("\n[Summary]")
    print(f"Rows: {len(df)}")
    print(f"Static time mean: {static_time.mean():.2f}s")
    print(f"Probing time mean: {probing_time.mean():.2f}s")
    print(f"Main tokens mean: {main_tokens.mean():.2f}K")
    print(f"Probing tokens mean: {probing_tokens.mean():.2f}K")
    print(f"Show fliers: {show_fliers}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RQ3 time/token overhead boxplots for ProbeFL."
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
        "--show-fliers",
        action="store_true",
        help="Show outlier points in the boxplots. By default, outliers are hidden."
    )

    args = parser.parse_args()

    setup_matplotlib()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_bug)
    df = filter_valid_rows(df, drop_error_rows=args.drop_error_rows)

    if df.empty:
        raise ValueError("No rows left after filtering.")

    plot_overhead(
        df,
        out_pdf=os.path.join(args.out_dir, "rq3_overhead_boxplot.pdf"),
        out_png=os.path.join(args.out_dir, "rq3_overhead_boxplot.png"),
        show_fliers=args.show_fliers
    )


if __name__ == "__main__":
    main()
