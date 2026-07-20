import os
import argparse
import pandas as pd


PROJECT_ORDER = [
    "Chart",
    "Closure",
    "Lang",
    "Math",
    "Mockito",
    "Time",
    "Cli",
    "Codec",
    "Collections",
    "Compress",
    "Csv",
    "Gson",
    "JacksonCore",
    "JacksonDatabind",
    "JacksonXml",
    "Jsoup",
]


REQUIRED_COLUMNS = [
    "project",
    "bug_id",
    "verification_attempts_total",
    "success_attempts",
    "submit_verification_report",
    "debugger_action_total",
    "dynamic_verify_time",
]


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


def safe_div(num, den, default=0.0):
    if den == 0:
        return default
    return num / den


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


def count_bugs(group, use_unique_bugs=False):
    """
    The input file is expected to be a per-bug CSV. By default, we use the
    number of rows as the number of processed bug-level runs.

    If --use-unique-bugs is set, we count unique bug_id values instead.
    """
    if use_unique_bugs and "bug_id" in group.columns:
        return group["bug_id"].nunique()
    return len(group)


def summarize_project(group, use_unique_bugs=False):
    bugs = count_bugs(group, use_unique_bugs=use_unique_bugs)

    attempts_series = get_num(group, "verification_attempts_total")
    success_series = get_num(group, "success_attempts")
    report_series = get_num(group, "submit_verification_report")
    debugger_action_series = get_num(group, "debugger_action_total")
    dynamic_time_series = get_num(group, "dynamic_verify_time")

    total_attempts = float(attempts_series.sum())
    total_success = float(success_series.sum())
    total_reports = float(report_series.sum())
    total_debugger_actions = float(debugger_action_series.sum())
    total_dynamic_time = float(dynamic_time_series.sum())

    invoked_bugs = int((attempts_series > 0).sum())

    invoked_rate = safe_div(invoked_bugs, bugs) * 100.0
    attempts_per_bug = safe_div(total_attempts, bugs)
    report_rate = safe_div(total_reports, total_attempts) * 100.0
    success_rate = safe_div(total_success, total_attempts) * 100.0
    actions_per_attempt = safe_div(total_debugger_actions, total_attempts)
    time_per_attempt = safe_div(total_dynamic_time, total_attempts)
    time_per_bug = safe_div(total_dynamic_time, bugs)

    return {
        "Project": group["project"].iloc[0],
        "Bugs": int(bugs),
        "Invoked Bugs": int(invoked_bugs),
        "Invoked Rate": invoked_rate,
        "Total Attempts": total_attempts,
        "Attempts/Bug": attempts_per_bug,
        "Submitted Reports": total_reports,
        "Report Rate": report_rate,
        "Successful Attempts": total_success,
        "Success Rate": success_rate,
        "Debugger Actions": total_debugger_actions,
        "Actions/Attempt": actions_per_attempt,
        "Dynamic Time": total_dynamic_time,
        "Time/Bug": time_per_bug,
        "Time/Attempt": time_per_attempt,
    }


def summarize_all(df, use_unique_bugs=False):
    rows = []

    for project, group in df.groupby("project"):
        rows.append(summarize_project(group, use_unique_bugs=use_unique_bugs))

    summary = pd.DataFrame(rows)

    # Sort projects by Defects4J order, then alphabetically for unknown projects.
    order_map = {p: i for i, p in enumerate(PROJECT_ORDER)}
    summary["_order"] = summary["Project"].map(lambda x: order_map.get(x, 10_000))
    summary = summary.sort_values(["_order", "Project"]).drop(columns=["_order"])

    # Overall row, computed from raw totals rather than averaging project rows.
    overall = summarize_project(df.assign(project="Overall"), use_unique_bugs=use_unique_bugs)
    overall["Project"] = "Overall"

    summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)

    return summary


def fmt_int(x):
    return f"{int(round(float(x)))}"


def fmt_float(x, digits=2):
    return f"{float(x):.{digits}f}"


def fmt_pct(x, digits=1):
    return f"{float(x):.{digits}f}\\%"


def latex_escape(text):
    """
    Escape LaTeX special characters for project names.
    """
    text = str(text)
    replacements = {
        "_": "\\_",
        "%": "\\%",
        "&": "\\&",
        "#": "\\#",
        "$": "\\$",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def make_latex_table(summary, caption, label):
    """
    Main paper table.

    We intentionally keep only a few columns:
        Project
        Bugs
        Invoked Rate
        Attempts/Bug
        Report Rate
        Actions/Attempt

    These columns are enough to discuss:
        1. how frequently runtime probing is triggered;
        2. how many probing attempts are used per bug;
        3. how often probing attempts produce reports;
        4. how many debugger actions each probing attempt uses.
    """

    lines = []

    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lrrrrr@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Project} & "
        "\\textbf{Bugs} & "
        "\\textbf{Invoked Rate} & "
        "\\textbf{Attempts/Bug} & "
        "\\textbf{Report Rate} & "
        "\\textbf{Actions/Attempt} \\\\"
    )
    lines.append("\\midrule")

    for _, row in summary.iterrows():
        project = row["Project"]

        if project == "Overall":
            lines.append("\\midrule")
            line = (
                f"\\textbf{{{latex_escape(project)}}} & "
                f"\\textbf{{{fmt_int(row['Bugs'])}}} & "
                f"\\textbf{{{fmt_pct(row['Invoked Rate'])}}} & "
                f"\\textbf{{{fmt_float(row['Attempts/Bug'], 2)}}} & "
                f"\\textbf{{{fmt_pct(row['Report Rate'])}}} & "
                f"\\textbf{{{fmt_float(row['Actions/Attempt'], 2)}}} \\\\"
            )
        else:
            line = (
                f"{latex_escape(project)} & "
                f"{fmt_int(row['Bugs'])} & "
                f"{fmt_pct(row['Invoked Rate'])} & "
                f"{fmt_float(row['Attempts/Bug'], 2)} & "
                f"{fmt_pct(row['Report Rate'])} & "
                f"{fmt_float(row['Actions/Attempt'], 2)} \\\\"
            )

        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def make_extended_latex_table(summary, caption, label):
    """
    Optional extended table if you want to include Success Rate as well.

    This is usually better for appendix because it is wider.
    """

    lines = []

    lines.append("\\begin{table}[t]")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}lrrrrrr@{}}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Project} & "
        "\\textbf{Bugs} & "
        "\\textbf{Invoked Rate} & "
        "\\textbf{Attempts/Bug} & "
        "\\textbf{Report Rate} & "
        "\\textbf{Success Rate} & "
        "\\textbf{Actions/Attempt} \\\\"
    )
    lines.append("\\midrule")

    for _, row in summary.iterrows():
        project = row["Project"]

        if project == "Overall":
            lines.append("\\midrule")
            line = (
                f"\\textbf{{{latex_escape(project)}}} & "
                f"\\textbf{{{fmt_int(row['Bugs'])}}} & "
                f"\\textbf{{{fmt_pct(row['Invoked Rate'])}}} & "
                f"\\textbf{{{fmt_float(row['Attempts/Bug'], 2)}}} & "
                f"\\textbf{{{fmt_pct(row['Report Rate'])}}} & "
                f"\\textbf{{{fmt_pct(row['Success Rate'])}}} & "
                f"\\textbf{{{fmt_float(row['Actions/Attempt'], 2)}}} \\\\"
            )
        else:
            line = (
                f"{latex_escape(project)} & "
                f"{fmt_int(row['Bugs'])} & "
                f"{fmt_pct(row['Invoked Rate'])} & "
                f"{fmt_float(row['Attempts/Bug'], 2)} & "
                f"{fmt_pct(row['Report Rate'])} & "
                f"{fmt_pct(row['Success Rate'])} & "
                f"{fmt_float(row['Actions/Attempt'], 2)} \\\\"
            )

        lines.append(line)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_interpretation_hints(summary):
    overall = summary[summary["Project"] == "Overall"].iloc[0]

    print("\n[Overall interpretation hints]")
    print(f"- Invoked Rate: {overall['Invoked Rate']:.1f}%")
    print(f"- Attempts/Bug: {overall['Attempts/Bug']:.2f}")
    print(f"- Report Rate: {overall['Report Rate']:.1f}%")
    print(f"- Success Rate: {overall['Success Rate']:.1f}%")
    print(f"- Actions/Attempt: {overall['Actions/Attempt']:.2f}")

    # Projects with high attempts per bug.
    project_rows = summary[summary["Project"] != "Overall"].copy()
    if not project_rows.empty:
        high_attempts = project_rows.sort_values("Attempts/Bug", ascending=False).head(3)
        high_actions = project_rows.sort_values("Actions/Attempt", ascending=False).head(3)
        low_report = project_rows.sort_values("Report Rate", ascending=True).head(3)

        print("\n[Top projects by Attempts/Bug]")
        for _, r in high_attempts.iterrows():
            print(f"- {r['Project']}: {r['Attempts/Bug']:.2f}")

        print("\n[Top projects by Actions/Attempt]")
        for _, r in high_actions.iterrows():
            print(f"- {r['Project']}: {r['Actions/Attempt']:.2f}")

        print("\n[Lowest projects by Report Rate]")
        for _, r in low_report.iterrows():
            print(f"- {r['Project']}: {r['Report Rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize runtime probing invocation statistics for RQ3."
    )

    parser.add_argument(
        "--per-bug",
        default="./rq3/probefl_rq3_full_per_bug_details.csv",
        help="Input per-bug RQ3 CSV."
    )

    parser.add_argument(
        "--out-dir",
        default="./rq3/tables",
        help="Output directory."
    )

    parser.add_argument(
        "--drop-error-rows",
        action="store_true",
        help="Drop rows where error_occurred is true."
    )

    parser.add_argument(
        "--use-unique-bugs",
        action="store_true",
        help=(
            "Count unique bug_id values instead of rows. "
            "By default, rows are treated as processed bug-level runs."
        )
    )

    parser.add_argument(
        "--extended",
        action="store_true",
        help="Generate a wider LaTeX table including Success Rate."
    )

    parser.add_argument(
        "--caption",
        default="Runtime probing invocation and reporting statistics across projects.",
        help="LaTeX table caption."
    )

    parser.add_argument(
        "--label",
        default="tab:rq3_probing_invocation",
        help="LaTeX table label."
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.per_bug)

    require_columns(df, REQUIRED_COLUMNS)

    df = filter_valid_rows(df, drop_error_rows=args.drop_error_rows)

    if df.empty:
        raise ValueError("No rows left after filtering.")

    summary = summarize_all(df, use_unique_bugs=args.use_unique_bugs)

    csv_path = os.path.join(args.out_dir, "rq3_probing_invocation_summary.csv")
    tex_path = os.path.join(args.out_dir, "rq3_probing_invocation_table.tex")

    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    if args.extended:
        latex_table = make_extended_latex_table(
            summary,
            caption=args.caption,
            label=args.label
        )
    else:
        latex_table = make_latex_table(
            summary,
            caption=args.caption,
            label=args.label
        )

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {tex_path}")

    print("\n[LaTeX table]")
    print(latex_table)

    print_interpretation_hints(summary)


if __name__ == "__main__":
    main()
