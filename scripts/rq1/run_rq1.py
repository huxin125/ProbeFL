import argparse
import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ORDER = ["Chart", "Closure", "Lang", "Math", "Time"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def to_int(x, default=None):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def normalize_project_bug(project, bug_id=None, bug_name=None):
    if bug_name is not None and str(bug_name).strip():
        s = str(bug_name).strip()
        s = s.replace("-", "_")
        if "_" in s:
            p, b = s.split("_", 1)
            bid = to_int(b, None)
            if bid is not None:
                return p, bid, f"{p}_{bid}"

    p = str(project).strip()
    bid = to_int(bug_id, None)
    if bid is None:
        return p, None, ""
    return p, bid, f"{p}_{bid}"


def compute_top_flags_from_rank(rank):
    r = to_float(rank, None)
    if r is None or r <= 0:
        return 0, 0, 0, 0, 0.0

    top1 = int(r <= 1)
    top3 = int(r <= 3)
    top5 = int(r <= 5)
    top10 = int(r <= 10)
    rr = 1.0 / r
    return top1, top3, top5, top10, rr


def find_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def load_baseline_csv(path, tool_name):
    path = Path(path)
    df = pd.read_csv(path)

    project_col = find_col(df, ["project", "Project"])
    bug_id_col = find_col(df, ["bug_id", "bugId", "id"])
    bug_name_col = find_col(df, ["bug_name", "bug", "bug_id_name"], default=None)

    rank_col = find_col(df, ["rank", "Rank", "autofl_rank"], default=None)
    rr_col = find_col(df, ["RR", "rr", "MRR"], default=None)

    missing_col = find_col(df, ["missing_result", "missing"], default=None)
    invalid_col = find_col(df, ["invalid_result", "invalid"], default=None)
    empty_col = find_col(df, ["empty_output", "empty_result", "empty"], default=None)

    evaluated_col = find_col(df, ["evaluated"], default=None)
    has_gt_col = find_col(df, ["has_gt", "has_ground_truth"], default=None)

    rows = []

    for _, row in df.iterrows():
        project = row[project_col] if project_col else ""
        bug_id = row[bug_id_col] if bug_id_col else None
        bug_name = row[bug_name_col] if bug_name_col else None

        p, bid, bname = normalize_project_bug(project, bug_id, bug_name)
        if not bname:
            continue

        rank = row[rank_col] if rank_col else ""
        rank_num = to_float(rank, None)

        top1, top3, top5, top10, rr_from_rank = compute_top_flags_from_rank(rank_num)

        if rr_col and rr_col in df.columns:
            rr = to_float(row[rr_col], rr_from_rank)
            if rr is None:
                rr = rr_from_rank
        else:
            rr = rr_from_rank

        if rank_num is None:
            rank_out = ""
        else:
            rank_out = rank_num

        missing_result = int(to_int(row[missing_col], 0) or 0) if missing_col else 0
        invalid_result = int(to_int(row[invalid_col], 0) or 0) if invalid_col else 0
        empty_result = int(to_int(row[empty_col], 0) or 0) if empty_col else 0

        if evaluated_col:
            evaluated = int(to_int(row[evaluated_col], 1) or 0)
        else:
            evaluated = 1

        if has_gt_col:
            has_gt = int(to_int(row[has_gt_col], 1) or 0)
        else:
            has_gt = 1

        rows.append({
            "tool": tool_name,
            "project": p,
            "bug_id": bid,
            "bug_name": bname,

            "evaluated": evaluated,
            "has_gt": has_gt,

            "missing_result": missing_result,
            "invalid_result": invalid_result,
            "empty_result": empty_result,
            "missing_algorithm_row": 0,

            "rank": rank_out,
            "top1": top1,
            "top3": top3,
            "top5": top5,
            "top10": top10,
            "RR": rr,
        })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["project", "bug_id"]).drop_duplicates(
            subset=["tool", "bug_name"],
            keep="first"
        )

    return out


def load_probefl_scores(score_dir, projects, mode_name="ProbeFL_FULL", tool_name="ProbeFL"):
    score_dir = Path(score_dir)
    rows = []

    for project in projects:
        path = score_dir / f"{project}_{mode_name}_scores.json"
        if not path.exists():
            print(f"[WARN] Missing ProbeFL score file: {path}")
            continue

        data = load_json(path)
        buggy_methods = data.get("buggy_methods", {})

        for bug_name, method_dict in buggy_methods.items():
            p, bid, bname = normalize_project_bug(project, None, bug_name)
            if not bname:
                continue

            ranks = []
            if isinstance(method_dict, dict):
                for _, info in method_dict.items():
                    if not isinstance(info, dict):
                        continue
                    r = to_float(info.get("autofl_rank"), None)
                    if r is not None and r > 0:
                        ranks.append(r)

            if ranks:
                best_rank = min(ranks)
                has_gt = 1
                evaluated = 1
            else:
                best_rank = None
                has_gt = 0
                evaluated = 0

            top1, top3, top5, top10, rr = compute_top_flags_from_rank(best_rank)

            rows.append({
                "tool": tool_name,
                "project": p,
                "bug_id": bid,
                "bug_name": bname,

                "evaluated": evaluated,
                "has_gt": has_gt,

                "missing_result": 0,
                "invalid_result": 0,
                "empty_result": 0,
                "missing_algorithm_row": 0,

                "rank": best_rank if best_rank is not None else "",
                "top1": top1,
                "top3": top3,
                "top5": top5,
                "top10": top10,
                "RR": rr,
            })

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["project", "bug_id"]).drop_duplicates(
            subset=["tool", "bug_name"],
            keep="first"
        )

    return out


def eligible_bug_set(df):
    if df.empty:
        return set()
    g = df[(df["evaluated"] == 1) & (df["has_gt"] == 1)]
    return set(g["bug_name"].tolist())


def project_sort_key(bug_name):
    if "_" not in bug_name:
        return (999, bug_name, 0)

    p, b = bug_name.split("_", 1)
    try:
        bid = int(b)
    except Exception:
        bid = 0

    try:
        pidx = PROJECT_ORDER.index(p)
    except ValueError:
        pidx = 999

    return (pidx, p, bid)


def make_failure_row(tool, bug_name):
    p, bid, bname = normalize_project_bug("", None, bug_name)

    return {
        "tool": tool,
        "project": p,
        "bug_id": bid,
        "bug_name": bname,

        "evaluated": 1,
        "has_gt": 1,

        "missing_result": 1,
        "invalid_result": 0,
        "empty_result": 0,
        "missing_algorithm_row": 1,

        "rank": "",
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "top10": 0,
        "RR": 0.0,
    }


def build_eval_set(tool_dfs, policy, ref_tool):
    sets = {}

    for tool, df in tool_dfs.items():
        sets[tool] = eligible_bug_set(df)

    if policy == "common":
        eval_set = None
        for tool, s in sets.items():
            if eval_set is None:
                eval_set = set(s)
            else:
                eval_set &= set(s)
        if eval_set is None:
            eval_set = set()

    elif policy == "probefl_ref":
        if ref_tool not in sets:
            raise ValueError(f"Reference tool {ref_tool} not found.")
        eval_set = set(sets[ref_tool])

    elif policy == "union":
        eval_set = set()
        for tool, s in sets.items():
            eval_set |= set(s)

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return sorted(eval_set, key=project_sort_key), sets


def build_merged_details(tool_dfs, eval_bugs):
    rows = []

    for tool, df in tool_dfs.items():
        index = {}
        for _, r in df.iterrows():
            index[r["bug_name"]] = r.to_dict()

        for bug_name in eval_bugs:
            if bug_name in index:
                row = dict(index[bug_name])
            else:
                row = make_failure_row(tool, bug_name)

            rows.append(row)

    merged = pd.DataFrame(rows)
    merged = merged.sort_values(["tool", "project", "bug_id"]).reset_index(drop=True)
    return merged


def summarize(merged):
    rows = []

    tools = sorted(merged["tool"].unique().tolist())

    for tool in tools:
        tool_df = merged[merged["tool"] == tool]

        projects = [p for p in PROJECT_ORDER if p in set(tool_df["project"])]
        for project in projects + ["Overall"]:
            if project == "Overall":
                g = tool_df
            else:
                g = tool_df[tool_df["project"] == project]

            if g.empty:
                continue

            total = len(g)
            top1 = int(g["top1"].sum())
            top3 = int(g["top3"].sum())
            top5 = int(g["top5"].sum())
            top10 = int(g["top10"].sum())
            mrr = float(pd.to_numeric(g["RR"], errors="coerce").fillna(0.0).mean())

            located_ranks = pd.to_numeric(g["rank"], errors="coerce").dropna()
            mfr = float(located_ranks.mean()) if len(located_ranks) > 0 else None

            rows.append({
                "tool": tool,
                "project": project,
                "total": total,

                "top1": top1,
                "top3": top3,
                "top5": top5,
                "top10": top10,

                "top1_rate": top1 / total * 100.0 if total else 0.0,
                "top3_rate": top3 / total * 100.0 if total else 0.0,
                "top5_rate": top5 / total * 100.0 if total else 0.0,
                "top10_rate": top10 / total * 100.0 if total else 0.0,

                "MRR": mrr,
                "MFR_located_only": mfr,

                "missing_result": int(g["missing_result"].sum()),
                "invalid_result": int(g["invalid_result"].sum()),
                "empty_result": int(g["empty_result"].sum()),
                "missing_algorithm_row": int(g["missing_algorithm_row"].sum()),
            })

    return pd.DataFrame(rows)


def print_summary(summary_df):
    print("\n" + "=" * 100)
    print("RQ1 Unified Summary")
    print("=" * 100)

    for tool in sorted(summary_df["tool"].unique().tolist()):
        print(f"\n[{tool}]")
        g = summary_df[summary_df["tool"] == tool].copy()

        cols = [
            "project", "total",
            "top1", "top1_rate",
            "top3", "top3_rate",
            "top5", "top5_rate",
            "top10", "top10_rate",
            "MRR",
            "missing_algorithm_row",
        ]

        print(g[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Merge RQ1 per-bug results from ProbeFL, AutoFL, PingFL, and Ochiai."
    )

    parser.add_argument("--autofl-csv", default="data/baseline/autofl_gpt5.4_topk_details.csv")
    parser.add_argument("--pingfl-csv", default="data/baseline/pingfl_gpt54_topk.csv")
    parser.add_argument("--ochiai-csv", default="data/baseline/ochiai_strict_score_worst_details.csv")

    parser.add_argument(
        "--probefl-score-dir",
        default="scripts/rq1/outputs"
    )

    parser.add_argument(
        "--probefl-mode",
        default="ProbeFL_FULL"
    )

    parser.add_argument(
        "--projects",
        nargs="+",
        default=PROJECT_ORDER
    )

    parser.add_argument(
        "--policy",
        choices=["common", "probefl_ref", "union"],
        nargs="+",
        default=["common", "probefl_ref"],
        help="Which evaluation set policies to run."
    )

    parser.add_argument(
        "--out-dir",
        default="./scripts/rq1/outputs"
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tool_dfs = {}

    if args.probefl_score_dir:
        tool_dfs["ProbeFL"] = load_probefl_scores(
            score_dir=args.probefl_score_dir,
            projects=args.projects,
            mode_name=args.probefl_mode,
            tool_name="ProbeFL"
        )

    if args.autofl_csv:
        tool_dfs["AutoFL"] = load_baseline_csv(args.autofl_csv, "AutoFL")

    if args.pingfl_csv:
        tool_dfs["PingFL"] = load_baseline_csv(args.pingfl_csv, "PingFL")

    if args.ochiai_csv:
        tool_dfs["Ochiai"] = load_baseline_csv(args.ochiai_csv, "Ochiai")

    if not tool_dfs:
        raise ValueError("No tool results loaded.")

    print("\n[Loaded tools]")
    for tool, df in tool_dfs.items():
        print(f"{tool}: rows={len(df)}, eligible={len(eligible_bug_set(df))}")

    for policy in args.policy:
        print(f"\n{'='*60}")
        print(f"Running policy: {policy}")
        print(f"{'='*60}")

        eval_bugs, tool_sets = build_eval_set(
            tool_dfs=tool_dfs,
            policy=policy,
            ref_tool="ProbeFL"
        )

        print(f"Eval set size: {len(eval_bugs)}")

        eval_set_df = pd.DataFrame({
            "bug_name": eval_bugs,
            "project": [b.split("_", 1)[0] if "_" in b else "" for b in eval_bugs],
            "bug_id": [to_int(b.split("_", 1)[1], None) if "_" in b else None for b in eval_bugs],
        })

        merged = build_merged_details(tool_dfs, eval_bugs)
        summary_df = summarize(merged)

        eval_set_path = out_dir / f"rq1_eval_set_{policy}.csv"
        merged_path = out_dir / f"rq1_merged_per_bug_{policy}.csv"
        summary_path = out_dir / f"rq1_summary_{policy}.csv"

        eval_set_df.to_csv(eval_set_path, index=False, encoding="utf-8-sig")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

        print_summary(summary_df)

        print(f"\n[Saved] {eval_set_path}")
        print(f"[Saved] {merged_path}")
        print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()
