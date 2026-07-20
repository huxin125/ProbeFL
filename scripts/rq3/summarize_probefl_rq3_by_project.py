import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


# ============================================================
# Basic utilities
# ============================================================

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur


def to_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def to_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except Exception:
        return default


def parse_project_from_bug_id(bug_id):
    bug_id = str(bug_id)
    if "_" in bug_id:
        return bug_id.split("_", 1)[0]
    return ""


# ============================================================
# Function/tool call extraction
# ============================================================

def count_function_calls(messages):
    """
    Count function calls in OpenAI-style messages.
    """
    counter = Counter()

    if not isinstance(messages, list):
        return counter

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        fc = msg.get("function_call")
        if isinstance(fc, dict):
            name = fc.get("name")
            if name:
                counter[name] += 1

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                if isinstance(fn, dict):
                    name = fn.get("name")
                    if name:
                        counter[name] += 1

    return counter


def is_debugger_action(action_name):
    """
    Only count real debugger operations as debugger actions.
    submit_verification_report is not counted as a debugger action.
    """
    if not action_name:
        return False
    return str(action_name).startswith("jdb_")


# ============================================================
# Dynamic probing statistics
# ============================================================

def extract_dynamic_stats(data):
    attempts = data.get("dynamic_verification_attempts", [])
    if not isinstance(attempts, list):
        attempts = []

    nominated_methods = len(attempts)

    # Sum of metadata.attempts_count.
    # If absent, count each dynamic_verification_attempt as one verification attempt.
    verification_attempts_total = 0

    trajectory_counter = Counter()
    sub_message_call_counter = Counter()
    failure_pattern_counter = Counter()

    target_lines_total = 0
    locals_count_total = 0
    fields_count_total = 0
    call_count_total = 0

    guardrail_reject_count = 0
    tool_error_count = 0

    verify_time_total = 0.0
    success_attempts = 0

    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue

        metadata = attempt.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        attempts_count = metadata.get("attempts_count", None)
        if attempts_count is None:
            verification_attempts_total += 1
        else:
            verification_attempts_total += to_int(attempts_count, 1)

        target_lines = metadata.get("target_lines", [])
        if isinstance(target_lines, list):
            target_lines_total += len(target_lines)

        failure_patterns = metadata.get("failure_patterns", [])
        if isinstance(failure_patterns, list):
            for p in failure_patterns:
                p = str(p)
                failure_pattern_counter[p] += 1
                if p.upper() == "SUCCESS":
                    success_attempts += 1

        probing_trajectories = metadata.get("probing_trajectories", [])
        if isinstance(probing_trajectories, list):
            for traj in probing_trajectories:
                if isinstance(traj, list):
                    for action in traj:
                        trajectory_counter[str(action)] += 1

        cfs = metadata.get("context_fusion_stats", {})
        if isinstance(cfs, dict):
            locals_count_total += to_int(cfs.get("locals_count"), 0)
            fields_count_total += to_int(cfs.get("fields_count"), 0)
            call_count_total += to_int(cfs.get("call_count"), 0)

        guardrail_reject_count += to_int(metadata.get("guardrail_reject_count"), 0)
        tool_error_count += to_int(metadata.get("tool_error_count"), 0)
        verify_time_total += to_float(metadata.get("verify_time_seconds"), 0.0)

        sub_msgs = attempt.get("sub_agent_messages", [])
        sub_message_call_counter.update(count_function_calls(sub_msgs))

    # Prefer recorded probing trajectories because they are usually cleaner.
    # If unavailable, fall back to function calls in sub-agent messages.
    if sum(trajectory_counter.values()) > 0:
        action_counter = trajectory_counter
        action_source = "trajectory"
    else:
        action_counter = sub_message_call_counter
        action_source = "sub_agent_messages"

    debugger_action_total = sum(
        cnt for action, cnt in action_counter.items()
        if is_debugger_action(action)
    )

    return {
        "nominated_methods": nominated_methods,
        "verification_attempts_total": verification_attempts_total,
        "success_attempts": success_attempts,

        "target_lines_total": target_lines_total,
        "locals_count_total": locals_count_total,
        "fields_count_total": fields_count_total,
        "call_count_total": call_count_total,

        "guardrail_reject_count": guardrail_reject_count,
        "tool_error_count": tool_error_count,
        "verify_time_total": verify_time_total,

        "action_counter": action_counter,
        "action_source": action_source,
        "debugger_action_total": debugger_action_total,
        "failure_pattern_counter": failure_pattern_counter,
    }


# ============================================================
# Extract one JSON file
# ============================================================

def extract_one_file(path):
    data = load_json(path)
    if not isinstance(data, dict):
        return None

    bug_id = data.get("bug_id", path.stem)
    project = parse_project_from_bug_id(bug_id)

    mode_name = data.get("mode_name", "")
    ablation_mode = data.get("ablation_mode", "")

    total_time = to_float(data.get("total_time_seconds"), 0.0)
    static_time = to_float(
        safe_get(data, ["cost_breakdown", "static_search_time"], 0.0),
        0.0
    )
    dynamic_time = to_float(
        safe_get(data, ["cost_breakdown", "dynamic_verify_time"], 0.0),
        0.0
    )

    main_prompt = to_int(
        safe_get(data, ["cost_breakdown", "main_agent_tokens", "prompt"], 0),
        0
    )
    main_completion = to_int(
        safe_get(data, ["cost_breakdown", "main_agent_tokens", "completion"], 0),
        0
    )

    sub_prompt = to_int(
        safe_get(data, ["cost_breakdown", "sub_agent_total_tokens", "prompt"], 0),
        0
    )
    sub_completion = to_int(
        safe_get(data, ["cost_breakdown", "sub_agent_total_tokens", "completion"], 0),
        0
    )

    main_tokens = main_prompt + main_completion
    sub_tokens = sub_prompt + sub_completion
    total_tokens = main_tokens + sub_tokens

    main_tool_calls = count_function_calls(data.get("messages", []))
    dyn = extract_dynamic_stats(data)

    ac = dyn["action_counter"]
    fp = dyn["failure_pattern_counter"]

    row = {
        "file": str(path),
        "bug_id": bug_id,
        "project": project,

        "tool": data.get("tool", "ProbeFL"),
        "main_model": data.get("main_model", ""),
        "verifier_model": data.get("verifier_model", ""),
        "mode_name": mode_name,
        "ablation_mode": ablation_mode,
        "error_occurred": int(bool(data.get("error_occurred", False))),

        # Time
        "total_time_seconds": total_time,
        "static_search_time": static_time,
        "dynamic_verify_time": dynamic_time,
        "dynamic_time_ratio": dynamic_time / total_time if total_time > 0 else 0.0,

        # Tokens
        "main_prompt_tokens": main_prompt,
        "main_completion_tokens": main_completion,
        "main_total_tokens": main_tokens,
        "sub_prompt_tokens": sub_prompt,
        "sub_completion_tokens": sub_completion,
        "sub_total_tokens": sub_tokens,
        "total_tokens": total_tokens,
        "sub_token_ratio": sub_tokens / total_tokens if total_tokens > 0 else 0.0,

        # Dynamic probing
        "nominated_methods": dyn["nominated_methods"],
        "verification_attempts_total": dyn["verification_attempts_total"],
        "success_attempts": dyn["success_attempts"],
        "verify_time_total": dyn["verify_time_total"],

        "target_lines_total": dyn["target_lines_total"],
        "locals_count_total": dyn["locals_count_total"],
        "fields_count_total": dyn["fields_count_total"],
        "call_count_total": dyn["call_count_total"],

        "guardrail_reject_count": dyn["guardrail_reject_count"],
        "tool_error_count": dyn["tool_error_count"],

        "action_source": dyn["action_source"],
        "debugger_action_total": dyn["debugger_action_total"],

        # Common debugger actions
        "jdb_get_locals": ac.get("jdb_get_locals", 0),
        "jdb_print_variable": ac.get("jdb_print_variable", 0),
        "jdb_run_to_breakpoint": ac.get("jdb_run_to_breakpoint", 0),
        "jdb_where": ac.get("jdb_where", 0),
        "jdb_step": ac.get("jdb_step", 0),
        "jdb_next": ac.get("jdb_next", 0),
        "jdb_eval": ac.get("jdb_eval", 0),
        "jdb_list": ac.get("jdb_list", 0),
        "submit_verification_report": ac.get("submit_verification_report", 0),

        # Main-agent static/search tool calls
        "get_failing_tests_covered_classes": main_tool_calls.get("get_failing_tests_covered_classes", 0),
        "get_failing_tests_covered_methods_for_class": main_tool_calls.get("get_failing_tests_covered_methods_for_class", 0),
        "get_code_snippet": main_tool_calls.get("get_code_snippet", 0),
        "nominate_suspicious_method": main_tool_calls.get("nominate_suspicious_method", 0),

        # Encoded counters for later inspection
        "failure_patterns": ";".join(f"{k}:{v}" for k, v in sorted(fp.items())),
        "action_distribution": ";".join(f"{k}:{v}" for k, v in sorted(ac.items())),
    }

    return row


# ============================================================
# Summary helpers
# ============================================================

def p90(series):
    if len(series) == 0:
        return 0.0
    return series.quantile(0.9)


def make_project_order(projects):
    preferred = ["Chart", "Closure", "Lang", "Math", "Time"]
    ordered = [p for p in preferred if p in projects]
    rest = sorted([p for p in projects if p not in preferred])
    return ordered + rest


def summarize_cost_by_project(df):
    rows = []

    projects = make_project_order(set(df["project"].dropna().tolist()))

    for project in projects + ["Overall"]:
        if project == "Overall":
            g = df
        else:
            g = df[df["project"] == project]

        if len(g) == 0:
            continue

        avg_total_time = g["total_time_seconds"].mean()
        avg_static_time = g["static_search_time"].mean()
        avg_dynamic_time = g["dynamic_verify_time"].mean()
        avg_total_tokens = g["total_tokens"].mean()
        avg_main_tokens = g["main_total_tokens"].mean()
        avg_sub_tokens = g["sub_total_tokens"].mean()

        rows.append({
            "project": project,
            "bugs": len(g),
            "error_count": int(g["error_occurred"].sum()),

            "avg_total_time_seconds": avg_total_time,
            "median_total_time_seconds": g["total_time_seconds"].median(),
            "p90_total_time_seconds": p90(g["total_time_seconds"]),

            "avg_static_search_time": avg_static_time,
            "avg_dynamic_verify_time": avg_dynamic_time,
            "avg_dynamic_time_ratio": avg_dynamic_time / avg_total_time if avg_total_time > 0 else 0.0,

            "avg_total_tokens": avg_total_tokens,
            "avg_main_agent_tokens": avg_main_tokens,
            "avg_runtime_probing_agent_tokens": avg_sub_tokens,
            "avg_runtime_probing_token_ratio": avg_sub_tokens / avg_total_tokens if avg_total_tokens > 0 else 0.0,

            "avg_main_prompt_tokens": g["main_prompt_tokens"].mean(),
            "avg_main_completion_tokens": g["main_completion_tokens"].mean(),
            "avg_sub_prompt_tokens": g["sub_prompt_tokens"].mean(),
            "avg_sub_completion_tokens": g["sub_completion_tokens"].mean(),
        })

    return pd.DataFrame(rows)


def summarize_behavior_by_project(df):
    rows = []

    projects = make_project_order(set(df["project"].dropna().tolist()))

    for project in projects + ["Overall"]:
        if project == "Overall":
            g = df
        else:
            g = df[df["project"] == project]

        if len(g) == 0:
            continue

        total_verification_attempts = g["verification_attempts_total"].sum()
        total_success_attempts = g["success_attempts"].sum()

        rows.append({
            "project": project,
            "bugs": len(g),

            "avg_nominated_methods": g["nominated_methods"].mean(),
            "avg_verification_attempts": g["verification_attempts_total"].mean(),
            "attempt_success_rate": (
                total_success_attempts / total_verification_attempts
                if total_verification_attempts > 0 else 0.0
            ),

            "avg_debugger_actions": g["debugger_action_total"].mean(),
            "avg_jdb_get_locals": g["jdb_get_locals"].mean(),
            "avg_jdb_print_variable": g["jdb_print_variable"].mean(),
            "avg_jdb_run_to_breakpoint": g["jdb_run_to_breakpoint"].mean(),
            "avg_jdb_where": g["jdb_where"].mean(),
            "avg_jdb_step": g["jdb_step"].mean(),
            "avg_jdb_next": g["jdb_next"].mean(),

            "avg_target_lines": g["target_lines_total"].mean(),
            "avg_inspected_locals": g["locals_count_total"].mean(),
            "avg_inspected_fields": g["fields_count_total"].mean(),
            "avg_call_contexts": g["call_count_total"].mean(),

            "avg_tool_errors": g["tool_error_count"].mean(),
            "sum_tool_errors": int(g["tool_error_count"].sum()),
            "avg_guardrail_rejections": g["guardrail_reject_count"].mean(),
            "sum_guardrail_rejections": int(g["guardrail_reject_count"].sum()),

            "avg_get_covered_classes": g["get_failing_tests_covered_classes"].mean(),
            "avg_get_covered_methods": g["get_failing_tests_covered_methods_for_class"].mean(),
            "avg_get_code_snippet": g["get_code_snippet"].mean(),
            "avg_nominate_suspicious_method": g["nominate_suspicious_method"].mean(),
        })

    return pd.DataFrame(rows)


def parse_counter_string(s):
    counter = Counter()
    if not isinstance(s, str) or not s.strip():
        return counter

    parts = s.split(";")
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.rsplit(":", 1)
        counter[k] += to_int(v, 0)

    return counter


def summarize_counter_by_project(df, counter_col, output_name):
    rows = []

    projects = make_project_order(set(df["project"].dropna().tolist()))

    for project in projects + ["Overall"]:
        if project == "Overall":
            g = df
        else:
            g = df[df["project"] == project]

        total = Counter()
        for x in g[counter_col].tolist():
            total.update(parse_counter_string(x))

        for key, count in sorted(total.items()):
            rows.append({
                "project": project,
                output_name: key,
                "count": count,
                "bugs": len(g),
                "avg_per_bug": count / len(g) if len(g) > 0 else 0.0,
            })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Summarize ProbeFL RQ3 cost and runtime probing behavior by project."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing ProbeFL JSON result files."
    )

    parser.add_argument(
        "--glob",
        default="**/*.json",
        help="Glob pattern for JSON files. Default: **/*.json"
    )

    parser.add_argument(
        "--output-prefix",
        default="probefl_rq3",
        help="Output file prefix."
    )

    parser.add_argument(
        "--mode-name",
        default="full",
        help="Only include files with this mode_name. Use 'ALL' to disable filtering."
    )

    parser.add_argument(
        "--ablation-mode",
        default="none",
        help="Only include files with this ablation_mode. Use 'ALL' to disable filtering."
    )

    parser.add_argument(
        "--exclude-errors",
        action="store_true",
        help="Exclude files where error_occurred is true from summaries."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.glob))

    print("=" * 100)
    print("ProbeFL RQ3 Summarization by Project")
    print("=" * 100)
    print(f"Input dir      : {input_dir}")
    print(f"Glob           : {args.glob}")
    print(f"Found files    : {len(files)}")
    print(f"mode_name      : {args.mode_name}")
    print(f"ablation_mode  : {args.ablation_mode}")
    print(f"exclude_errors : {args.exclude_errors}")
    print("=" * 100)

    rows = []

    for path in files:
        row = extract_one_file(path)
        if row is None:
            continue

        if args.mode_name != "ALL" and row["mode_name"] != args.mode_name:
            continue

        if args.ablation_mode != "ALL" and row["ablation_mode"] != args.ablation_mode:
            continue

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("[ERROR] No valid rows after filtering.")
        return

    detail_df = df.copy()

    summary_df = df.copy()
    if args.exclude_errors:
        summary_df = summary_df[summary_df["error_occurred"] == 0].copy()

    if summary_df.empty:
        print("[ERROR] No rows left for summaries after excluding errors.")
        return

    # Output paths
    per_bug_csv = f"{args.output_prefix}_per_bug_details.csv"
    cost_csv = f"{args.output_prefix}_cost_by_project.csv"
    behavior_csv = f"{args.output_prefix}_behavior_by_project.csv"
    failure_csv = f"{args.output_prefix}_failure_patterns_by_project.csv"
    action_csv = f"{args.output_prefix}_action_distribution_by_project.csv"

    # Generate summaries
    cost_df = summarize_cost_by_project(summary_df)
    behavior_df = summarize_behavior_by_project(summary_df)
    failure_df = summarize_counter_by_project(
        summary_df,
        counter_col="failure_patterns",
        output_name="failure_pattern"
    )
    action_df = summarize_counter_by_project(
        summary_df,
        counter_col="action_distribution",
        output_name="action"
    )

    # Save CSVs
    detail_df.to_csv(per_bug_csv, index=False, encoding="utf-8-sig")
    cost_df.to_csv(cost_csv, index=False, encoding="utf-8-sig")
    behavior_df.to_csv(behavior_csv, index=False, encoding="utf-8-sig")
    failure_df.to_csv(failure_csv, index=False, encoding="utf-8-sig")
    action_df.to_csv(action_csv, index=False, encoding="utf-8-sig")

    # Print summaries
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 100)
    print("Cost by Project")
    print("=" * 100)
    print(cost_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("Runtime Probing Behavior by Project")
    print("=" * 100)
    print(behavior_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("Saved files")
    print("=" * 100)
    print(per_bug_csv)
    print(cost_csv)
    print(behavior_csv)
    print(failure_csv)
    print(action_csv)
    print("=" * 100)


if __name__ == "__main__":
    main()
