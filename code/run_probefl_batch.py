import os
import sys
import time
import json
import argparse
import traceback
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from probefl import AutoDebugger
from lib.protocol import PersistentJDBBackend
from lib.repo_interface import get_repo_interface, get_bug_metadata


D4J_PROJECTS = {
    "Chart": 26,
    "Closure": 133,
    "Lang": 65,
    "Math": 106,
    "Time": 27
}


def is_bug_completed(json_path):
    """[cleaned]：[cleaned] error [cleaned]"""
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return "buggy_methods" in data and not data.get("error_occurred", False)
    except Exception:
        return False


def cleanup_environment(pid=None):
    """
    [cleaned]（[cleaned]）。
    - [cleaned] pid：[cleaned]（[cleaned] backend.shutdown() [cleaned]）
    - [cleaned] pid：[cleaned]（[cleaned]）
    """
    if pid is None:
        return
    try:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass  # (comment removed)


def has_available_snippet(bug_name):
    """
    [cleaned]：[cleaned] snippet [cleaned] failing test snippet [cleaned]，[cleaned]，[cleaned]。
    """
    try:
        ri = get_repo_interface(
            bug_name,
            show_line_number=True,
            postprocess_test_snippet=True
        )

        available_fail_tests = [
            sig for sig in ri.failing_test_signatures
            if ri.get_test_snippet(sig) is not None
        ]

        if not available_fail_tests:
            print(f"  ⏭️  [SKIP] {bug_name}: no available failing test snippet.")
            return False

        return True

    except Exception as e:
        print(f"  ⏭️  [SKIP] {bug_name}: failed to load snippets: {e}")
        return False


def collect_dynamic_stats(backend):
    """
    [cleaned] backend [cleaned]。
    [cleaned]：[cleaned] backend.dynamic_stats [cleaned] mode [cleaned]。
    """
    dynamic_stats = []
    total_verify_time = 0.0
    sub_agent_tokens = {"prompt": 0, "completion": 0}

    if hasattr(backend, "dynamic_stats"):
        dynamic_stats = backend.dynamic_stats
        for stat in dynamic_stats:
            total_verify_time += stat["metadata"].get("verify_time_seconds", 0)
            sub_agent_tokens["prompt"] += stat["sub_agent_tokens"].get("prompt", 0)
            sub_agent_tokens["completion"] += stat["sub_agent_tokens"].get("completion", 0)

    return dynamic_stats, total_verify_time, sub_agent_tokens


def worker_process(
    bug_name,
    main_model,
    verifier_model,
    prompt_file,
    base_label,
    max_budget,
    debug,
    timeout_seconds=None,
    modes_filter="all"
):
    """
    [cleaned] worker [cleaned] bug。
    [cleaned] bug：
      - checkout/compile [cleaned]；
      - [cleaned] full + [cleaned]；
      - [cleaned] shutdown。

    timeout_seconds: [cleaned] bug [cleaned]（[cleaned]），[cleaned] mode
    """

    log_prefix = f"[{bug_name}]"

    if not has_available_snippet(bug_name):
        return

    deadline = time.time() + timeout_seconds if timeout_seconds else None

    all_modes = {
        "full": {
            "label": f"{base_label}_FULL",
            "ablation_mode": "none"
        },
        "no_context_bridge": {
            "label": f"{base_label}_NO_CONTEXT_BRIDGE",
            "ablation_mode": "no_context_bridge"
        },
        "no_reflection": {
            "label": f"{base_label}_NO_REFLECTION",
            "ablation_mode": "no_reflection"
        },
        "no_dynamic": {
            "label": f"{base_label}_NO_DYNAMIC",
            "ablation_mode": "no_dynamic"
        }
    }
    if modes_filter == "all":
        modes = all_modes
    else:
        modes = {k: v for k, v in all_modes.items() if k in modes_filter}

    out_files = {}
    all_done = True

    for mode_name, cfg in modes.items():
        out_dir = os.path.abspath(f"results/{cfg['label']}/{main_model}")
        os.makedirs(out_dir, exist_ok=True)

        out_json = os.path.join(out_dir, f"{bug_name}.json")
        out_files[mode_name] = out_json

        if not is_bug_completed(out_json):
            all_done = False

    if all_done:
        print(f"  ⏭️  {bug_name} already completed in all ProbeFL modes. Skipping.")
        return

    backend = None
    try:
        backend = PersistentJDBBackend(
            bug_name,
            debug=debug,
            show_line_number=True,
            postprocess_test_snippet=True,
            ablation_mode="none",
            verifier_model=verifier_model  # (comment removed)
        )

        res = backend.startup()
        if res.is_error:
            print(f"  ❌ [Backend] Startup failed for {bug_name}: {res.output}")
            return

        for mode_name, cfg in modes.items():
            if deadline and time.time() > deadline:
                print(f"{log_prefix} ⏰ Deadline reached, skipping [{mode_name.upper()}] and remaining modes.")
                break

            out_json = out_files[mode_name]

            if is_bug_completed(out_json):
                print(f"{log_prefix} ⏭️  [{mode_name.upper()}] already completed. Skipping mode.")
                continue

            ablation_mode = cfg["ablation_mode"]
            label = cfg["label"]

            print(f"  🚀 [{label}] Running {bug_name} | ablation_mode={ablation_mode}")

            backend.ablation_mode = ablation_mode
            backend.dynamic_stats = []

            reset_res = backend.reset_session()

            ad = AutoDebugger(
                bug_name,
                main_model,
                prompt_file,
                backend=backend,
                debug=debug,
                allow_multi_predictions=True
            )

            start_time = time.time()
            grade = None
            error_occurred = False

            try:
                grade = ad.run(max_budget)
            except Exception as e:
                error_occurred = True
                grade = {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                if debug:
                    print(f"\n{'=' * 60}\nError in {bug_name} [{mode_name}]:\n{traceback.format_exc()}\n{'=' * 60}")

            end_time = time.time()
            total_time = end_time - start_time

            dynamic_stats, total_verify_time, sub_agent_tokens = collect_dynamic_stats(backend)

            sub_agent_api_calls = 0
            for stat in dynamic_stats:
                for msg in stat.get("sub_agent_messages", []):
                    if isinstance(msg, dict) and msg.get("function_call"):
                        sub_agent_api_calls += 1

            try:
                bug_metadata = get_bug_metadata(bug_name,
                    show_line_number=True,
                    postprocess_test_snippet=True)
            except Exception:
                bug_metadata = {}

            try:
                environment_stats = backend.get_environment_stats()
            except Exception:
                environment_stats = {}

            error_category = None
            if error_occurred and isinstance(grade, dict):
                error_str = str(grade.get('error', ''))
                if 'InvalidRequestError' in error_str or 'context_length' in error_str.lower():
                    error_category = 'CONTEXT_LENGTH'
                elif 'RateLimitError' in error_str or 'rate_limit' in error_str.lower():
                    error_category = 'RATE_LIMIT'
                elif 'Timeout' in error_str or 'timeout' in error_str.lower():
                    error_category = 'TIMEOUT'
                elif 'JDB' in error_str or 'SESSION_LOST' in error_str:
                    error_category = 'JDB_ERROR'
                elif 'checkout' in error_str.lower() or 'compile' in error_str.lower():
                    error_category = 'ENVIRONMENT'
                else:
                    error_category = 'OTHER'

            result = {
                "tool": base_label,
                "bug_id": bug_name,
                "main_model": main_model,
                "verifier_model": verifier_model,
                "ablation_mode": ablation_mode,
                "mode_name": mode_name,
                "total_time_seconds": total_time,

                "bug_metadata": bug_metadata,

                "cost_breakdown": {
                    "static_search_time": total_time - total_verify_time,
                    "dynamic_verify_time": total_verify_time,
                    "main_agent_tokens": {
                        "prompt": ad.prompt_tokens,
                        "completion": ad.completion_tokens
                    },
                    "sub_agent_total_tokens": sub_agent_tokens,
                    "main_agent_api_calls": getattr(ad, '_api_call_count', 0),
                    "sub_agent_api_calls": sub_agent_api_calls,
                },

                "environment_stats": environment_stats,

                "dynamic_verification_attempts": dynamic_stats,
                "messages": ad.messages,
                "interaction_records": {
                    "step_histories": ad._interaction_records,
                    "mid_to_message": ad._message_map
                },
                "buggy_methods": grade,
                "error_occurred": error_occurred,
                "error_category": error_category,
            }

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            print(f"  ✅ [{label}] {bug_name} saved. Total Time: {total_time:.1f}s")

    except Exception as fatal_error:
        print(f"  ❌ [FATAL] {bug_name}: {fatal_error}")
        if debug:
            traceback.print_exc()

    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as cleanup_error:
                print(f"  ⚠️  Cleanup error for {bug_name}: {cleanup_error}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Main agent model."
    )

    parser.add_argument(
        "--verifier_model",
        type=str,
        default="gpt-4o-mini",
        help="Verifier/sub-agent model."
    )

    parser.add_argument(
        "--projects",
        type=str,
        default="Chart",
        help='e.g., "Chart,Lang,Math" or "all"'
    )

    parser.add_argument(
        "--base_label",
        type=str,
        default="ProbeFL"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="prompts/system_msg_newfl.txt"
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=30
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=2400
    )

    parser.add_argument(
        "--debug",
        action="store_true"
    )

    parser.add_argument("--start_bug", type=int, default=1)
    parser.add_argument("--end_bug", type=int, default=None)

    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = serial). Different bugs run in parallel."
    )

    parser.add_argument(
        "--modes",
        type=str,
        default="all",
        help="Comma-separated modes: full,no_context_bridge,no_reflection,no_dynamic (default: all)"
    )

    args = parser.parse_args()

    if args.modes == "all":
        modes_filter = "all"
    else:
        modes_filter = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.projects.lower() == "all":
        target_projects = list(D4J_PROJECTS.keys())
    else:
        target_projects = [p.strip() for p in args.projects.split(",") if p.strip()]

    all_tasks = []
    for proj in target_projects:
        max_bug = D4J_PROJECTS.get(proj)
        if max_bug is None:
            print(f"⚠️ Unknown project: {proj}, skipping.")
            continue
        start_bug = args.start_bug
        end_bug = args.end_bug if args.end_bug is not None else max_bug
        end_bug = min(end_bug, max_bug)
        for bug_id in range(start_bug, end_bug + 1):
            all_tasks.append(f"{proj}_{bug_id}")

    print(f"🚀 ProbeFL Multi-Project Batch Runner")
    print(f"🤖 Main Model: {args.model}")
    print(f"🧪 Verifier Model: {args.verifier_model}")
    print(f"📌 Budget: {args.budget} | Timeout: {args.timeout}s")
    print(f"📂 Projects: {target_projects}")
    print(f"🔧 Parallel workers: {args.parallel}")
    print(f"📋 Total tasks: {len(all_tasks)}")

    timeout_seconds = args.timeout if args.timeout else None

    if args.parallel <= 1:
        print("▶️  Running in SERIAL mode")
        for bug_name in all_tasks:
            print(f"\n{'='*50}\n📌 Starting {bug_name}\n{'='*50}")

            p = multiprocessing.Process(
                target=worker_process,
                args=(
                    bug_name,
                    args.model,
                    args.verifier_model,
                    args.prompt,
                    args.base_label,
                    args.budget,
                    args.debug,
                    timeout_seconds,
                    modes_filter
                )
            )

            p.start()
            p.join(args.timeout)

            if p.is_alive():
                print(f"⚠️  {bug_name} TIMEOUT. Terminating...")
                worker_pid = p.pid
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()
                    p.join()
                cleanup_environment(pid=worker_pid)
    else:
        print(f"▶️  Running in PARALLEL mode ({args.parallel} workers)")
        print(f"⏰ Deadline per task: {args.timeout}s")

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_bug = {}
            for bug_name in all_tasks:
                future = executor.submit(
                    worker_process,
                    bug_name,
                    args.model,
                    args.verifier_model,
                    args.prompt,
                    args.base_label,
                    args.budget,
                    args.debug,
                    timeout_seconds,
                    modes_filter
                )
                future_to_bug[future] = bug_name

            completed = 0
            failed = 0
            for future in as_completed(future_to_bug):
                bug_name = future_to_bug[future]
                try:
                    future.result(timeout=args.timeout + 60)  # (comment removed)
                    completed += 1
                    print(f"[Main] ✅ {bug_name} completed ({completed}/{len(all_tasks)})")
                except FutureTimeoutError:
                    failed += 1
                    print(f"[Main] ⚠️ {bug_name} TIMEOUT (hard limit)")
                except Exception as e:
                    failed += 1
                    print(f"[Main] ❌ {bug_name} failed: {e}")

            print(f"\n[Main] Done. Completed: {completed}, Failed/Timeout: {failed}, Total: {len(all_tasks)}")


if __name__ == "__main__":
    main()
