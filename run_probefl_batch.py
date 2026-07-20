import os
import sys
import time
import json
import argparse
import traceback
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

from autofl import AutoDebugger
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
    """断点续传检查器：只有成功写出且没有 error 才算完成"""
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
    清理僵尸进程（并行安全版本）。
    - 不传 pid：不做任何全局清理（并行时 backend.shutdown() 已处理各自进程）
    - 传 pid：只清理指定进程树（用于超时后强制终止）
    """
    if pid is None:
        # 并行模式下不做全局清理，避免误杀其他 worker
        return
    try:
        import signal
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass  # 进程已经不存在


def has_available_snippet(bug_name):
    """
    预检查：如果 snippet 或 failing test snippet 不可用，就跳过，不生成结果报告。
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
    从 backend 中提取动态验证统计。
    注意：调用前必须保证 backend.dynamic_stats 是当前 mode 独立的。
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
    一个 worker 负责一个 bug。
    对同一个 bug：
      - checkout/compile 一次；
      - 连续跑 full + 三个消融；
      - 最后统一 shutdown。

    timeout_seconds: 每个 bug 的最大运行时间（秒），超时后跳过剩余 mode
    """

    # 日志前缀：并行模式下区分不同 worker 的输出
    log_prefix = f"[{bug_name}]"

    # 1. snippet 预检查：没有 snippet 就不生成报告
    if not has_available_snippet(bug_name):
        return

    # 每个 bug 独立计算 deadline
    deadline = time.time() + timeout_seconds if timeout_seconds else None

    # 2. 定义 ProbeFL 的实验模式
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
    # 根据 modes_filter 筛选要跑的模式
    if modes_filter == "all":
        modes = all_modes
    else:
        modes = {k: v for k, v in all_modes.items() if k in modes_filter}

    # 3. 检查是否所有 mode 都已经完成
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

    # 4. 初始化 backend：同一个 bug 只 checkout/compile 一次
    backend = None
    try:
        backend = PersistentJDBBackend(
            bug_name,
            debug=debug,
            show_line_number=True,
            postprocess_test_snippet=True,
            ablation_mode="none",
            verifier_model=verifier_model  # 需要你在 PersistentJDBBackend 中加这个参数
        )

        res = backend.startup()
        if res.is_error:
            print(f"  ❌ [Backend] Startup failed for {bug_name}: {res.output}")
            return

        # 5. 依次运行 full + 三个消融
        for mode_name, cfg in modes.items():
            # 并行模式下的超时保护：如果剩余时间不够，跳过后续 mode
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

            # 每个 mode 前必须清空动态统计，防止跨 mode 污染
            backend.ablation_mode = ablation_mode
            backend.dynamic_stats = []

            # 每个 mode 前重置 JDB session，保证状态干净
            reset_res = backend.reset_session()
            # reset_session 当前没有返回值，所以这里不判断也可以

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

            # [新增] 统计子 Agent API 调用次数（从 sub_agent_messages 中统计 function_call）
            sub_agent_api_calls = 0
            for stat in dynamic_stats:
                for msg in stat.get("sub_agent_messages", []):
                    if isinstance(msg, dict) and msg.get("function_call"):
                        sub_agent_api_calls += 1

            # [新增] 收集 bug 元数据
            try:
                bug_metadata = get_bug_metadata(bug_name,
                    show_line_number=True,
                    postprocess_test_snippet=True)
            except Exception:
                bug_metadata = {}

            # [新增] 收集环境准备统计
            try:
                environment_stats = backend.get_environment_stats()
            except Exception:
                environment_stats = {}

            # [新增] 错误类别
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

                # [新增] bug 元数据
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

                # [新增] 环境准备统计
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

    # 解析 modes_filter
    if args.modes == "all":
        modes_filter = "all"
    else:
        modes_filter = [m.strip() for m in args.modes.split(",") if m.strip()]

    if args.projects.lower() == "all":
        target_projects = list(D4J_PROJECTS.keys())
    else:
        target_projects = [p.strip() for p in args.projects.split(",") if p.strip()]

    # 收集所有待处理任务
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

    # 每个 bug 的超时时间（worker 内部使用）
    timeout_seconds = args.timeout if args.timeout else None

    if args.parallel <= 1:
        # ===== 串行模式（保持原有行为） =====
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
        # ===== 并行模式 =====
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
                    future.result(timeout=args.timeout + 60)  # 额外 60s 缓冲
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
