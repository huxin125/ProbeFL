#!/usr/bin/env python3
"""第三轮重试：对失败 bug 进行 2 并行重试"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

FAILED_LIST = "/tmp/failed_round4.txt"
LOG_FILE = "results/rerun_round4.log"
RESULT_DIR = "results/ProbeFL_54_FULL/gpt-5.4-2026-03-05"
PARALLEL = 2

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_one(bug_name, idx, total):
    out_json = f"{RESULT_DIR}/{bug_name}.json"
    log(f"START {bug_name} ({idx}/{total})")

    cmd = [
        "python3", "-u", "autofl.py",
        "-m", "gpt-5.4-2026-03-05",
        "--verifier_model", "gpt-5.4-2026-03-05",
        "-b", bug_name,
        "-p", "prompts/system_msg_newfl.txt",
        "-o", out_json,
        "--max_budget", "10",
        "--show_line_number",
        "--postprocess_test_snippet",
        "--allow_multi_predictions",
        "--dynamic",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        # append output to log
        with open(LOG_FILE, "a") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(result.stderr)
        if result.returncode == 0:
            log(f"✅ {bug_name} ({idx}/{total})")
        else:
            log(f"❌ {bug_name} ({idx}/{total}) [rc={result.returncode}]")
    except subprocess.TimeoutExpired:
        log(f"⏰ {bug_name} ({idx}/{total}) TIMEOUT (1h)")
    except Exception as e:
        log(f"💥 {bug_name} ({idx}/{total}) {e}")

def main():
    with open(FAILED_LIST) as f:
        bugs = [line.strip() for line in f if line.strip()]

    total = len(bugs)
    log(f"重跑 Round 3 | {datetime.now()} | 并行={PARALLEL}")
    log(f"共 {total} bugs")

    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        futures = {}
        for i, bug in enumerate(bugs, 1):
            future = executor.submit(run_one, bug, i, total)
            futures[future] = bug

        completed = 0
        for future in as_completed(futures):
            completed += 1
            # future.result() already logged in run_one

    log(f"完成 {datetime.now()}")

if __name__ == "__main__":
    main()
