import json
import os
import argparse
import re
import collections
from tqdm import tqdm
from copy import deepcopy
from lib.repo_interface import get_repo_interface

import ast
import javalang
from lib.name_utils import get_method_name

def file2bug(json_file):
    if not json_file.endswith(".json"):
        return None
    try:
        return os.path.basename(json_file).removeprefix('XFL-').removesuffix('.json')
    except:
        return None

def get_prediction_status(raw_prediction, error_occurred=False):
    if error_occurred:
        return "OtherError"
    if isinstance(raw_prediction, str): 
        error_message = raw_prediction
        if "openai.error.InvalidRequestError" not in error_message:
            return "OtherError"
        else:
            return "InvalidRequestError"
    else:
        return "OK"

def parse_response(response):
    return [
        expr.removeprefix('`').removesuffix('`')
        for expr in response.strip().splitlines()
    ]

def print_divider():
    print("-"*60)

def compute_autofl_scores(result_dirs, project=None, verbose=False):
    json_status = {}
    score_results = {}
    no_pred_runs = {}
    
    efficiency_stats = {} 

    for result_dir in result_dirs:
        file_iterator = sorted(os.listdir(result_dir))
        if verbose:
            print(f"Processing {result_dir}...")
            file_iterator = tqdm(file_iterator)
        for fname in file_iterator:
            bug_name = file2bug(fname)
            if bug_name is None:
                continue            
            if project and not bug_name.startswith(project):
                continue

            json_status[bug_name] = json_status.get(bug_name, {"OK": [], "OtherError": [], "InvalidRequestError": []}) 
            score_results[bug_name] = score_results.get(bug_name, {})   
            
            # === [新增指标初始化] ===
            efficiency_stats[bug_name] = efficiency_stats.get(bug_name, {
                'time': 0.0, 
                'main_prompt': 0, 'main_comp': 0, 
                'sub_prompt': 0, 'sub_comp': 0, 
                'valid_count': 0,
                'num_methods_verified': 0,
                'guardrail_rejects': 0,
                'tool_errors': 0,
                'total_jdb_steps': 0
            })

            status_result, score_result = json_status[bug_name], score_results[bug_name]
    
            fpath = os.path.join(result_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                autofl_data = json.load(f)

            prediction = autofl_data.get("buggy_methods", "")
            error_occurred = autofl_data.get("error_occurred", False)
            pred_status = get_prediction_status(prediction, error_occurred)
            
            status_result[pred_status].append(fpath)

            if pred_status == "OK":
                stat = efficiency_stats[bug_name]
                # 兼容新老版本的耗时键名
                stat['time'] += autofl_data.get('total_time_seconds', autofl_data.get('time_seconds', 0.0))
                
                # === [修改：优先从 cost_breakdown 读取 Token] ===
                cost = autofl_data.get("cost_breakdown", {})
                main_tokens = cost.get("main_agent_tokens", autofl_data.get('main_agent_tokens', {'prompt': 0, 'completion': 0}))
                stat['main_prompt'] += main_tokens.get('prompt', 0)
                stat['main_comp'] += main_tokens.get('completion', 0)

                sub_tokens = cost.get("sub_agent_total_tokens", None)
                if sub_tokens is not None:
                    stat['sub_prompt'] += sub_tokens.get("prompt", 0)
                    stat['sub_comp'] += sub_tokens.get("completion", 0)
                else:
                    # 兼容老版本的通过 parse message 读取
                    sub_prompt, sub_comp = 0, 0
                    for msg in autofl_data.get("messages", []):
                        if (msg.get("role") == "function" and msg.get("name") == "nominate_suspicious_method") \
                           or (msg.get("role") == "tool"):  # 兼容新 tools API 格式
                            content = msg.get("content", "")
                            try:
                                json_start = content.find('{')
                                json_end = content.rfind('}')
                                if json_start != -1 and json_end != -1:
                                    report_json = json.loads(content[json_start:json_end+1])
                                    tokens = report_json.get("Tokens", {})
                                    sub_prompt += tokens.get("prompt", 0)
                                    sub_comp += tokens.get("completion", 0)
                            except Exception:
                                pass
                    stat['sub_prompt'] += sub_prompt
                    stat['sub_comp'] += sub_comp

                # === [新增：提取微观行为指标] ===
                attempts = autofl_data.get("dynamic_verification_attempts", [])
                stat['num_methods_verified'] += len(attempts)
                for attempt in attempts:
                    meta = attempt.get("metadata", {})
                    stat['guardrail_rejects'] += meta.get("guardrail_reject_count", 0)
                    stat['tool_errors'] += meta.get("tool_error_count", 0)
                    for traj in meta.get("probing_trajectories", []):
                        stat['total_jdb_steps'] += len(traj)

                stat['valid_count'] += 1

            if pred_status != "OK":
                continue 

            final_response = autofl_data["messages"][-1].get("content")
            if final_response is None:
                final_response = ""
            pred_exprs = parse_response(final_response)

            ri = get_repo_interface(bug_name)
            predicted_methods = {}
            for pred_expr in pred_exprs:
                for method in ri.get_matching_method_signatures(pred_expr):
                    predicted_methods[method] = predicted_methods.get(method, [])
                    predicted_methods[method].append(pred_expr)

            if not predicted_methods:
                no_pred_runs[bug_name] = no_pred_runs.get(bug_name, 0) + 1
                continue

            for method in predicted_methods:
                if method not in score_result:
                    score_result[method] = {"score": 0, "count": 0, "exprs": {}}
                score_result[method]["count"] += 1
                score_result[method]["score"] += 1/len(predicted_methods)
                score_result[method]["exprs"][fpath] = predicted_methods[method]

    for bug_name in score_results:
        ri = get_repo_interface(bug_name)
        all_methods = ri.method_signatures
        score_result = score_results[bug_name]
        num_all_runs = sum([len(json_status[bug_name][s]) for s in json_status[bug_name]])
        if num_all_runs == 0: continue

        for method in sorted(all_methods): 
            if method not in score_result:
                score_result[method] = {"score": 0, "count": 0, "exprs": {}}
            score_result[method]["score"] /= num_all_runs

    return json_status, score_results, efficiency_stats

def get_seen_methods_from_msgs(ri, messages, language):
    seen_method_sigs = []
    
    for msg in messages:
        func_call = msg.get('function_call')
        if func_call and func_call.get('name') in ('get_code_snippet', 'get_comments'):
            try:
                args = json.loads(func_call.get('arguments', '{}'))
                if 'signature' in args:
                    seen_method_sigs.append(args['signature'])
            except Exception:
                pass

    all_seen_method_names = []

    for msg in messages:
        content_data = None
        if msg['role'] == 'user':
            raw_content = msg.get('content', '')
            if f"```{language}" not in raw_content:
                continue
            content_data = raw_content
        elif (msg['role'] == 'function' and msg.get('name') == 'get_code_snippet') \
             or msg['role'] == 'tool':  # 兼容新 tools API 格式
            try:
                content_data = json.loads(msg.get('content', '""'))
            except Exception:
                continue
        else:
            continue

        if not isinstance(content_data, str):
            continue
        
        norm_content = ''
        for line in content_data.splitlines():
            norm_content += re.sub(r'^\s*\d+\s\:\s', '', line) + '\n' 
        if '```' in norm_content:
            try:
                norm_content = norm_content.split('```')[1].lstrip(language)
            except IndexError:
                pass 

        if language == "java":
            try:
                parsed_method = javalang.parse.parse(norm_content)  
                method_call_nodes = [e[1] for e in parsed_method.filter(javalang.tree.MethodInvocation)]
                all_seen_method_names += [e.member for e in method_call_nodes]
            except Exception:
                continue
        elif language == "python":
            try:
                parsed_method = ast.parse(norm_content)
                method_call_nodes = [e for e in ast.walk(parsed_method) if isinstance(e, ast.Call)]
                all_seen_method_names += [ast.unparse(e.func) for e in method_call_nodes]
            except Exception:
                continue

    candidates = {}
    for seen_method in all_seen_method_names:
        try:
            seen_exact_match, seen_match_candidates = ri.get_matching_method_or_candidates(seen_method+'()')
            if seen_match_candidates is not None:
                candidates[seen_method] = candidates.get(
                    seen_method,
                    [m["signature"] for m in seen_match_candidates]
                )
            else:
                candidates[seen_method] = candidates.get(
                    seen_method, seen_exact_match["signature"]
                )
            seen_method_sigs += [
                sig for sig in candidates[seen_method] if get_method_name(sig)==seen_method]
        except Exception:
            continue
            
    return seen_method_sigs

def add_auxiliary_scores(json_files, autofl_scores, language, default_aux_score=None, verbose=False):
    autofl_scores_aug = deepcopy(autofl_scores) 
    bug_name_iterator = autofl_scores_aug.keys()
    
    if verbose:
        print("Computing auxiliary scores...")
        bug_name_iterator = tqdm(bug_name_iterator)

    for bug_name in bug_name_iterator:
        ri = get_repo_interface(bug_name)
        if language == 'java':
            snippet_path = f"data/defects4j/{bug_name}/snippet.json"
        elif language == 'python':
            snippet_path = f"data/bugsinpy/{bug_name}/snippet.json"
        else:
            raise ValueError(f'Unknown language {language}')

        with open(snippet_path, "r", encoding="utf-8") as f:
            method_data = json.load(f)
            num_failing_tests = {
                m["signature"]: m["num_failing_tests"] if "num_failing_tests" in m else 0
                for m in method_data
            }

        seen_methods = []
        for fpath in json_files[bug_name]["OK"]:
            with open(fpath, 'r', encoding='utf-8') as f:
                autofl_data = json.load(f)
            messages = autofl_data.get("messages", [])
            seen_methods += get_seen_methods_from_msgs(ri, messages, language)
        seen_method_counter = collections.Counter(seen_methods)

        for method in autofl_scores_aug[bug_name]:
            if default_aux_score is None:
                if autofl_scores_aug[bug_name][method]["count"] > 0:
                    aux_score = (num_failing_tests.get(method, 0), 0) 
                else:
                    aux_score = (num_failing_tests.get(method, 0), seen_method_counter.get(method, 0))
            else:
                aux_score = default_aux_score
                
            autofl_scores_aug[bug_name][method]["aux_score"] = aux_score
            
    return autofl_scores_aug

def assign_rank(autofl_scores):
    autofl_scores_rank = deepcopy(autofl_scores)
    for bug_name in autofl_scores_rank:
        sort_keys = [] 
        for i, method in enumerate(autofl_scores_rank[bug_name]):
            score = autofl_scores_rank[bug_name][method]["score"]
            sort_key = [-score]
            aux_score = autofl_scores_rank[bug_name][method]["aux_score"]
            if isinstance(aux_score, tuple) or isinstance(aux_score, list):
                sort_key += list([-s for s in aux_score])
            elif isinstance(aux_score, float) or isinstance(aux_score, int):
                sort_key.append(-aux_score)
            else:
                raise Exception(f"Unsupported aux score type: {aux_score}")
            sort_key += [i, method]
            sort_keys.append(tuple(sort_key))
            
        for r, sort_item in enumerate(sorted(sort_keys)):
            method = sort_item[-1]
            autofl_scores_rank[bug_name][method]["rank"] = r + 1
    return autofl_scores_rank

def get_buggy_method_ranks(method_scores, key="autofl_rank"):
    buggy_method_ranks = {}
    for bug_name in method_scores:
        ri = get_repo_interface(bug_name)
        buggy_method_ranks[bug_name] = {}
        for method in ri.buggy_method_signatures:
            rank = method_scores[bug_name][method]["rank"] if method in method_scores[bug_name] else None
            buggy_method_ranks[bug_name][method] = {key: rank}
    return buggy_method_ranks

def calculate_acc(buggy_method_ranks, key="autofl_rank", n=1):
    acc = 0
    for bug_name in buggy_method_ranks:
        ranks = [
            buggy_method_ranks[bug_name][method][key]
            for method in buggy_method_ranks[bug_name]
            if buggy_method_ranks[bug_name][method][key] is not None
        ]
        if not ranks:
            continue
        if any([r <= n for r in ranks]):
            acc += 1
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JADE Enhanced Score & Efficiency Computer")
    parser.add_argument('result_dirs', nargs="+", type=str)
    parser.add_argument('--output', '-o', type=str, default="scores.json")
    parser.add_argument('--project', '-p', type=str, default=None)
    parser.add_argument('--language', '-l', type=str, default="java")
    parser.add_argument('--verbose', '-v', action="store_true")
    parser.add_argument('--minimize', '-m', action="store_true")
    parser.add_argument('--aux', '-a', action="store_true")
    args = parser.parse_args()

    # 1. 提取所有状态并评分 (此处融合了微观行为统计)
    json_files, autofl_scores, efficiency_stats = compute_autofl_scores(args.result_dirs, args.project, args.verbose)

    if not autofl_scores:
        print("⚠️ No valid results found. Exiting.")
        exit(0)

    # 2. 赋予辅助分数
    if args.aux:
        method_scores = add_auxiliary_scores(json_files, autofl_scores, args.language, verbose=args.verbose)
    else:
        method_scores = add_auxiliary_scores(json_files, autofl_scores, args.language, default_aux_score=0, verbose=args.verbose)
    
    # 3. 排序并比对 Ground Truth
    method_scores = assign_rank(method_scores)
    buggy_method_ranks = get_buggy_method_ranks(method_scores, key="autofl_rank")

    # ============================================================
    # 汇总与计算均值
    # ============================================================
    summary = {"total": len(method_scores)}
    for n in range(1, 11):
        summary[f"acc@{n}"] = calculate_acc(buggy_method_ranks, key="autofl_rank", n=n)

    avg_time = 0
    avg_main_prompt, avg_main_comp = 0, 0
    avg_sub_prompt, avg_sub_comp = 0, 0
    avg_methods_verified, avg_jdb_steps, avg_guardrail, avg_tool_error = 0, 0, 0, 0

    valid_bugs = [b for b in efficiency_stats if efficiency_stats[b]['valid_count'] > 0]
    
    if valid_bugs:
        total_time = sum(efficiency_stats[b]['time'] / efficiency_stats[b]['valid_count'] for b in valid_bugs)
        avg_time = total_time / len(valid_bugs)
        
        avg_main_prompt = sum(efficiency_stats[b]['main_prompt'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)
        avg_main_comp = sum(efficiency_stats[b]['main_comp'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)
        avg_sub_prompt = sum(efficiency_stats[b]['sub_prompt'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)
        avg_sub_comp = sum(efficiency_stats[b]['sub_comp'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)

        avg_methods_verified = sum(efficiency_stats[b]['num_methods_verified'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)
        avg_jdb_steps = sum(efficiency_stats[b]['total_jdb_steps'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)
        avg_guardrail = sum(efficiency_stats[b]['guardrail_rejects'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)
        avg_tool_error = sum(efficiency_stats[b]['tool_errors'] / efficiency_stats[b]['valid_count'] for b in valid_bugs) / len(valid_bugs)

    # 打印格式化输出
    print("\n" + "="*60)
    print("🎯 FL Accuracy Summary")
    print("="*60)
    print(json.dumps(summary, indent=4))
    
    print("\n" + "="*60)
    print("⏱️  Efficiency & Cost Summary (Average per Bug)")
    print("="*60)
    print(f"Total Valid Bugs Assessed : {len(valid_bugs)}")
    print(f"Avg Localization Time     : {avg_time:.2f} seconds")
    print(f"Avg Main Agent Prompt     : {int(avg_main_prompt)} tokens")
    print(f"Avg Main Agent Completion : {int(avg_main_comp)} tokens")
    print(f"Avg Sub Agent Prompt      : {int(avg_sub_prompt)} tokens")
    print(f"Avg Sub Agent Completion  : {int(avg_sub_comp)} tokens")
    total_avg_tokens = avg_main_prompt + avg_main_comp + avg_sub_prompt + avg_sub_comp
    print(f"--------------------------------------------------")
    print(f"TOTAL Avg Tokens / Bug    : {int(total_avg_tokens)} tokens")
    
    print("\n" + "="*60)
    print("🕵️  Dynamic Verification Micro-Behaviors (Average)")
    print("="*60)
    print(f"Methods Nominated         : {avg_methods_verified:.2f} methods")
    print(f"Total JDB Steps           : {avg_jdb_steps:.1f} steps")
    print(f"Guardrail Rejects         : {avg_guardrail:.2f} times")
    print(f"Tool Errors               : {avg_tool_error:.2f} times")
    print("="*60 + "\n")

    # 保存扩充后的 JSON
    data = {
        "summary": summary,
        "efficiency_summary": {
            "avg_time_sec": avg_time,
            "avg_main_prompt": avg_main_prompt,
            "avg_main_comp": avg_main_comp,
            "avg_sub_prompt": avg_sub_prompt,
            "avg_sub_comp": avg_sub_comp,
            "avg_methods_verified": avg_methods_verified,
            "avg_jdb_steps": avg_jdb_steps,
            "avg_guardrail_rejects": avg_guardrail,
            "avg_tool_errors": avg_tool_error
        },
        "buggy_methods": buggy_method_ranks,
        "bug_efficiency_details": efficiency_stats
    }
    
    if not args.minimize:
        data["predictions"] = method_scores

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
