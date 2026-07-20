import json
import time
import argparse
import traceback
import hashlib
from copy import deepcopy
from lib import name_utils, llm_utils
from lib.repo_interface import get_repo_interface, get_bug_metadata
from lib.protocol import BaseBackend, StaticD4JBackend, PersistentJDBBackend
import os

RESULT_DIR = './results/'

class AutoDebugger(llm_utils.OpenAIEngine):
    def __init__(self, bug_name, model_type, system_file, backend=None, test_offset=None,
            max_num_tests=None, allow_multi_predictions=False,
            summarize_messages=False, debug=False):
        super().__init__()
        # 注入重构后的后端实现
        self._bug_name = bug_name
        self._model = model_type

        # 使用传入的backend，不再直接使用get_repo_interface
        self.backend = backend
        if hasattr(self.backend, '_ri'):
            self._ri = self.backend._ri
        else:
            self._ri = None
        # self._ri = get_repo_interface(bug_name, **ri_kwargs)

        self._test_offset = test_offset
        self._max_num_tests = max_num_tests
        self._allow_multi_predictions = allow_multi_predictions
        self._summarize_messages = summarize_messages

        # 处理 system_file 路径
        if system_file:
            # 如果是相对路径，转换为相对于 autofl.py 的绝对路径
            if not os.path.isabs(system_file):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self._system_file = os.path.join(script_dir, system_file)
            else:
                self._system_file = system_file
        else:
            self._system_file = None
        
        # 验证文件存在
        if self._system_file and not os.path.exists(self._system_file):
            raise FileNotFoundError(
                f"System prompt file not found: {self._system_file}\n"
                f"Script directory: {os.path.dirname(os.path.abspath(__file__))}\n"
                f"Current working directory: {os.getcwd()}"
            )

        self._debug = debug
        self._function_descriptions = self._get_all_function_descriptions()

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _replace_last_with_memo(self, memo):
        self.messages = self.messages[:-1] # replace recent two queries with memo
        self.messages.append({'role': 'assistant', 'content': 'Summary: ' + memo})

    def _append_to_messages(self, message):
        # to easily control debug behavior
        if self._debug:
            print(message)
        self.messages.append(message)

    @property
    def _system_message(self):
        with open(self._system_file) as f:
            system_message = f.read().strip()
        
        # 如果使用动态调试模式,添加动态工具说明
        if isinstance(self.backend, PersistentJDBBackend) and getattr(self.backend, 'ablation_mode', 'none') != 'no_dynamic':
            system_message += "\n\n## Dynamic Verification Mode Enabled\n"
            system_message += "You can use the `nominate_suspicious_method` tool to verify your hypothesis. "
            system_message += "Once you nominate a method, a specialized sub-agent will perform deep JDB debugging "
            system_message += "and provide you with a comprehensive verification report. "
            system_message += "Use this tool when you have identified a potentially buggy method through static analysis.\n"

        if self._allow_multi_predictions:
            system_message += "\n\nAfter providing this diagnosis, you will be prompted to suggest which methods would be the best locations to be fixed. The answers should be in the form of `ClassName.MethodName(ArgType1, ArgType2, ...)` without commentary (one per line), as your answer will be automatically processed before finally being presented to the user."
        else:
            system_message += "\n\nAfter providing this diagnosis, you will be prompted to suggest which method would be the best location to be fixed. You will provide a single answer, in the form of `ClassName.MethodName(ArgType1, ArgType2, ...)`, as your answer will be automatically processed before finally being presented to the user."
        return system_message

    def _init_interaction_records(self):
        self._mid_map = {} # md5_hash -> mid (message id)
        self._message_map = {} # mid -> message
        self._interaction_records = [] # list of dict

    def _append_to_interaction_records(self, prompt_messages, response_message):
        def _save_message_and_get_mid(message):
            s = json.dumps(message).encode('utf-8')
            md5_hash = hashlib.md5(s).digest()
            if md5_hash not in self._mid_map:
                self._mid_map[md5_hash] = f"m{len(self._mid_map) + 1}"
                self._message_map[self._mid_map[md5_hash]] = deepcopy(message)
            return self._mid_map[md5_hash]

        self._interaction_records.append({
            "prompt_messages": [_save_message_and_get_mid(m) for m in prompt_messages],
            "response_message": _save_message_and_get_mid(response_message)
        })

    def _get_all_function_descriptions(self):
        """合并静态工具与动态工具的描述，返回 OpenAI v1.0 tools 格式"""
        raw_funcs = []
        if self._ri is not None:
            raw_funcs.extend(self._ri.function_descriptions)

        if isinstance(self.backend, PersistentJDBBackend):
            # ===== Ablation: no_dynamic — 移除 nomination 工具，纯静态模式 =====
            ablation = getattr(self.backend, 'ablation_mode', 'none')
            if ablation != 'no_dynamic':
                raw_funcs.extend(PersistentJDBBackend.MAIN_AGENT_FUNCTION_DESCRIPTIONS)

        # 转换为 tools 格式
        return [{"type": "function", "function": f} for f in raw_funcs]

    def startup(self):
        self._init_interaction_records()
        self.messages = []

        self._append_to_messages({'role': 'system', 'content': self._system_message})

        fail_test_signatures = [
            signature for signature in self._ri.failing_test_signatures
            if self._ri.get_test_snippet(signature) is not None
        ]

        if not fail_test_signatures:
            raise ValueError(f'Could not find test snippet for bug {self._bug_name}')

        if self._test_offset is not None:
            # rotate list
            offset = self._test_offset % len(fail_test_signatures)
            fail_test_signatures = fail_test_signatures[offset:] + fail_test_signatures[:offset]

        if self._max_num_tests is not None:
            fail_test_signatures = fail_test_signatures[:self._max_num_tests]

        user_message = f"The test `{fail_test_signatures}` failed.\n"
        test_snippets = "\n\n".join(self._ri.get_test_snippet(signature).rstrip() for signature in fail_test_signatures)
        user_message += f"The test looks like:\n\n```{self._ri.language}\n{test_snippets}\n```\n\n"

        failing_traces = "\n\n".join(self._ri.get_fail_info(signature, minimize=True).rstrip() for signature in fail_test_signatures)
        user_message += f"It failed with the following error message and call stack:\n\n```\n{failing_traces}\n```\n\n"

        user_message += f'Start by calling the `{self._ri.initial_coverage_getter}` function.'

        self._append_to_messages({
            'role': 'user',
            'content': user_message,
        })

        # no-LLM call of first instruction (LLM always calls this anyway)
        fake_call_id = "call_initial"
        self._append_to_messages({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": fake_call_id,
                "type": "function",
                "function": {
                    "name": self._ri.initial_coverage_getter,
                    "arguments": "{}"
                }
            }],
            "function_call": {  # backward compat for stored messages
                "name": self._ri.initial_coverage_getter,
                "arguments": "{}"
            }
        })
        self._append_to_messages({
            "role": "tool",
            "tool_call_id": fake_call_id,
            "content": json.dumps(self._ri.fname2func[self._ri.initial_coverage_getter]())
        })

    def call_function(self, response_message):
        """参考 deepagents 的工具分发机制重构"""
        function_name = response_message["function_call"]["name"]
        # function_to_call = self._ri.fname2func[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        #function_response = function_to_call(**function_args)
        
        result = self.backend.dispatch(function_name, function_args)
        if(result.is_error):
            if self._debug:
                print(f"Error executing {function_name}: {result.output}")
            # 如果后端报错，可以将错误信息返回给 LLM 让其尝试修复指令
            return function_name, result.output
        
        # 尝试解析 JSON
        try:
            parsed_output = json.loads(result.output)
            return function_name, parsed_output
        except json.JSONDecodeError:
            return function_name, result.output

    def step(self, function_call_mode="auto"):
        if self._summarize_messages:
            prompt_messages = self.messages + [{'role': 'system', 'content': 'Summarize the important content of the immediate prior message. If you are unsure of the solution, call a function afterwards. Be concise, but fully qualify all names.'}]
        else:
            prompt_messages = self.messages

        response = self.get_LLM_response(
            model=self._model,
            messages=prompt_messages,
            tools=self._function_descriptions,
            tool_choice=function_call_mode,  # "auto" or "none"
        )

        # [+] 新增 Token 统计
        if "usage" in response:
            self.prompt_tokens += response["usage"].get("prompt_tokens", 0)
            self.completion_tokens += response["usage"].get("completion_tokens", 0)

        if self._summarize_messages:
            llm_summary = response['choices'][0]['message']['content']
            if llm_summary is not None:
                self._replace_last_with_memo(llm_summary)

        response_message = response["choices"][0]["message"]

        if response_message.get("content") is None:
            response_message["content"] = ""

        self._append_to_interaction_records(prompt_messages, response_message)

        # check if GPT wanted to call tools
        tool_calls = response_message.get("tool_calls")
        if tool_calls:
            # normalize: add function_call for backward compat (first tool call only)
            tc_first = tool_calls[0]
            response_message["function_call"] = {
                "name": tc_first["function"]["name"],
                "arguments": tc_first["function"]["arguments"]
            }

            self._append_to_messages(response_message) # extend conversation with assistant's reply

            # 处理所有 tool calls（支持并行调用）
            for tc in tool_calls:
                try:
                    # 为每个 tool call 构造兼容的 function_call 格式来执行
                    fake_msg = {
                        "function_call": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }
                    function_name, function_response = self.call_function(fake_msg)
                except Exception as e:
                    if self._debug or isinstance(e, KeyboardInterrupt):
                        raise e
                    else:
                        function_response = f"Error: {str(e)}"

                tool_message = {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(function_response),
                }
                self._append_to_messages(tool_message)
            return False # not done
        else:
            self._append_to_messages(response_message)  # extend conversation with assistant's reply
            
            # 1. 预算耗尽保护：如果是最后一步 (run 方法传入了 "none")，直接放行进入 finish()
            if function_call_mode == "none":
                return True
                
            # 2. 意图识别：检查文本内容，看 Agent 是不是真的完成了所有调查
            content = response_message.get("content", "").strip()
            if content == "READY" or "ready to conclude" in content.lower() or "final diagnosis" in content.lower():
                return True # 返回 True，正常结束调查循环
                
            # 3. 拦截发呆：说明它既没耗尽预算，也没说准备好，纯粹是忘了输出 JSON！骂醒它！
            nudge_msg = {
                "role": "user",
                "content": (
                    "SYSTEM ALERT: You explained your thoughts but did not output a valid tool call. \n"
                    "- If you need to examine code or verify a hypothesis, you MUST explicitly invoke a function (e.g., `get_code_snippet`, `nominate_suspicious_method`).\n"
                    "- If you have completely finished the investigation and are ready to provide the final buggy method signature, reply EXACTLY with the word 'READY'."
                )
            }
            self._append_to_messages(nudge_msg)
            
            return False # 返回 False，强行让循环继续，带着 Nudge 信息请求下一轮

    def finish(self):
        finishing_string = "Based on the available information, provide the signatures of the most likely culprit methods for the bug. Your answer will be processed automatically, so make sure to only answer with the accurate signatures of all likely culprits (in `ClassName.MethodName(ArgType1, ArgType2, ...)` format), without commentary (one per line). "
        if not self._allow_multi_predictions:
            finishing_string = finishing_string.replace('signatures', 'signature')
            finishing_string = finishing_string.replace('methods', 'method')
            finishing_string = finishing_string.replace(' (one per line)', '')
            finishing_string = finishing_string.replace('all likely culprits', 'the most likely culprit')

        querying_buggy_methods = {
            'role': 'user',
            'content': finishing_string
        }
        self._append_to_messages(querying_buggy_methods)
        response = self.get_LLM_response(
            model=self._model,
            messages=self.messages,
        )

        # [+] 新增 Token 统计
        if "usage" in response:
            self.prompt_tokens += response["usage"].get("prompt_tokens", 0)
            self.completion_tokens += response["usage"].get("completion_tokens", 0)

        response_message = response["choices"][0]["message"]
        self._append_to_messages(response_message)
        content = response_message.get('content')
        if content is None:
            return ""  # 部分模型/proxy 在 tool_calls 场景下 content 可能为 null
        return content.strip()

    def grade(self, answer):
        if self._allow_multi_predictions:
            pred_exprs = answer.splitlines()
        else:
            pred_exprs = [answer]

        matching_method_signatures = {
            pred_expr: self._ri.get_matching_method_signatures(pred_expr)
            for pred_expr in pred_exprs
        }

        grade_result = {}
        for method in self._ri.buggy_method_signatures:
            pred_match = [
                pred_expr for pred_expr in pred_exprs
                if method in matching_method_signatures[pred_expr]
            ]
            grade_result[method] = {
                'is_found': len(pred_match) > 0,
                'matching_answer': pred_match
            }
        return grade_result

    def run(self, budget=10):
        self.startup()
        for i in range(budget):
            if i == budget - 1:
                function_call_mode = "none"
            else:
                function_call_mode = "auto"
            done = self.step(function_call_mode)
            time.sleep(0.1)
            if done:
                break
        final_response = self.finish()
        grade_result = self.grade(final_response)
        return grade_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='gpt-4o-2024-11-20')
    parser.add_argument('--verifier_model', default='gpt-4o-mini')
    parser.add_argument('-b', '--bug_name', default='Chart_1')
    parser.add_argument('-o', '--out', default='test.json')
    parser.add_argument('-p', '--prompt', default='prompts/system_msg_expbug.txt')
    parser.add_argument('-t', '--max_num_tests', default=None, type=int)
    parser.add_argument('--test_offset', default=0, type=int)
    parser.add_argument('--max_budget', default=10, type=int)
    parser.add_argument('--allow_multi_predictions', action="store_true")
    parser.add_argument('--summarize_messages', action="store_true")
    parser.add_argument('--show_line_number', action="store_true")
    parser.add_argument('--postprocess_test_snippet', action="store_true")
    parser.add_argument('--debug', action="store_true")

    parser.add_argument('--dynamic', default=False, action="store_true", help="Whether to use the dynamic JDB backend instead of the static backend. WARNING: the dynamic backend is still in early development and may be unstable.")

    parser.add_argument(
        '--ablation_mode',
        default='none',
        choices=[
            'none',
            'no_context_bridge',
            'no_runtime_fusion',
            'no_fusion',
            'no_reflection',
            'no_dynamic'
        ],
        help='Ablation mode for dynamic verification.'
    )

    args = parser.parse_args()

    initial_cwd = os.getcwd()
    output_file = os.path.abspath(args.out)
    if args.debug:
        print(f"[Debug] Initial working directory: {initial_cwd}")
        print(f"[Debug] Output file will be saved to: {output_file}")

    # 准备后端参数
    ri_kwargs = {
        'show_line_number': True,
        'postprocess_test_snippet': args.postprocess_test_snippet
    }

    # 自动选择 prompt 文件
    if args.dynamic and args.prompt == 'prompts/system_msg_expbug.txt':
        # 如果用户启用了动态模式但没有指定 prompt，自动使用动态 prompt
        # args.prompt = 'prompts/system_msg_dynamic_debug.txt'
        args.prompt = 'prompts/system_msg_newfl.txt'
        print(f"[Info] Dynamic mode enabled, using prompt: {args.prompt}")

    # NO_DYNAMIC 消融模式：自动切换到纯静态 prompt
    if args.ablation_mode == 'no_dynamic' and args.prompt == 'prompts/system_msg_newfl.txt':
        args.prompt = 'prompts/system_msg_static.txt'
        print(f"[Info] no_dynamic ablation: switching to static prompt: {args.prompt}")

    backend = None
    try:
        if args.dynamic:
            backend = PersistentJDBBackend(args.bug_name, **ri_kwargs, debug=args.debug, ablation_mode=args.ablation_mode, verifier_model=args.verifier_model)
            # 调用 startup 来准备环境并连接到执行阶段
            res = backend.startup()
            if res.is_error:
                # 修复隐患 1：如果启动失败，直接抛出异常，交给 finally 清理
                raise RuntimeError(f"Failed to start PersistentJDBBackend: {res.output}")
        else:
            # 注意：这里使用 StaticD4JBackend 作为默认后端，如果需要使用 PersistentJDBBackend，请修改这里的实例化
            backend = StaticD4JBackend(args.bug_name, **ri_kwargs)

        ad = AutoDebugger(args.bug_name, args.model, args.prompt,
            backend=backend,
            test_offset=args.test_offset,
            max_num_tests=args.max_num_tests,
            allow_multi_predictions=args.allow_multi_predictions,
            summarize_messages=args.summarize_messages,
            debug=args.debug
        )

        grade = None
        error_occurred = False
        
        start_time = time.time()

        try:
            grade = ad.run(args.max_budget)
        except Exception as e:
            error_occurred = True
            grade = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            if args.debug:
                print(f"\n{'='*60}\n错误详情:\n{traceback.format_exc()}\n{'='*60}\n")
        
        end_time = time.time()

        # 1. 计算总耗时 (Wall-clock time)
        total_time = end_time - start_time 

        # 2. 初始化统计变量
        dynamic_stats = []
        total_verify_time = 0.0
        sub_agent_total_tokens = {"prompt": 0, "completion": 0}
        sub_agent_api_calls = 0

        # 3. 从后端提取动态验证的深度数据 (RQ2, RQ3, RQ4 的核心)
        if isinstance(backend, PersistentJDBBackend) and hasattr(backend, "dynamic_stats"):
            dynamic_stats = backend.dynamic_stats
            for stat in dynamic_stats:
                # 累加所有子 Agent 的验证耗时
                total_verify_time += stat["metadata"].get("verify_time_seconds", 0)
                # 累加所有子 Agent 消耗的 Token
                sub_agent_total_tokens["prompt"] += stat["sub_agent_tokens"].get("prompt", 0)
                sub_agent_total_tokens["completion"] += stat["sub_agent_tokens"].get("completion", 0)
                # 统计子 Agent API 调用次数
                for msg in stat.get("sub_agent_messages", []):
                    if isinstance(msg, dict) and msg.get("function_call"):
                        sub_agent_api_calls += 1

        # 4. 计算静态搜索耗时 (总时间 - 验证时间)
        static_search_time = total_time - total_verify_time
        
        # 5. 主 Agent 的 Token 消耗
        main_agent_tokens = {
            "prompt": ad.prompt_tokens,
            "completion": ad.completion_tokens
        }

        # 6. [新增] 收集 bug 元数据
        try:
            bug_metadata = get_bug_metadata(args.bug_name,
                show_line_number=True,
                postprocess_test_snippet=args.postprocess_test_snippet)
        except Exception:
            bug_metadata = {}

        # 7. [新增] 收集环境准备统计（仅动态模式）
        environment_stats = {}
        if isinstance(backend, PersistentJDBBackend):
            try:
                environment_stats = backend.get_environment_stats()
            except Exception:
                pass

        # 8. [新增] 统计错误类别（用于按错误类型分层分析）
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

        # 9. 最终保存逻辑
        try:
            out_dir = os.path.dirname(output_file)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump({
                    'bug_id': args.bug_name,
                    'model': args.model,
                    'main_model': args.model,
                    'verifier_model': args.verifier_model,
                    'ablation_mode': args.ablation_mode,
                    'total_time_seconds': total_time,

                    # [bug 元数据] 搜索空间、测试数量等
                    'bug_metadata': bug_metadata,

                    # [开销细分] 对应论文 RQ4
                    'cost_breakdown': {
                        'static_search_time': static_search_time,
                        'dynamic_verify_time': total_verify_time,
                        'main_agent_tokens': main_agent_tokens,
                        'sub_agent_total_tokens': sub_agent_total_tokens,
                        'main_agent_api_calls': getattr(ad, '_api_call_count', 0),
                        'sub_agent_api_calls': sub_agent_api_calls,
                    },

                    # [环境准备统计] checkout, compile, JVM 重启次数
                    'environment_stats': environment_stats,

                    # [动态验证详情] 包含 Metadata, verification_result, Dialog_History
                    'dynamic_verification_attempts': dynamic_stats,

                    'messages': ad.messages, # 主 Agent 对话
                    'interaction_records': {
                        "step_histories": ad._interaction_records,
                        "mid_to_message": ad._message_map
                    },
                    'buggy_methods': grade,
                    'error_occurred': error_occurred,
                    'error_category': error_category,
                }, f, indent=4)
            print(f"\n✅ 完整实验数据已保存到: {output_file}")

        except Exception as save_error:
            print(f"\n❌ 保存结果失败: {save_error}")
            traceback.print_exc()

    except Exception as fatal_error:
        print(f"\n❌ 发生严重错误: {fatal_error}")
        if args.debug:
            traceback.print_exc()

    finally:
        # 修复点 3：把切回目录放在 finally 的第一步，保证绝对执行
        current_cwd = os.getcwd()
        if current_cwd != initial_cwd:
            if args.debug:
                print(f"[Debug] Changing back to initial directory: {initial_cwd}")
            os.chdir(initial_cwd) # 强制离开 /tmp 目录
            
        # 此时进程已经安全回到 autofl 目录，再执行 shutdown 删掉 /tmp 就是绝对安全的
        if backend and hasattr(backend, 'shutdown'):
            if args.debug: print(f"[Debug] Executing final backend shutdown...")
            try:
                backend.shutdown()
            except Exception as cleanup_error:
                print(f"\n 清理后端时出错: {cleanup_error}")