import json
import re
from lib import llm_utils
from lib.protocol import PersistentJDBBackend

class JDBVerifyAgent(llm_utils.OpenAIEngine):
    def __init__(self, jdb_backend, model, max_steps=25): # (comment removed)
        super().__init__()
        self.jdb = jdb_backend
        self.model = model
        self.max_steps = max_steps
        self.jdb_tools = PersistentJDBBackend.JDB_FUNCTION_DESCRIPTIONS
        self.sub_prompt_tokens = 0
        self.sub_completion_tokens = 0
        self.ablation_mode = jdb_backend.ablation_mode

    def verify(self, method_id, method_code, hypothesis, test_info,
               known_static_facts=None, target_lines=None, variables_of_interest=None, begin_line=None, verification_goal=None):
        
        metadata = {
            "target_lines": target_lines or [],
            "attempts_count": 0,
            "failure_patterns": [],           # (comment removed)
            "probing_trajectories": [],       # (comment removed)
            "context_fusion_stats": {"locals_count": 0, "fields_count": 0, "call_count": 0},
            "anchoring_hit_lines": [],         # (comment removed)
            "guardrail_reject_count": 0,  # (comment removed)
            "tool_error_count": 0         # (comment removed)
        }

        class_name = method_id.split('(')[0].rsplit('.', 1)[0]
        lines_str = ", ".join(map(str, target_lines)) if target_lines else "None"
        vars_str = ", ".join(variables_of_interest) if variables_of_interest else "None"

        verification_tools_raw = self.jdb_tools + [{
            "name": "submit_verification_report",
            "description": "Submit your final verification report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bug_type": {
                        "type": "string",
                        "enum": ["CRASH_CONFIRMED", "LOGIC_ERROR", "HELPER_METHOD_SYMPTOM", "NO_BUG", "UNREACHED_TARGET", "INCONCLUSIVE"],
                        "description": "Select 'HELPER_METHOD_SYMPTOM' if this is a utility method doing what it's told, but the OUTCOME is wrong because the CALLER passed bad data.Select 'UNREACHED_TARGET' if the program finished or crashed before reaching the target method/lines."
                    },
                    "root_cause_explanation": {
                        "type": "string", 
                        "description": "Explanation. If 'HELPER_METHOD_SYMPTOM' is selected, explicitly name the Caller method that is likely responsible."
                    },
                    "observed_evidence": {
                        "type": "string",
                        "description": "Key variable states observed."
                    }
                },
                "required": ["bug_type", "root_cause_explanation", "observed_evidence"]
            }
        }]

        verification_tools = [{"type": "function", "function": f} for f in verification_tools_raw]

        # ==========================================
        # Adaptive Reflection Loop
        # ==========================================
        max_attempts = 1 if self.ablation_mode == "no_reflection" else 2

        reflection_guidance = ""  # (comment removed)
        suggested_breakpoints = target_lines if target_lines else [] # (comment removed)

        for attempt in range(1, max_attempts + 1):
            
            metadata["attempts_count"] += 1
            current_trajectory = []
            metadata["probing_trajectories"].append(current_trajectory)

            print(f"\n[VerifyAgent] Starting Verification Attempt {attempt}/{max_attempts}")
            self.jdb.reset_session()
            
            initial_status = "System initialized.\n"
            breakpoint_set = False

            if suggested_breakpoints:
                for line in suggested_breakpoints:
                    res = self.jdb.dispatch("jdb_set_breakpoint", {
                        "location": f"{class_name}:{line}"
                    })
                    initial_status += f"- {res.output}\n"
                    if not res.is_error:
                        breakpoint_set = True
            if not breakpoint_set:
                res = self.jdb.dispatch("jdb_set_breakpoint", {
                    "location": method_id
                })
                initial_status += f"- {res.output}\n"
                if not res.is_error:
                    breakpoint_set = True
            if not breakpoint_set and begin_line:
                for offset in range(0, 4):
                    fallback_line = int(begin_line) + offset
                    res = self.jdb.dispatch("jdb_set_breakpoint", {
                        "location": f"{class_name}:{fallback_line}"
                    })
                    initial_status += f"- fallback line {fallback_line}: {res.output}\n"
                    if not res.is_error:
                        breakpoint_set = True
                        break

            run_res = self.jdb.dispatch("jdb_run_to_breakpoint", {})
            initial_status += f"\nCurrent Status:\n{run_res.output}"
            
            
            output_lower = str(run_res.output).lower()
            if run_res.is_error or "exited" in output_lower or "finished" in output_lower:
                is_vm_suspended = False
            else:
                is_vm_suspended = True  # (comment removed)


            if run_res.is_error and "CRITICAL" in run_res.output:
                 reflection_guidance = f"Strategy failed immediately: {run_res.output}\nSUGGESTION: The crash happens before hitting your breakpoints. Try setting a breakpoint earlier in the method."
            
            strategy_section = """
HUMAN EXPERT DEBUGGING COGNITIVE MODEL (MUST FOLLOW):

1. **BACKWARD REASONING ([cleaned])**:
   Do not start stepping from line 1. Look at the Hypothesis and the error message. Identify the EXACT line in the code that returns the wrong value or throws the error. Set your FIRST breakpoint at or right before that critical line.

2. **THE "EXPECTATION vs. REALITY" CHECK ([cleaned])**:
   Whenever you use `jdb_get_locals` or `jdb_print_variable`, you MUST compare the output with your hypothesis. 
   - Ask yourself: "Is this variable's value correct for this failing test case?" 
   - If it is correct, the bug is further down.
   - If it is wrong, the bug happened BEFORE this line.

3. **THE "LOCAL DATA-FLOW" HEURISTIC ([cleaned])**:
   If you hit a breakpoint and observe a variable holds a wrong/suspicious value, DO NOT guess how it got there. 
   ACTION: Look at the code snippet to find WHERE this variable was assigned (e.g., `this.pattern = ...` or `int x = compute()`). Set a NEW breakpoint at that assignment line, and `jdb_run_to_breakpoint` again to see what parameters created this bad value.

4. **THE "TEST CASE DELTA" RULE ([cleaned])**:
   The code may run multiple times (e.g., first with a passing string, then with a failing string). If you hit a breakpoint and the variables show the *passing* case (e.g., `value="'d'd"` instead of `value="'d'd'"`), DO NOT step. Call `jdb_run_to_breakpoint` immediately to skip to the failing execution.

5. **THE "CHAIN OF COMMAND" & "STATE ORIGIN" RULE**:
   - If the method is a helper, check `jdb_where`. 
   - [NEW] If the bug is caused by a wrong value in an object FIELD (e.g., `this.parsePattern`), do not just look at the stack. Object fields are initialized in constructors or `init()` methods. 
   - ACTION: If a field is suspicious, report that the bug likely exists in the initialization phase (e.g., the `init` method).

6. **THE "EVIDENCE FIRST" LAW ([cleaned]) [MANDATORY]**:
    - You CANNOT submit a report with empty "Evidence".
    - You MUST use `jdb_print_variable` to capture the specific value that matches the 'VERIFICATION GOAL'.
    - Example: If the goal is "Check for null", you MUST execute `print node.getFirstChild()` and see "null" in the output BEFORE submitting.
    - Copy the EXACT JDB output into the `Evidence` field of your report.  

7. **THE "GROUND TRUTH" RULE**:
   - The real-world test FAILED. If your JDB session suggests the code is "Correct" or "Passes", YOU ARE LIKELY WRONG. 
   - You must find the discrepancy. Why does the real test throw an Exception while your JDB session seems fine? 
   - Check if the method is called multiple times. The bug might only trigger in the LAST call.
               
8. **THE "FAULT ORIGIN" RULE (Defects4J Standard)**:
   - Do NOT blame the test case. The bug ALWAYS exists within the project's source code. 
   - Even if a method's code executes logically "as written", if it fails to handle edge-case inputs gracefully and returns a value that directly triggers the test failure downstream, this method is the root cause. Mark it as a LOGIC_ERROR.

9. **THE "BURDEN OF PROOF" RULE (CRITICAL)**:
   - You CANNOT submit `Bug_Type: LOGIC_ERROR` merely because you reached the end of the method or cannot observe the return value.
   - To confirm a bug, your `observed_evidence` MUST explicitly show either:
     (A) an abnormal runtime value, such as Exception, NaN, unexpected null, or out-of-bounds; OR
     (B) a normal runtime value that triggers an incorrect branch/return according to the source code and failing test expectation.
   - Example for (B): if `dataset` is a valid object but the code executes `if (dataset != null) return emptyResult;`, the normal value `dataset != null` is valid evidence because it triggers the wrong branch.
   - If the variables show normal values AND no incorrect control-flow consequence can be identified, submit `NO_BUG` and advise the Main Agent to check related caller/callee methods.
"""

            if reflection_guidance:
                strategy_section += f"""
                **ADAPTIVE STRATEGY UPDATE (Based on Previous Failure)**:
                {reflection_guidance}
"""
            else:
                strategy_section += "\n- Use `jdb_run_to_breakpoint` carefully to skip irrelevant loops."

            system_prompt = f"""
                You are an expert JDB Debugging Agent.
                TARGET METHOD: {method_id}
                VERIFICATION GOAL: {verification_goal}
                SUSPICIOUS LINES: [{lines_str}]
                VARIABLES: [{vars_str}]
                ==== STATIC HINTS FROM MAIN AGENT (MAY BE WRONG) ====
                The following information is a hypothesis or observation produced by static analysis.
                You MUST verify it against the actual method code, runtime state, and call context.
                Do NOT treat it as ground truth.
                {known_static_facts}
                =====================================================
                CRITICAL: You are a runtime observer, NOT the final judge.
                - LOGIC_ERROR requires CONCRETE evidence of wrong behavior (e.g., a variable has a value that directly contradicts expected computation, or a null field that should have been initialized).
                - Do NOT classify as LOGIC_ERROR just because you reached the method, stepped through some lines, or observed normal-looking variable values.
                - If you are UNSURE whether the observed behavior is actually buggy, prefer INCONCLUSIVE.
                {strategy_section}
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": (
                    f"Test: {test_info}\nMethod Code:\n{method_code}\n"
                    f"Hypothesis: {hypothesis}\n\n{initial_status}"
                )}
            ]

            attempt_failed_reason = None # (comment removed)
            last_executed_tool = None # (comment removed)
            consecutive_same_tool = 0  # (comment removed)
            last_tool_name = None

            for step in range(self.max_steps):
                response = self.get_LLM_response(
                    messages=messages,
                    model=self.model,
                    tools=verification_tools
                )

                if "usage" in response:
                    self.sub_prompt_tokens += response["usage"].get("prompt_tokens", 0)
                    self.sub_completion_tokens += response["usage"].get("completion_tokens", 0)

                msg = response["choices"][0]["message"]
                if msg.get("content") is None: msg["content"] = ""
                messages.append(msg)

                tool_calls = msg.get("tool_calls")
                call = msg.get("function_call")
                if tool_calls and not call:
                    tc = tool_calls[0]
                    msg["function_call"] = {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                    call = msg["function_call"]
                if call:
                    func_name = call["name"]
                    args_str = call["arguments"]

                    current_trajectory.append(func_name)    # (comment removed)

                    if func_name == last_tool_name:
                        consecutive_same_tool += 1
                    else:
                        consecutive_same_tool = 1
                        last_tool_name = func_name
                    if consecutive_same_tool > 5 and func_name in ("jdb_print_variable", "jdb_get_locals", "jdb_next", "jdb_list"):
                        messages.append({
                            "role": "user",
                            "content": (
                                f"⚠️ STALENESS WARNING: You have called `{func_name}` {consecutive_same_tool} times in a row. "
                                "You are likely stuck in a loop. STOP and re-evaluate:\n"
                                "- Have you found the bug? If yes, call `submit_verification_report` NOW.\n"
                                "- If not, try a DIFFERENT approach: check `jdb_where` for the call stack, "
                                "or set a breakpoint at a different location.\n"
                                "- Do NOT call the same tool again without a clear reason."
                            )
                        })
                        consecutive_same_tool = 0

                    tc_id = tool_calls[0]["id"] if tool_calls else None

                    if tool_calls and len(tool_calls) > 1:
                        for tc_extra in tool_calls[1:]:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc_extra["id"],
                                "content": json.dumps({"status": "SKIPPED", "message": "Batch tool calls not supported in this mode."})
                            })

                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str

                        if func_name == "submit_verification_report":
                            b_type = args.get("bug_type", "NO_BUG")
                            explanation = args.get("root_cause_explanation", "")
                            evidence = args.get("observed_evidence", {})

                            bug_found_types = ["CRASH_CONFIRMED", "LOGIC_ERROR", "HELPER_METHOD_SYMPTOM"]
                            
                            is_evidence_empty = not evidence or (isinstance(evidence, dict) and not evidence) or (isinstance(evidence, str) and not evidence.strip())

                            if b_type in bug_found_types and is_evidence_empty:
                                metadata["guardrail_reject_count"] += 1

                                warn_msg = (
                                    "SYSTEM REJECT: You submitted a verification report claiming a bug (Type: " + b_type + "), "
                                    "but 'observed_evidence' is EMPTY. This is strictly forbidden.\n"
                                    "You cannot guess based on code reading. You MUST prove it with runtime values.\n"
                                    "ACTION REQUIRED: \n"
                                    "1. Use `jdb_print_variable` to capture the suspicious variable (e.g., check for null, wrong size, etc.).\n"
                                    "2. Call `submit_verification_report` AGAIN, putting that EXACT output into 'observed_evidence'."
                                )
                                print(f"    -> Blocked Report: Agent claimed bug but provided empty evidence.")
                                
                                messages.append({"role": "tool", "tool_call_id": (tc_id or ""), "content": warn_msg})
                                continue # (comment removed)

                            is_confirmed = b_type in bug_found_types
                            metadata["failure_patterns"].append("SUCCESS")

                            return json.dumps({
                                "Bug_Confirmed": is_confirmed,
                                "Bug_Type": b_type,
                                "Explanation": explanation,
                                "Evidence": evidence,
                                "Tokens": {"prompt": self.sub_prompt_tokens, "completion": self.sub_completion_tokens},
                                "Metadata": metadata,
                                "Dialog_History": messages
                            }, indent=2, ensure_ascii=False)
                        print(f"  [Attempt {attempt} - Step {step+1}] Calling Tool: {func_name}")
                        
                        observation_tools = ["jdb_get_locals", "jdb_print_variable", "jdb_next", "jdb_list", "jdb_where"]

                        if func_name in observation_tools and last_executed_tool == "jdb_set_breakpoint":
                            warn_msg = "❌ SYSTEM BLOCK: You just set a new breakpoint, but you didn't RUN the program! The program has not reached your new breakpoint yet. 👉 ACTION REQUIRED: Call `jdb_run_to_breakpoint` right now."
                            print(f"    -> Blocked {func_name}: Forcing Agent to run after setting breakpoint.")
                            messages.append({"role": "tool", "tool_call_id": (tc_id or ""), "content": warn_msg})
                            continue

                        if func_name in observation_tools and not is_vm_suspended:
                            warn_msg = "❌ SYSTEM BLOCK: The program is NOT suspended at a breakpoint (it likely exited or crashed). You cannot inspect variables. 👉 ACTION REQUIRED: Set your breakpoints and call `jdb_run_to_breakpoint` to restart execution."
                            print(f"    -> Blocked {func_name}: VM is not suspended.")
                            messages.append({"role": "tool", "tool_call_id": (tc_id or ""), "content": warn_msg})
                            continue

                        res = self.jdb.dispatch(func_name, args)
                        last_executed_tool = func_name # (comment removed)
                        output_str = str(res.output)

                        if res.is_error:
                            metadata["tool_error_count"] += 1

                        output_lower = output_str.lower()
                        
                        if func_name == "jdb_get_locals" and not res.is_error:
                            try:
                                parsed_locals = json.loads(output_str)
                                metadata["context_fusion_stats"]["locals_count"] += len(parsed_locals.get("locals", []))
                                metadata["context_fusion_stats"]["fields_count"] += len(parsed_locals.get("fields", []))
                                metadata["context_fusion_stats"]["call_count"] += 1
                            except: pass # (comment removed)

                        if func_name in ["jdb_run_to_breakpoint", "jdb_next"] and not res.is_error:
                            m = re.search(r'(\d+,\d+|\d+)\s+=>', output_str)
                            if m:
                                hit_line = int(m.group(1).replace(',', ''))
                                metadata["anchoring_hit_lines"].append(hit_line)
                        # =====================================


                        if func_name in ["jdb_run_to_breakpoint", "jdb_next"]:
                            if res.is_error or "exited" in output_lower or "finished" in output_lower or "fatal" in output_lower:
                                is_vm_suspended = False
                            else:
                                is_vm_suspended = True 

                        # ==========================================
                        # ==========================================
                        
                        # Session lost / JDB timeout: stop wasting steps.
                        if "[SESSION_LOST]" in output_str or "[JDB_TIMEOUT]" in output_str:
                            metadata["failure_patterns"].append("SESSION_LOST")
                            attempt_failed_reason = "SESSION_LOST"
                            messages.append({"role": "tool", "tool_call_id": (tc_id or ""), "content": output_str})

                            if self.ablation_mode == "no_reflection" or attempt == max_attempts:
                                return json.dumps({
                                    "Bug_Confirmed": False,
                                    "Bug_Type": "INCONCLUSIVE",
                                    "Explanation": "Verification failed because the JDB session was lost or timed out.",
                                    "Evidence": output_str,
                                    "Tokens": {"prompt": self.sub_prompt_tokens, "completion": self.sub_completion_tokens},
                                    "Metadata": metadata,
                                    "Dialog_History": messages
                                }, indent=2, ensure_ascii=False)

                            reflection_guidance = (
                                "PREVIOUS FAILURE [SESSION_LOST]: The JDB session ended unexpectedly or timed out. "
                                "In the next attempt, restart from a valid method-entry breakpoint and avoid invalid line breakpoints."
                            )
                            break


                        if res.is_error and "CRITICAL" in str(res.output):
                            print(f"Exception Detected! Capturing trace...")
                            messages.append({"role": "tool", "tool_call_id": (tc_id or ""), "content": str(res.output)})
                            
                            if self.ablation_mode == "no_reflection":
                                print(f"  [Ablation] no_reflection: stopping after first failed attempt.")
                                attempt_failed_reason = "CRASH"
                                metadata["failure_patterns"].append("CRASH")
                                break

                            attempt_failed_reason = "CRASH"
                            metadata["failure_patterns"].append("CRASH")
                            crash_details = str(res.output).split('\n')[1] if '\n' in str(res.output) else str(res.output)
                            reflection_guidance = (
                                f"PREVIOUS FAILURE [CRASH]: The program crashed with '{crash_details}' when executing '{func_name}'.\n"
                                f"DYNAMIC STRATEGY ADJUSTMENT: The crash confirms the exact bug location. "
                                f"In this attempt, DO NOT use `run` past this point. Set a breakpoint right before the crash and use `jdb_get_locals` to capture the state that causes the crash."
                            )
                            break

                        elif "[TERMINAL_STATE]" in str(res.output):
                            print(f"  Program exited! Triggering reflection...")
                            
                            if self.ablation_mode == "no_reflection":
                                print(f"  [Ablation] no_reflection: stopping after premature exit.")
                                attempt_failed_reason = "EXIT"
                                metadata["failure_patterns"].append("EXIT")
                                break

                            attempt_failed_reason = "EXIT"
                            metadata["failure_patterns"].append("EXIT")
                            if attempt == 1:
                                reflection_guidance = (
                                    f"PREVIOUS FAILURE [PREMATURE EXIT]: The program finished execution without hitting the buggy state during '{func_name}'.\n"
                                    f"DYNAMIC STRATEGY ADJUSTMENT: Two possibilities: 1) You overshot the breakpoint. 2) This method is NEVER executed dynamically (e.g., cached or wrong path).\n"
                                    f"In this Attempt 2, set the breakpoint at the very beginning of the method. If it exits again, YOU MUST conclude that the method is unreached."
                                )
                            else:
                                reflection_guidance = (
                                    f"FATAL DYNAMIC FAILURE: The program exited again.\n"
                                    f"CONCLUSION: This target method is DEFINITELY NOT EXECUTED in this test run.\n"
                                    f"ACTION REQUIRED: Immediately call `submit_verification_report` and set `bug_type` to 'UNREACHED_TARGET'."
                                )
                            break # (comment removed)

                        messages.append({"role": "tool", "tool_call_id": (tc_id or ""), "content": str(res.output)})

                    except Exception as e:
                        messages.append({"role": "tool", "tool_call_id": (tc_id or ""), "content": f"Tool Execution Error: {str(e)}"})
                # else:
                #     messages.append({"role": "user", "content": "Please inspect the state using JDB tools."})
                else:
                    warn_msg = (
                        "SYSTEM ALERT: You just talked without calling any JDB tools! "
                        "You have a limited budget. You MUST output a `function_call` in every step (e.g., `jdb_next`, `jdb_print_variable`, or `submit_verification_report`)."
                    )
                    messages.append({"role": "user", "content": warn_msg})
                    print(f"    -> Blocked: Agent rambled without tool call.")
                

            if not attempt_failed_reason:
                print(f"  [Attempt {attempt}] Steps exhausted! Forcing final report submission...")
                metadata["failure_patterns"].append("BUDGET_EXHAUSTED")

                if attempt == max_attempts:
                    print(" Critical: Forcing Agent to submit partial report!")
                    
                    final_tool_set = [t for t in verification_tools if t.get("function", {}).get("name") == "submit_verification_report"]
                    
                    bail_out_message = {
                        "role": "user", 
                        "content": "CRITICAL: STOP DEBUGGING. You have ZERO steps left. You MUST immediately call `submit_verification_report` with your findings so far. Do not explain, just call the function."
                    }
                    messages.append(bail_out_message)
                    
                    try:
                        final_response = self.get_LLM_response(
                            messages=messages, 
                            model=self.model,
                            tools=final_tool_set
                        )

                        if "usage" in final_response:
                            self.sub_prompt_tokens += final_response["usage"].get("prompt_tokens", 0)
                            self.sub_completion_tokens += final_response["usage"].get("completion_tokens", 0)
                            
                        final_msg = final_response["choices"][0]["message"]

                        if final_msg.get("tool_calls") and not final_msg.get("function_call"):
                            tc = final_msg["tool_calls"][0]
                            final_msg["function_call"] = {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }

                        if final_msg.get("function_call") and final_msg["function_call"]["name"] == "submit_verification_report":
                            args_str = final_msg["function_call"]["arguments"]
                            args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            print(" Successfully extracted partial report.")

                            b_type = args.get("bug_type", "NO_BUG")
                            is_confirmed = b_type in ["CRASH_CONFIRMED", "LOGIC_ERROR", "HELPER_METHOD_SYMPTOM"]
                            evidence = args.get("observed_evidence", {})

                            is_evidence_empty = not evidence or (isinstance(evidence, dict) and not evidence) or (isinstance(evidence, str) and not evidence.strip())

                            if is_confirmed and is_evidence_empty:
                                print("  [Guardrail] Forced submission claimed bug but provided empty evidence. Overriding to INCONCLUSIVE.")
                                is_confirmed = False
                                b_type = "INCONCLUSIVE"
                                explanation = "[SYSTEM OVERRIDE] Agent claimed bug but provided empty evidence at timeout. " + args.get("root_cause_explanation", "")
                            else:
                                if is_evidence_empty:
                                    b_type = "INCONCLUSIVE" 
                                explanation = args.get("root_cause_explanation", "Forced submission due to budget limits.")
                            # ===================================================

                            return json.dumps({
                                    "Bug_Confirmed": is_confirmed,
                                    "Bug_Type": b_type,
                                    "Explanation": explanation,
                                    "Evidence": args.get("observed_evidence", {}),
                                    "Tokens": {"prompt": self.sub_prompt_tokens, "completion": self.sub_completion_tokens},
                                    "Metadata": metadata,
                                    "Dialog_History": messages
                                }, indent=2, ensure_ascii=False)
                        else:
                            print(" Agent refused to submit report. Generating fallback report.")
                            return json.dumps({
                                "Bug_Confirmed": False,
                                "Bug_Type": "INCONCLUSIVE",
                                "Explanation": "Verification partial: Agent exhausted budget and failed to submit a structured report.",
                                "Evidence": {},
                                "Tokens": {"prompt": self.sub_prompt_tokens, "completion": self.sub_completion_tokens},
                                "Metadata": metadata,
                                "Dialog_History": messages
                            }, indent=2)
                        
                    except Exception as e:
                        print(f" Error during forced submission: {e}")
                        return json.dumps({
                            "Bug_Confirmed": False,
                            "Bug_Type": "INCONCLUSIVE",
                            "Explanation": f"Verification failed during forced report submission: {str(e)}",
                            "Evidence": {},
                            "Tokens": {"prompt": self.sub_prompt_tokens, "completion": self.sub_completion_tokens},
                            "Metadata": metadata,
                            "Dialog_History": messages
                        }, indent=2)
                    
                else:
                    reflection_guidance = (
                        "PREVIOUS FAILURE: You ran out of steps (Budget Exhausted). \n"
                        "ANALYSIS: You likely acted like a novice: either stepping (`next`) too many times inside a loop, or repetitively listing/printing the same information.\n"
                        "NEW STRATEGY: In this attempt, act like a senior developer. Look at the code, identify the most critical line, set a breakpoint there directly, and run to it. Skip irrelevant executions."
                    )


        metadata["failure_patterns"].append("INCONCLUSIVE")
        return json.dumps({
            "Bug_Confirmed": False,
            "Bug_Type": "INCONCLUSIVE",
            "Explanation": "Verification inconclusive: Unable to gather evidence after multiple attempts and budget exhaustion.",
            "Evidence": {},
            "Tokens": {"prompt": self.sub_prompt_tokens, "completion": self.sub_completion_tokens},
            "Metadata": metadata,
            "Dialog_History": messages
        }, indent=2)
