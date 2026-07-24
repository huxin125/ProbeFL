import abc
from collections import defaultdict
import json
import pexpect
import re
import time
import os
from .schema import ExecutionResult
from lib.repo_interface import get_repo_interface
import subprocess
import shutil
import socket
import fcntl
from contextlib import closing

def get_free_port(retries=5):
    """Get a free port from the OS with retry to prevent parallel TOCTOU race."""
    for attempt in range(retries):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
        time.sleep(0.05)
        return port
    raise RuntimeError(f"Failed to find a free port after {retries} attempts")

class BaseBackend(abc.ABC):
    """Backend protocol service base class."""

    @abc.abstractmethod
    def dispatch(self, func_name: str, kwargs: dict) -> ExecutionResult:
        """Dispatch and execute tool commands."""
        pass

class StaticD4JBackend(BaseBackend):
    """Static D4J backend protocol service."""

    def __init__(self, bug_name:str, **ri_kwargs):
        self._ri = get_repo_interface(bug_name, **ri_kwargs)

    def dispatch(self, func_name: str, kwargs: dict) -> ExecutionResult:
        try:
            func = self._ri.fname2func.get(func_name)
            if not func:
                return ExecutionResult(output=f"Function '{func_name}' not found.", exit_code=1, is_error=True)
            
            result = func(**kwargs)

            if isinstance(result, (dict, list, defaultdict)):
                if isinstance(result, defaultdict):
                    result = dict(result)  # defaultdict -> dict
                output = json.dumps(result, ensure_ascii=False)
            elif isinstance(result, str):
                output = result
            else:
                output = str(result)
            
            return ExecutionResult(output=output, exit_code=0, is_error=False)
        
        except Exception as e:
            return ExecutionResult(output=str(e), exit_code=1, is_error=True)
        
class PersistentJDBBackend(BaseBackend):
    """
    [cleaned] JDB [cleaned]
    [cleaned] pexpect [cleaned]，[cleaned]
    """
    JDB_FUNCTION_DESCRIPTIONS = [
        {
            "name": "jdb_set_breakpoint",
            "description": "Set a breakpoint at a specific method or line. Use this before running the program.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Explain WHY this specific line is critical. What logic is triggered here that matches the bug hypothesis?"
                    },
                    "location": {
                        "type": "string",
                        "description": "Method signature (e.g., 'com.example.MyClass.myMethod') or file:line (e.g., 'MyClass.java:42')"
                    }
                },
                "required": ["thought","location"]
            }
        },
        {
            "name": "jdb_run_to_breakpoint",
            "description": "Run the program until it hits a breakpoint. If you are ALREADY at a breakpoint but the current state is not the bug-triggering one, call this to CONTINUE execution until the method is hit again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "What state do you expect the VM to be in when it stops? If you are continuing from a previous hit, explain why the previous state wasn't the one that triggers the bug."
                    }
                },
                "required": ["thought"]
            }
        },
        {
            "name": "jdb_get_locals",
            "description": "Get all local variables in the current stack frame. ALWAYS call this before printing variables.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "jdb_print_variable",
            "description": "Print the value of a specific variable. The variable MUST exist in the current locals (call jdb_get_locals first to see available variables).",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "CRITICAL: State WHAT value you EXPECT this variable to have if the code were working correctly. (e.g., 'Expected: index < length, because...'). This makes the bug obvious when the actual value appears."
                    },
                    "variable": {
                        "type": "string",
                        "description": "The EXACT variable name from the locals list (case-sensitive)"
                    }
                },
                "required": ["thought","variable"]
            }
        },
        {
            "name": "jdb_next",
            "description": "Execute the next line of code (step over method calls). Use this to move forward in execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "What specific change are you looking for in the next line? If you are just trying to reach a far-away line, use a breakpoint instead of multiple 'next' calls."
                    }
                },
                "required": ["thought"]
            }
        },
        {
            "name": "jdb_list",
            "description": "Show the source code around the current execution point.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "jdb_where",
            "description": "Print the current system stack trace (Call Stack). USE THIS to identify who called the current method, especially when you are debugging a helper/private method.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "jdb_track_local_variable",
            "description": "Automatically step forward (up to 20 steps) and track how a specific local variable changes. Highly recommended for debugging loops or tracking state evolution. DO NOT use manual 'next' in a loop, use this instead!",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Why is the EVOLUTION of this variable important? (e.g., 'I want to see exactly which iteration of the loop makes the pointer null')."
                    },
                    "variable": {
                        "type": "string",
                        "description": "The EXACT variable name to track (e.g., 'i', 'node', 'this.pattern')"
                    }
                },
                "required": ["thought","variable"]
            }
        }
    ]

    MAIN_AGENT_FUNCTION_DESCRIPTIONS = [
        {
            "name": "nominate_suspicious_method",
            "description": "Nominate a suspicious Java method for dynamic verification. A sub-agent will use JDB to verify your hypothesis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method_signature": {
                        "type": "string",
                        "description": "The full method signature, e.g., 'org.apache.commons.math3.complex.Complex.reciprocal()'"
                    },
                    "suspected_issue": {
                        "type": "string",
                        "description": "Detailed hypothesis of why this method is buggy and what to look for."
                    },
                    "known_static_facts": {
                        "type": "string",
                        "description": "Initial observations from static code. DO NOT treat these as absolute truths; they are just hints for the sub-agent to verify."
                    },
                    "target_lines": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Specific line numbers inside the method where the bug likely triggers (e.g., [234, 237])."
                    },
                    "variables_of_interest": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key variables to observe at those lines to prove the hypothesis (e.g., ['real', 'imaginary', 'divisor'])."
                    },
                    "verification_goal": {
                        "type": "string",
                        "description": "The EXACT condition to prove the bug. E.g., 'Check if variable 'regex' is null' or 'Verify if loop index 'i' exceeds 10'."
                    }
                },
                "required": ["method_signature", "suspected_issue", "known_static_facts","target_lines", "variables_of_interest", "verification_goal"]
            }
        }
    ]

    def __init__(self, bug_name:str, debug: bool=True, ablation_mode="none", verifier_model: str = "gpt-4o-mini", **ri_kwargs):
        self.bug_name = bug_name
        self._ri = get_repo_interface(bug_name, **ri_kwargs)
        self.session = None # (comment removed)
        self.prompt = r'(?:^|\r\n)(>|[\w\.-]+\[\d+\])\s+'
        self.test_process = None # (comment removed)
        self._debug = debug
        self.playground_path = f"/tmp/)}_buggy_{os.getpid()}"
        self.classpath = ""
        self.src_dir = ""
        self.test_src_dir = ""
        self.fail_test_file = None
        self.fail_test_line = None
        self._allowed_locals = set() # (comment removed)
        self._java_cmd_template = []
        self.current_port = None
        self.ablation_mode = ablation_mode # "none", "no_fusion", "no_reflection"
        self.verifier_model = verifier_model
        self._checkout_time_seconds = 0.0
        self._compile_time_seconds = 0.0
        self._jvm_restart_count = 0

    def startup(self):
        """Initialize debug environment: checkout, setup, connect to execution."""
        try:
            self._initial_cwd = os.getcwd()  # (comment removed)
            # if os.path.exists(self.playground_path):
            #     shutil.rmtree(self.playground_path)

            needs_checkout = not os.path.exists(self.playground_path)

            
            if needs_checkout:
                if(self._debug): print(f"--- [Backend] Checkout {self.bug_name} ---")
                project, bug_id = self.bug_name.split("_")
                checkout_cmd = f"defects4j checkout -p {project} -v {bug_id}b -w {self.playground_path}"
                lock_path = f"/tmp/
                _t0 = time.time()
                with open(lock_path, 'w') as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    subprocess.run(checkout_cmd.split(), check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._checkout_time_seconds = time.time() - _t0
            else:
                if self._debug: print(f"--- [Backend] {self.bug_name} already exists, skipping checkout. ---")

            os.chdir(self.playground_path)

            if needs_checkout:
                if self._debug: print(f"--- [Backend] Compiling Source ---")
                _t0 = time.time()
                subprocess.run("defects4j compile".split(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self._compile_time_seconds = time.time() - _t0
            else:
                if self._debug: print(f"--- [Backend] Skipping compilation for speed. ---")
            if self._debug: print(f"--- [Backend] Get Classpath and SrcDir ---")
            cp_proc = subprocess.run("defects4j export -p cp.test".split(), check=True, capture_output=True, text=True)
            self.classpath = cp_proc.stdout.strip()
            src_proc = subprocess.run("defects4j export -p dir.src.classes".split(), check=True, capture_output=True, text=True)
            self.src_dir = os.path.join(self.playground_path, src_proc.stdout.strip())
            if self._debug: print(f"--- [Backend] Src Dir: {self.src_dir} ---")

            try:
                test_src_proc = subprocess.run(
                    "defects4j export -p dir.src.tests".split(),
                    check=True,
                    capture_output=True,
                    text=True
                )
                self.test_src_dir = os.path.join(self.playground_path, test_src_proc.stdout.strip())
                if self._debug:
                    print(f"--- [Backend] Test Src Dir: {self.test_src_dir} ---")
            except Exception as e:
                self.test_src_dir = ""
                if self._debug:
                    print(f"--- [Backend] Failed to get test src dir: {e} ---")

            fail_test = self._ri.failing_test_signatures[0] # (comment removed)

            try:
                fail_info = self._ri.get_fail_info(fail_test, minimize=True)
                m = re.search(r'\(([^:]+\.java):(\d+)\)', fail_info)
                if m:
                    self.fail_test_file = m.group(1)
                    self.fail_test_line = int(m.group(2))
                    if self._debug:
                        print(f"--- [Backend] Failure Location: {self.fail_test_file}:{self.fail_test_line} ---")
            except Exception as e:
                if self._debug:
                    print(f"--- [Backend] Failed to parse failure location: {e} ---")

            test_class_clean = fail_test.split("(")[0] # (comment removed)
            if "::" in test_class_clean:
                test_class_clean = test_class_clean.split("::")[0] # (comment removed)
            else:
                if test_class_clean.count(".") > 1:
                    test_class_clean = test_class_clean.rsplit(".", 1)[0]
            
            if self._debug: print(f"--- [Backend] Failing Test Class: {test_class_clean} ---")
            self._java_cmd_template = [
                "java",
                "-Xmx2g",  # (comment removed)
                "-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address={port}", # (comment removed)
                "-cp", self.classpath,
                "org.junit.runner.JUnitCore",
                test_class_clean
            ]

            self._start_jdb_session()   # (comment removed)

            return ExecutionResult(output="JDB session started and connected successfully.", is_error=False)
        except Exception as e:
            self.shutdown() # (comment removed)
            return ExecutionResult(output=f"Startup Error: {str(e)}", is_error=True)

    def _start_jdb_session(self):
        """Start JVM and JDB attach logic, auto-retry on port conflict."""
        self._jvm_restart_count += 1

        max_retries = 3
        for retry in range(max_retries):
            self.current_port = get_free_port()

            java_cmd = [
                arg.format(port=self.current_port) if "{port}" in arg else arg
                for arg in self._java_cmd_template
            ]

            if self._debug and retry > 0:
                print(f"--- [Backend] Retry {retry}/{max_retries} (port {self.current_port}) ---")

            stderr_target = subprocess.PIPE
            stdout_target = None if self._debug else subprocess.DEVNULL
            self.test_process = subprocess.Popen(
                java_cmd,
                stdout=stdout_target,
                stderr=stderr_target,
                text=True
            )

            jvm_ready = False
            for _ in range(30):  # (comment removed)
                time.sleep(1)
                if self.test_process.poll() is not None:
                    stderr_output = ""
                    try:
                        stderr_output = self.test_process.stderr.read()
                    except Exception:
                        pass
                    if self._debug:
                        print(f"--- [Backend] JVM crashed during startup!")
                        print(f"    Exit code: {self.test_process.returncode}")
                        print(f"    Stderr: {stderr_output[:500]}")
                    if retry < max_retries - 1:
                        break  # (comment removed)
                    else:
                        raise RuntimeError(
                            f"JVM crashed during startup (exit code {self.test_process.returncode}). "
                            f"Stderr: {stderr_output[:300]}"
                        )
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect(('localhost', self.current_port))
                    s.close()
                    jvm_ready = True
                    break
                except (socket.error, ConnectionRefusedError, OSError):
                    continue

            if not jvm_ready:
                if self._debug:
                    print(f"--- [Backend] JVM port {self.current_port} not ready after 30s")
                if retry < max_retries - 1:
                    self._terminate_processes()
                    continue
                else:
                    raise RuntimeError(
                        f"JVM did not start listening on port {self.current_port} after 30s"
                    )

            if self._debug: print(f"--- [Backend] JVM ready on port {self.current_port}, attaching JDB ---")
            source_paths = [self.src_dir]
            if self.test_src_dir:
                source_paths.append(self.test_src_dir)
            sourcepath = ":".join(source_paths)
            self.session = pexpect.spawn(
                f"jdb -sourcepath {sourcepath} -attach localhost:{self.current_port}",
                encoding='utf-8',
                timeout=20
            )

            try:
                self.session.expect(self.prompt, timeout=15)

                sync_token = "JDB_SYNC_ACK"
                self.session.sendline(f"print \"{sync_token}\"")
                self.session.expect(sync_token)
                self.session.expect(self.prompt, timeout=10)

                health_token = "JVM_HEALTH_OK"
                self.session.sendline(f"print \"{health_token}\"")
                idx = self.session.expect([health_token, pexpect.TIMEOUT, pexpect.EOF], timeout=10)
                if idx != 0:
                    raise RuntimeError("JVM health check failed: cannot execute code")
                self.session.expect(self.prompt, timeout=10)

                return  # (comment removed)
            except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF) as e:
                stderr_output = ""
                try:
                    stderr_output = self.test_process.stderr.read()
                except Exception:
                    pass
                if self._debug:
                    print(f"--- [Backend] JDB attach/handshake failed (port {self.current_port})")
                    print(f"    JVM exit code: {self.test_process.returncode if self.test_process.poll() is not None else 'still running'}")
                    print(f"    JVM stderr: {stderr_output[:300] if stderr_output else '(empty)'}")
                if retry < max_retries - 1:
                    self._terminate_processes()
                    time.sleep(0.5)
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to attach JDB after {max_retries} attempts. "
                        f"Last error on port {self.current_port}: {e}. "
                        f"JVM stderr: {stderr_output[:200] if stderr_output else '(empty)'}"
                    )

    def _terminate_processes(self):
        """Kill process only, keep files."""
        if self.session:
            try:
                self.session.sendline("exit")
                self.session.close()
            except:
                pass
            finally:
                self.session = None

        if self.test_process:
            try:
                self.test_process.terminate()
                self.test_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.test_process.kill()
                    self.test_process.wait(timeout=5)
                except Exception:
                    pass
            except Exception:
                pass
            finally:
                self.test_process = None

    def reset_session(self):
        """Reset debug session: ensure VerifyAgent gets clean context."""
        if self._debug: print(f"--- [Backend] Resetting JDB Session ---")
        self._terminate_processes()
        self._allowed_locals = set()    # (comment removed)
        self._start_jdb_session()       # (comment removed)
    
    def shutdown(self):
        """Clean up resources, close JDB connection and test process."""
        if self._debug: print(f"--- [Backend] Tearing down environment ---")
        self._terminate_processes()     # (comment removed)
        if hasattr(self, '_initial_cwd'):
            try:
                os.chdir(self._initial_cwd)
            except OSError:
                pass
        if os.path.exists(self.playground_path):
            if self._debug: print(f"--- [Backend] Cleaning workspace: {self.playground_path} ---")
            shutil.rmtree(self.playground_path, ignore_errors=True)

    def get_environment_stats(self):
        """Return environment setup and JVM lifecycle statistics."""
        return {
            "checkout_time_seconds": self._checkout_time_seconds,
            "compile_time_seconds": self._compile_time_seconds,
            "jvm_restart_count": self._jvm_restart_count,
        } 

    def dispatch(self, func_name: str, kwargs: dict) -> ExecutionResult:
        if func_name == "nominate_suspicious_method":
            return self._handle_nomination(kwargs)

        static_func = self._ri.fname2func.get(func_name)
        if static_func:
            try:
                result = static_func(**kwargs)
                if isinstance(result, (dict, list, defaultdict)):
                    if isinstance(result, defaultdict):
                        result = dict(result)
                    output = json.dumps(result, ensure_ascii=False)
                elif isinstance(result, str):
                    output = result
                else:
                    output = str(result)
                return ExecutionResult(output=output, is_error=False)
            except Exception as e:
                return ExecutionResult(output=str(e), is_error=True)

        if not self.session or not self.session.isalive():
            jvm_status = "unknown"
            if self.test_process:
                if self.test_process.poll() is not None:
                    stderr_tail = ""
                    try:
                        remaining = self.test_process.stderr.read()
                        if remaining:
                            stderr_tail = remaining[-200:]
                    except Exception:
                        pass
                    jvm_status = f"exited with code {self.test_process.returncode}. stderr: {stderr_tail}"
                else:
                    jvm_status = "still running"
            return ExecutionResult(
                output=f"[SESSION_LOST] JDB session is not active. JVM status: {jvm_status}",
                is_error=True
            )

        try:
            res = None
            if func_name == "jdb_set_breakpoint":
                res = self._set_breakpoint(kwargs.get("location"))
            elif func_name == "jdb_run_to_breakpoint":
                res = self._run_to_breakpoint()
            elif func_name == "jdb_print_variable":
                res = self._print_variable(kwargs.get("variable"))
            elif func_name == "jdb_get_locals":
                res = self._get_locals()
            elif func_name == "jdb_next":
                res = self._step_next()
            elif func_name == "jdb_list":
                res = self._list_code()
            elif func_name == "jdb_where":
                res = self._get_stack_trace()
            elif func_name == "jdb_track_local_variable": # (comment removed)
                res = self._track_local_variable(kwargs.get("variable"))
            else:
                return ExecutionResult(
                    output=f"Function '{func_name}' not found in repo interface.", 
                    is_error=True
                )
            
            if res and res.output:
                res.output = self._truncate_output(res.output)
            return res

        except pexpect.exceptions.EOF:
            return ExecutionResult(
                output="[SESSION_LOST] JDB reached EOF. The debuggee process likely exited.",
                is_error=True
            )
        except pexpect.exceptions.TIMEOUT:
            return ExecutionResult(
                output="[JDB_TIMEOUT] Timed out while waiting for JDB response.",
                is_error=True
            )
        except Exception as e:
            return ExecutionResult(output=f"[TOOL_ERROR] {str(e)}", is_error=True)


    def _handle_nomination(self, kwargs: dict) -> ExecutionResult:
        """Core logic: intercept nomination, invoke VerifyAgent."""
        from lib.verify_agent import JDBVerifyAgent

        method_signature = kwargs.get("method_signature")
        suspected_issue = kwargs.get("suspected_issue")
        known_static_facts = kwargs.get("known_static_facts", "No static facts provided.")
        target_lines = kwargs.get("target_lines", [])
        variables_of_interest = kwargs.get("variables_of_interest", [])
        verification_goal = kwargs.get("verification_goal", "Not specified")

        if self.ablation_mode == "no_dynamic":
            if self._debug:
                print("[Ablation] no_dynamic: skipping JDB verification entirely.")
            return ExecutionResult(
                output=json.dumps({
                    "Bug_Confirmed": None,
                    "Bug_Type": "NO_DYNAMIC_ABLATION",
                    "Explanation": (
                        "Runtime verification is NOT available in this configuration. "
                        "No JDB session was started and no runtime evidence was collected. "
                        "You MUST rely entirely on your static code analysis to make the final diagnosis. "
                        "Do NOT treat this as evidence against the nominated method — it simply means "
                        "no runtime check was performed."
                    ),
                    "Evidence": {},
                    "Metadata": {},
                    "Tokens": {"prompt": 0, "completion": 0},
                    "Dialog_History": []
                }),
                is_error=False
            )

        # ===== Ablation: remove static-to-dynamic context bridge =====
        if self.ablation_mode == "no_context_bridge":
            suspected_issue = (
                "No prior hypothesis is provided due to ablation. "
                "The verifier must infer suspicious behavior only from the test, stack trace, and method code."
            )
            known_static_facts = "No static facts provided due to ablation."
            target_lines = []
            variables_of_interest = []
            verification_goal = (
                "No explicit verification goal is provided due to ablation. "
                "Inspect the method's runtime behavior and determine whether it contributes to the failure."
            )

            if self._debug:
                print("[Ablation] no_context_bridge: removed hypothesis, static facts, target lines, variables, and verification goal.")

        begin_line = None
        method_code = "Source code not found."
        try:
            method_info = self._ri.get_method_info(method_signature)
            if method_info:
                begin_line = method_info.get("begin_line")
            
            code_result = self._ri.fname2func.get("get_code_snippet")(**{"signature": method_signature})
            if code_result:
                method_code = str(code_result)
        except Exception as e:
            if self._debug: print(f"--- [Backend Error] Failed to fetch method info: {e} ---")
            pass

        test_info = self._ri.failing_test_signatures[0] 

        if self._debug:
            print(f"--- [Backend] Initiating Sub-Agent ---")
            print(f"Target: {method_signature}")
            print(f"Lines: {target_lines} | Vars: {variables_of_interest}")
            print(f"--- [Backend] Initiating Sub-Agent with Static Facts ---")
            print(f"Facts: {known_static_facts[:100]}...")

        verifier = JDBVerifyAgent(jdb_backend=self, model=self.verifier_model)
        
        start_time = time.time()

        report_json_str = verifier.verify(
            method_id=method_signature,
            method_code=method_code,
            hypothesis=suspected_issue,
            test_info=test_info,      # (comment removed)
            known_static_facts=known_static_facts, # (comment removed)
            target_lines=target_lines,
            variables_of_interest=variables_of_interest,
            begin_line=begin_line,
            verification_goal = verification_goal
        )

        verify_time = time.time() - start_time

        try:
            report_data = json.loads(report_json_str)
            b_type = report_data.get("Bug_Type", report_data.get("bug_type", ""))
            
            meta = report_data.pop("Metadata", {})
            tokens = report_data.pop("Tokens", {})
            dialog_history = report_data.pop("Dialog_History", [])
            meta["verify_time_seconds"] = verify_time

            if not hasattr(self, "dynamic_stats"):
                self.dynamic_stats = []
            
            verification_result = {
                "bug_confirmed": report_data.get("Bug_Confirmed", False),
                "bug_type": b_type,
                "explanation": report_data.get("Explanation", ""),
                "evidence": report_data.get("Evidence", {}),
                "evidence_is_empty": (
                    not report_data.get("Evidence")
                    or (isinstance(report_data.get("Evidence"), dict) and not report_data.get("Evidence"))
                    or (isinstance(report_data.get("Evidence"), str) and not report_data.get("Evidence").strip())
                )
            }

            self.dynamic_stats.append({
                "method_signature": method_signature,
                "metadata": meta,
                "sub_agent_tokens": tokens,
                "sub_agent_messages": dialog_history,
                "verification_result": verification_result
            })

            is_confirmed = report_data.get("Bug_Confirmed", False)

            report_json_str = json.dumps(report_data, indent=2, ensure_ascii=False)

            if b_type == "UNREACHED_TARGET":
                formatted_report = (
                    "=== Sub-Agent Verification Report (UNREACHED) ===\n"
                    "DYNAMIC DEBUGGING FAILED: The target lines were never executed.\n"
                    "SYSTEM DIRECTIVE: The test case followed a different branch. "
                    "Do NOT nominate this method again for dynamic debugging. Use static analysis."
                )
            else:
                formatted_report = (
                    f"=== Sub-Agent Verification Report ===\n"
                    f"Target: {method_signature}\n"
                    f"Result:\n{report_json_str}\n"
                    f"====================================="
                )

                if is_confirmed:
                    formatted_report += (
                        "\nSystem Note: [EVIDENCE VALIDATION REQUIRED] The sub-agent claims to have found a bug. "
                        "Before accepting this, you MUST evaluate the 'Evidence' provided above. "
                        "Valid evidence can be either: "
                        "(1) an abnormal runtime value, such as exception, NaN, unexpected null, or out-of-bounds; OR "
                        "(2) a normal runtime value that triggers an incorrect branch/return according to the source code and failing test expectation. "
                        "If the evidence only shows normal values without any incorrect control-flow consequence, reject the finding and investigate related caller/callee methods. "
                        "Only reply 'READY' if the evidence proves an abnormal state or an incorrect runtime branch."
                    )
                elif b_type == "INCONCLUSIVE":
                    formatted_report += (
                        "System Note: The sub-agent EXHAUSTED its budget and failed to reach a conclusion. "
                        "WARNING: This does NOT mean the method is bug-free. "
                        "If your static analysis strongly suggests the bug is here, YOU MAY re-nominate this method using DIFFERENT 'target_lines' or 'verification_goal'."
                    )
                else: # NO_BUG
                    formatted_report += (
                        "System Note: [NEGATIVE] The sub-agent did NOT find runtime evidence. \n"
                        "CRITICAL WARNING FOR MATH/LOGIC BUGS: Many logic bugs are 'silent' (no crashes, just wrong values). The sub-agent might have missed it due to narrow verification goals. \n"
                        "ACTION: DO NOT discard this method if your static analysis strongly suggests it is the root cause. Treat it as a 'High Suspicion, Unverified' candidate."
                    )

        except:
            formatted_report = f"=== Sub-Agent Report ===\n{report_json_str}"

        return ExecutionResult(output=formatted_report, is_error=False)


    def _set_breakpoint(self, location: str) -> ExecutionResult:
        """ Auto-detect stop in (method) or stop at (line). """
        self._flush_buffer()
        clean_loc = location.split('(')[0]

        if ":" in clean_loc:
            cmd = f"stop at {clean_loc}"
        else:
            cmd = f"stop in {clean_loc}"
        
        self.session.sendline(cmd)
        self.session.expect(self.prompt)
        output = self.session.before
        
        if self._debug: print(f"[DEBUG_SET_BP] Cmd: {cmd} | Output: {output.strip()}")

        success_keywords = ["Set breakpoint", "Deferring", "Deferring", "Setup", "Deferring"]
        if any(k in output for k in success_keywords):
             return ExecutionResult(output=f"Breakpoint set: {location}", is_error=False)
        
        return ExecutionResult(output=f"Failed to set breakpoint: {output.strip()}", is_error=True)


    def _run_to_breakpoint(self) -> ExecutionResult:
        self._flush_buffer()
        self.session.sendline("run")
        start_time = time.time()
        
        while(time.time() - start_time < 30): # (comment removed)
            try:
                index = self.session.expect(
                    [self.prompt, "VM already running", "VM already running"],
                    timeout=15
                )
            except pexpect.exceptions.EOF:
                return ExecutionResult(
                    output="[SESSION_LOST] JDB reached EOF while running to breakpoint. The program likely exited.",
                    is_error=True
                )
            except pexpect.exceptions.TIMEOUT:
                return ExecutionResult(
                    output="[JDB_TIMEOUT] Timeout while running to breakpoint.",
                    is_error=True
                )
            
            output = self.session.before
            matched_prompt = self.session.match.group(1).strip() if self.session.match and index == 0 else ""
            
            if index > 0:
                self.session.sendline("cont")
                continue

            error_msg = self._check_for_exception(output)
            if error_msg:
                return ExecutionResult(output=error_msg, is_error=True)

            if any(k in output for k in ["Breakpoint hit", "Breakpoint hit"]) or "[" in matched_prompt:
                self.session.sendline("print \"---SYNC---\"")
                self.session.expect("---SYNC---")
                self.session.expect(self.prompt)
                
                self.session.sendline("list")
                self.session.expect(self.prompt)
                return ExecutionResult(output=f"Hit!\n{self.session.before}", is_error=False)
            
            if any(k in output for k in ["exited", "exited", "Disconnected", "Disconnected"]):
                # return ExecutionResult(output="[PROGRAM_EXITED] The program has finished execution and exited.", is_error=False)
                return ExecutionResult(
                    output="[TERMINAL_STATE] The program exited without hitting any breakpoints. "
                        "The code at your target lines was NOT executed in this test run. "
                        "Inspection is IMPOSSIBLE. You must stop trying and report UNREACHED.", 
                    is_error=False
                )

            if matched_prompt == ">":
                continue
        
        return ExecutionResult(output="[JDB_TIMEOUT] Run timeout while waiting for breakpoint.",is_error=True)
    
    def _get_locals(self) -> ExecutionResult:
        """ Parse local variables as dict string. """
        self._flush_buffer()

        self.session.sendline("locals")
        self.session.expect(self.prompt)
        raw_locals = self.session.before
        
        # self.session.sendline("dump this")
        # self.session.expect(self.prompt)
        # raw_fields = self.session.before

        locals_list = []
        fields_list = []
        allowed = set()
        pat = re.compile(r'^\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(.+?)\s*$', re.M)

        for m in pat.finditer(raw_locals):
            name = m.group(1)
            value = m.group(2)

            type_match = re.search(r'instance of ([\w\.$]+)', value)
            var_type = type_match.group(1) if type_match else "unknown"

            locals_list.append({
            "name": name,
            "type": var_type,
            "value": value
            })
            allowed.add(name)

        # field_pat = re.compile(r'^\s*([A-Za-z_$][A-Za-z0-9_$]*):\s*(.+?)\s*$', re.M)
        # for m in field_pat.finditer(raw_fields):
        #     name, value = m.group(1), m.group(2)
        #     full_name = f"this.{name}"
            
        #     type_match = re.search(r'instance of ([\w\.$]+)', value)
        #     var_type = type_match.group(1) if type_match else "unknown"
            
        #     fields_list.append({"name": full_name, "type": var_type, "value": value})
        #     allowed.add(full_name)
        #     allowed.add(name) 

        if self.ablation_mode not in ["no_fusion", "no_runtime_fusion"]:
            self.session.sendline("dump this")
            self.session.expect(self.prompt)
            raw_fields = self.session.before

            field_pat = re.compile(r'^\s*([A-Za-z_$][A-Za-z0-9_$]*):\s*(.+?)\s*$', re.M)
            for m in field_pat.finditer(raw_fields):
                name, value = m.group(1), m.group(2)
                full_name = f"this.{name}"
                
                type_match = re.search(r'instance of ([\w\.$]+)', value)
                var_type = type_match.group(1) if type_match else "unknown"
                
                fields_list.append({"name": full_name, "type": var_type, "value": value})
                allowed.add(full_name)
                allowed.add(name) 
        else:
            if self._debug: print("[Ablation] Skipping 'dump this' (no_fusion mode).")

        self._allowed_locals = allowed
        
        combined_result = {
            "locals": locals_list,
            "fields": fields_list,
            "note": "If a variable is in 'fields', you can access it via 'this.variableName'"
        }

        return ExecutionResult(
            output=json.dumps(combined_result, ensure_ascii=False),
            is_error=False
        )

    def _step_next(self) -> ExecutionResult:
        self._flush_buffer() # (comment removed)
        self._allowed_locals = set()
        self.session.sendline("next")
        self.session.expect(self.prompt)
        time.sleep(0.2) # (comment removed)
        
        next_command_output = self.session.before
        
        error_msg = self._check_for_exception(next_command_output)
        if error_msg:
            return ExecutionResult(output=error_msg, is_error=True)

        self._flush_buffer()

        time.sleep(0.2)
        self._flush_buffer()
        
        self.session.sendline("list")
        self.session.expect(self.prompt)
        list_output = self.session.before
        
        line_info = "Unknown"
        match = re.search(r'(\d+,\d+|\d+)\s+=>', list_output)
        if match:
            line_info = match.group(1).replace(',', '') # (comment removed)
        
            
        return ExecutionResult(output=f"Location: {line_info}\nCode:\n{list_output.strip()}", is_error=False)
    
    def _print_variable(self, variable: str) -> ExecutionResult:
        """ Send print variable command (supports this. and arrays). """
        
        if not variable or not re.match(r'^[A-Za-z_$][A-Za-z0-9_$.\[\]]*$', variable):
            return ExecutionResult(
                output=json.dumps({"status": "ERROR", "message": f"Invalid format: '{variable}'"}, ensure_ascii=False),
                is_error=True
            )

        allowed = getattr(self, "_allowed_locals", None)
        if allowed is not None:
            alternatives = [variable, f"this.{variable}", variable.replace("this.", "")]
            if not any(alt in allowed for alt in alternatives):
                pass 

        self._flush_buffer()
        self.session.sendline(f"print {variable}")
        self.session.expect(self.prompt)
        raw_output = self.session.before
        
        if "Name unknown" in raw_output:
             return ExecutionResult(
                output=json.dumps({
                    "status": "NOT_IN_SCOPE",
                    "message": f"JDB: Name '{variable}' unknown in current stack frame."
                }, ensure_ascii=False),
                is_error=True
            )

        m = re.search(r'=\s*(.*)$', raw_output, re.M)
        if m:
            return ExecutionResult(
                output=json.dumps({
                    "status": "OK",
                    "name": variable,
                    "value": m.group(1).strip()
                }, ensure_ascii=False),
                is_error=False
            )

        return ExecutionResult(
            output=json.dumps({"status": "PARSE_FAIL", "content": raw_output}, ensure_ascii=False),
            is_error=True
        )

    def _list_code(self) -> ExecutionResult:
        """[cleaned]"""
        self._flush_buffer()
        self.session.sendline("list")
        self.session.expect(self.prompt)
        return ExecutionResult(output=self.session.before.strip(), is_error=False)

    def _get_stack_trace(self) -> ExecutionResult:
        """[cleaned] where [cleaned]"""
        self._flush_buffer()
        self.session.sendline("where")
        self.session.expect(self.prompt)
        stack_info = self.session.before
        if self._debug: print(f"[DEBUG_STACK] {stack_info}")
        return ExecutionResult(output=f"Current Stack:\n{stack_info.strip()}", is_error=False)

    def _track_local_variable(self, variable: str) -> ExecutionResult:
        """[cleaned]：[cleaned]20[cleaned]，[cleaned]"""
        self._flush_buffer()
        if not variable:
            return ExecutionResult(output="Error: No variable specified.", is_error=True)

        trajectory = []
        max_steps = 20 # (comment removed)
        
        for i in range(max_steps):
            self.session.sendline("next")
            self.session.expect(self.prompt)
            next_out = self.session.before
            
            error_msg = self._check_for_exception(next_out)
            if error_msg:
                trajectory.append(f"[Exception Occurred] {error_msg}")
                break
            if any(k in next_out for k in ["exited", "exited", "Disconnected", "Disconnected"]):
                trajectory.append("[Method Exited]")
                break
                
            self.session.sendline("list")
            self.session.expect(self.prompt)
            list_out = self.session.before
            match = re.search(r'(\d+,\d+|\d+)\s+=>', list_out)
            line_info = match.group(1).replace(',', '') if match else "Unknown Line"

            self.session.sendline(f"print {variable}")
            self.session.expect(self.prompt)
            print_out = self.session.before
            
            if "Name unknown" in print_out:
                trajectory.append(f"Line {line_info}: Name '{variable}' went out of scope.")
                break
                
            m = re.search(r'=\s*(.*)$', print_out, re.M)
            val = m.group(1).strip() if m else "Parse Failed"
            
            trajectory.append(f"Line {line_info}: {variable} = {val}")

        compressed_trajectory = []
        last_val = None
        for step in trajectory:
            if " = " in step:
                current_val = step.split(" = ")[1]
                if current_val == last_val:
                    continue # (comment removed)
                last_val = current_val
            compressed_trajectory.append(step)

        output_str = f"Trajectory of '{variable}' (filtered unchanged steps):\n" + "\n".join(compressed_trajectory)
        if len(trajectory) == max_steps:
            output_str += "\n... [Max tracking steps reached. Call again to continue tracking]"
            
        return ExecutionResult(output=output_str, is_error=False)

    def _check_for_exception(self, output: str) -> str:
        """[cleaned] JDB [cleaned]"""
        if not output:
            return ""
        if "Exception occurred" in output or "thrown" in output:
            return f"CRITICAL: Exception detected during execution!\n{output.strip()}"
        return ""

    def _flush_buffer(self):
        """[cleaned]，[cleaned]"""
        try:
            while True:
                self.session.expect(r'.+', timeout=0.1) # (comment removed)
        except pexpect.exceptions.TIMEOUT:
            pass
        except pexpect.exceptions.EOF:
            pass

    def _truncate_output(self, text: str, max_chars: int = 8000) -> str:
        """
        [cleaned]：[cleaned]（[cleaned]）[cleaned]。
        [cleaned]，[cleaned]，[cleaned]，[cleaned]。
        """
        if not text or len(text) <= max_chars:
            return text  # (comment removed)
        
        head_len = 3000
        tail_len = 3000
        
        head_text = text[:head_len]
        tail_text = text[-tail_len:]
        omitted_length = len(text) - head_len - tail_len
        
        warning_msg = f"\n\n... [SYSTEM WARNING: Output was too large. {omitted_length} characters omitted. Head and Tail preserved.] ...\n\n"
        
        return head_text + warning_msg + tail_text
   



    