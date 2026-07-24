from pydantic import BaseModel
from typing import Optional

class ExecutionResult(BaseModel):
    """[cleaned]，[cleaned] deepagents [cleaned]"""
    output: str         # (comment removed)
    exit_code:int = 0   # (comment removed)
    is_error: bool = False  # (comment removed)
    truncated: bool = False  # (comment removed)