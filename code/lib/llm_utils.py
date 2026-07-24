import os
import time
import random
from openai import OpenAI
from dotenv import load_dotenv
from abc import ABC

class OpenAIEngine(ABC):
    def __init__(self):
        load_dotenv()
        print(f"Base URL is: {os.environ.get('OPENAI_API_BASE_URL')}")
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE_URL"),
        )
        self._api_call_count = 0  # (comment removed)

    # Load environment variables from .env file
    def get_LLM_response(self, model, messages, **kwargs):
        """[cleaned] OpenAI API v1.0 [cleaned]，[cleaned]"""

        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.0

        max_retries = 8          # (comment removed)
        base_wait = 2            # (comment removed)
        max_wait = 120           # (comment removed)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                self._api_call_count += 1  # (comment removed)
                return response.model_dump()
            except Exception as e:
                error_str = str(e)

                if '429' in error_str or 'rate_limit' in error_str.lower():
                    tag = 'RATE_LIMIT'
                elif 'Connection' in error_str or 'connect' in error_str.lower():
                    tag = 'CONNECTION'
                elif '500' in error_str or '502' in error_str or '503' in error_str:
                    tag = 'SERVER_ERROR'
                elif 'timeout' in error_str.lower():
                    tag = 'TIMEOUT'
                else:
                    tag = 'UNKNOWN'

                base = min(base_wait ** attempt, max_wait)
                wait = base + random.uniform(0, base * 0.5)
                print(f"LLM [{tag}] attempt {attempt+1}/{max_retries}, waiting {wait:.1f}s: {error_str[:100]}")
                time.sleep(wait)

        raise Exception(f"Failed to get response from LLM after {max_retries} attempts (total wait ~{max_wait * max_retries}s).")