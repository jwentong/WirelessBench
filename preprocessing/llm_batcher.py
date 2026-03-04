# -*- coding: utf-8 -*-
"""
Concurrent LLM API Batcher
===========================

Thread-pool based concurrent LLM API caller supporting OpenAI-compatible endpoints.
Used for dataset expansion and LLM-based dataset validation.

Usage:
    from preprocessing.llm_batcher import Batcher

    batcher = Batcher(
        api_name="openai",
        api_key="sk-...",
        model_name="gpt-4o",
        temperature=0.7,
        system_prompt="You are a wireless communication expert.",
    )
    results = batcher.handle_message_list(["Question 1", "Question 2", ...])
"""

from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


class Batcher:
    """
    Concurrent LLM API caller with retry logic and index tracking.

    Supports OpenAI-compatible APIs (OpenAI, vLLM, DashScope, etc.).
    Missing responses are tracked via ``get_miss_index()``.
    """

    def __init__(
        self,
        api_name: str = "openai",
        api_key: str = "",
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        system_prompt: str = "",
        max_token: int = 4096,
        num_workers: int = 16,
        timeout_duration: int = 120,
        retry_attempts: int = 2,
        api_base_url: Optional[str] = None,
    ):
        """
        Args:
            api_name: API type - currently "openai" or "vllm" (both use OpenAI client)
            api_key: API key
            model_name: Model identifier
            temperature: Sampling temperature
            system_prompt: System prompt prepended to every request
            max_token: Maximum completion tokens
            num_workers: Concurrent thread count
            timeout_duration: Timeout per chunk (seconds)
            retry_attempts: Retries for timed-out requests
            api_base_url: Override base URL (e.g., for vLLM or DashScope)
        """
        self.client = OpenAI(api_key=api_key)
        if api_base_url:
            self.client.base_url = api_base_url

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_token = max_token
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index: List[int] = []

    # -- internal ---------------------------------------------------------

    def _call_llm(self, indexed_message: Tuple[int, str]) -> Tuple[int, Optional[str]]:
        """Call LLM for a single (index, prompt) pair."""
        index, prompt = indexed_message
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_token,
            )
            return (index, completion.choices[0].message.content)
        except Exception as e:
            print(f"[Batcher] Error at index {index}: {e}")
            self.miss_index.append(index)
            return (index, None)

    @staticmethod
    def _chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _process_messages(self, indexed_list: List[Tuple[int, str]]) -> List[Tuple[int, Optional[str]]]:
        """Process all messages with thread pool."""
        results: List[Tuple[int, Optional[str]]] = []
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        chunks = list(self._chunk_list(indexed_list, self.num_workers))

        try:
            for chunk in tqdm(chunks, desc="Processing batches"):
                future_map = {
                    executor.submit(self._call_llm, msg): msg for msg in chunk
                }
                for _ in range(self.retry_attempts):
                    done, not_done = wait(future_map.keys(), timeout=self.timeout_duration)
                    for future in not_done:
                        future.cancel()
                    results.extend(f.result() for f in done if f.done())
                    if not not_done:
                        break
                    future_map = {
                        executor.submit(self._call_llm, future_map[f]): future_map[f]
                        for f in not_done
                    }
        except Exception as e:
            print(f"[Batcher] Batch error: {e}")
        finally:
            executor.shutdown(wait=False)

        return results

    def _fill_missing(self, results: List[Tuple[int, Optional[str]]], total: int) -> List[Optional[str]]:
        """Ensure output list has exactly ``total`` entries aligned by index."""
        result_dict = {idx: val for idx, val in results}
        output = []
        for i in range(total):
            if i in result_dict:
                output.append(result_dict[i])
            else:
                self.miss_index.append(i)
                output.append(None)
        return output

    # -- public API -------------------------------------------------------

    def handle_message_list(self, messages: List[str]) -> List[Optional[str]]:
        """
        Send a list of prompts to the LLM concurrently and return aligned results.

        Args:
            messages: List of user prompts

        Returns:
            List of responses (same length as input; ``None`` for failures)
        """
        indexed = [(i, msg) for i, msg in enumerate(messages)]
        raw_results = self._process_messages(indexed)
        raw_results.sort(key=lambda x: x[0])
        return self._fill_missing(raw_results, len(messages))

    def get_miss_index(self) -> List[int]:
        """Return indices of messages that failed after all retries."""
        return sorted(set(self.miss_index))
