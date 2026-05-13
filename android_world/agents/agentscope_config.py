# Copyright 2026 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgentScope configuration and model factory for Android World agents.

Provides unified model creation supporting OpenAI, DashScope, and Gemini
backends through AgentScope's model wrappers.
"""

import asyncio
import os
from typing import Optional

from agentscope.model import (
    DashScopeChatModel,
    GeminiChatModel,
    OpenAIChatModel,
)


# Default model names per backend.
_DEFAULT_MODELS = {
    "openai": "gpt-4-turbo-2024-04-09",
    "dashscope": "qwen-vl-plus",
    "gemini": "gemini-1.5-pro-latest",
}


def get_model(
    model_backend: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    stream: bool = False,
):
  """Factory function to create AgentScope model instances.

  Args:
    model_backend: One of "openai", "dashscope", "gemini".
    model_name: Specific model name. Falls back to sensible defaults.
    temperature: Generation temperature (passed via generate_kwargs).
    api_key: API key for the backend. Falls back to env variables.
    stream: Whether to enable streaming mode.

  Returns:
    An AgentScope model instance (OpenAIChatModel, DashScopeChatModel, or
    GeminiChatModel).

  Raises:
    ValueError: If the model_backend is unknown or required API key is missing.
  """
  if model_backend == "openai":
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
      raise ValueError(
          "OPENAI_API_KEY environment variable not set. "
          "Set it or pass api_key explicitly."
      )
    model_name = model_name or _DEFAULT_MODELS["openai"]
    return OpenAIChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=stream,
        generate_kwargs={"temperature": temperature},
    )

  elif model_backend == "dashscope":
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
      raise ValueError(
          "DASHSCOPE_API_KEY environment variable not set. "
          "Set it or pass api_key explicitly."
      )
    model_name = model_name or _DEFAULT_MODELS["dashscope"]
    return DashScopeChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=stream,
        generate_kwargs={"temperature": temperature},
    )

  elif model_backend == "gemini":
    api_key = api_key or os.environ.get("GCP_API_KEY")
    if not api_key:
      raise ValueError(
          "GCP_API_KEY environment variable not set. "
          "Set it or pass api_key explicitly."
      )
    model_name = model_name or _DEFAULT_MODELS["gemini"]
    return GeminiChatModel(
        model_name=model_name,
        api_key=api_key,
        stream=stream,
    )

  else:
    raise ValueError(
        f"Unknown model_backend: {model_backend}. "
        f"Choose from: {list(_DEFAULT_MODELS.keys())}"
    )


def sync_model_call(model, messages):
  """Run an AgentScope model call synchronously.

  Creates a fresh coroutine each attempt to avoid "cannot reuse already
  awaited coroutine" errors. Handles both cases: when no event loop exists
  and when one is already running.

  Args:
    model: An AgentScope model instance.
    messages: List of message dicts to send to the model.

  Returns:
    The ChatResponse from the model.
  """
  # Each call creates a fresh coroutine to avoid reuse issues.
  async def _call():
    return await model(messages=messages)

  try:
    return asyncio.run(_call())
  except RuntimeError:
    # Event loop is already running. Create a new coroutine in a new
    # event loop on a separate thread.
    import concurrent.futures

    def _run_in_thread():
      return asyncio.run(_call())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(_run_in_thread)
      return future.result()


def sync_agent_call(agent, msg):
  """Run an AgentScope async agent call synchronously.

  AgentScope ReActAgent is fully async. This function bridges async agent
  execution into synchronous code, handling both cases: no event loop and
  already-running event loop.

  Args:
    agent: An AgentScope ReActAgent instance.
    msg: The message (str or Msg) to send to the agent.

  Returns:
    The Msg response from the agent.
  """
  async def _call():
    return await agent(msg)

  try:
    return asyncio.run(_call())
  except RuntimeError:
    # Event loop is already running. Use a separate thread.
    import concurrent.futures

    def _run_in_thread():
      return asyncio.run(_call())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(_run_in_thread)
      return future.result()


def sync_get_memory(memory):
  """Get messages from an AgentScope async memory synchronously.

  Args:
    memory: An AgentScope InMemoryMemory instance.

  Returns:
    List of Msg objects.
  """
  async def _call():
    return await memory.get_memory()

  try:
    return asyncio.run(_call())
  except RuntimeError:
    import concurrent.futures

    def _run_in_thread():
      return asyncio.run(_call())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(_run_in_thread)
      return future.result()
