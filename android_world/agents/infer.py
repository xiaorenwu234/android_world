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

"""Some LLM inference interface."""

import abc
import asyncio
import base64
import concurrent.futures
import io
import os
import time
from typing import Any, Optional

import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests

from android_world.agents import agentscope_config


ERROR_CALLING_LLM = 'Error calling LLM'


def _sync_model_call(model, messages):
  """Run an AgentScope model call synchronously with fresh coroutine."""
  async def _call():
    return await model(messages=messages)

  try:
    return asyncio.run(_call())
  except RuntimeError:
    def _run_in_thread():
      return asyncio.run(_call())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(_run_in_thread)
      return future.result()


def _call_agentscope_openai(
    text_prompt: str,
    images: list[np.ndarray],
    model_name: str,
    temperature: float = 0.0,
) -> tuple[str, Optional[bool], Any]:
  """Call OpenAI through AgentScope."""
  from android_world.agents import agentscope_tools  # pylint: disable=import-outside-toplevel
  model = agentscope_config.get_model(
      model_backend="openai",
      model_name=model_name,
      temperature=temperature,
  )
  messages = agentscope_tools.build_multimodal_messages(
      system_prompt="",
      user_text=text_prompt,
      images=images,
      backend="openai",
  )
  try:
    response = _sync_model_call(model, messages)
    if response.content:
      text = response.content[0].get("text", "")
      return text, None, response
    return ERROR_CALLING_LLM, None, None
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Error calling AgentScope OpenAI: {e}")
    return ERROR_CALLING_LLM, None, None


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
  """Converts a numpy array into a byte string for a JPEG image."""
  image = Image.fromarray(image)
  return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
  in_mem_file = io.BytesIO()
  image.save(in_mem_file, format='JPEG')
  # Reset file pointer to start
  in_mem_file.seek(0)
  img_bytes = in_mem_file.read()
  return img_bytes


class LlmWrapper(abc.ABC):
  """Abstract interface for (text only) LLM."""

  @abc.abstractmethod
  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    """Calling text-only LLM with a prompt.

    Args:
      text_prompt: Text prompt.

    Returns:
      Text output, is_safe, and raw output.
    """


class MultimodalLlmWrapper(abc.ABC):
  """Abstract interface for Multimodal LLM."""

  @abc.abstractmethod
  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Calling multimodal LLM with a prompt and a list of images.

    Args:
      text_prompt: Text prompt.
      images: List of images as numpy ndarray.

    Returns:
      Text output and raw output.
    """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Gemini GCP interface."""

  def __init__(
      self,
      model_name: str | None = None,
      max_retry: int = 3,
      temperature: float = 0.0,
      top_p: float = 0.95,
      enable_safety_checks: bool = True,
  ):
    if 'GCP_API_KEY' not in os.environ:
      raise RuntimeError('GCP API key not set.')
    genai.configure(api_key=os.environ['GCP_API_KEY'])
    self.llm = genai.GenerativeModel(
        model_name,
        safety_settings=None
        if enable_safety_checks
        else SAFETY_SETTINGS_BLOCK_NONE,
        generation_config=generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p
        ),
    )
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)

  def predict(
      self,
      text_prompt: str,
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(
        text_prompt, [], enable_safety_checks, generation_config
    )

  def is_safe(self, raw_response):
    try:
      return (
          raw_response.candidates[0].finish_reason
          != answer_types.FinishReason.SAFETY
      )
    except Exception:  # pylint: disable=broad-exception-caught
      #  Assume safe if the response is None or doesn't have candidates.
      return True

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      enable_safety_checks: bool = True,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Optional[bool], Any]:
    """Call multimodal LLM via AgentScope GeminiChatModel."""
    try:
      model = agentscope_config.get_model(
          model_backend="gemini",
          model_name=self.llm.model_name,
      )
      from android_world.agents import agentscope_tools  # pylint: disable=import-outside-toplevel
      messages = agentscope_tools.build_multimodal_messages(
          system_prompt="",
          user_text=text_prompt,
          images=images,
          backend="gemini",
      )
      response = _sync_model_call(model, messages)
      if response.content:
        text = response.content[0].get("text", "")
        return text, True, response
      return ERROR_CALLING_LLM, None, None
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Error calling AgentScope Gemini: {e}")
      return ERROR_CALLING_LLM, None, None

  def generate(
      self,
      contents: (
          content_types.ContentsType | list[str | np.ndarray | Image.Image]
      ),
      safety_settings: safety_types.SafetySettingOptions | None = None,
      generation_config: generation_types.GenerationConfigType | None = None,
  ) -> tuple[str, Any]:
    """Exposes the generate_content API.

    Args:
      contents: The input to the LLM.
      safety_settings: Safety settings.
      generation_config: Generation config.

    Returns:
      The output text and the raw response.
    Raises:
      RuntimeError:
    """
    counter = self.max_retry
    retry_delay = 1.0
    response = None
    if isinstance(contents, list):
      contents = self.convert_content(contents)
    while counter > 0:
      try:
        response = self.llm.generate_content(
            contents=contents,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        return response.text, response
      except Exception as e:  # pylint: disable=broad-exception-caught
        counter -= 1
        print('Error calling LLM, will retry in {retry_delay} seconds')
        print(e)
        if counter > 0:
          # Expo backoff
          time.sleep(retry_delay)
          retry_delay *= 2
    raise RuntimeError(f'Error calling LLM. {response}.')

  def convert_content(
      self,
      contents: list[str | np.ndarray | Image.Image],
  ) -> content_types.ContentsType:
    """Converts a list of contents to a ContentsType."""
    converted = []
    for item in contents:
      if isinstance(item, str):
        converted.append(item)
      elif isinstance(item, np.ndarray):
        converted.append(Image.fromarray(item))
      elif isinstance(item, Image.Image):
        converted.append(item)
    return converted


class DashScopeWrapper(MultimodalLlmWrapper):
  """DashScope multimodal wrapper via AgentScope.

  Uses AgentScope DashScopeChatModel as the underlying model caller.
  Implements the same MultimodalLlmWrapper interface as Gpt4Wrapper so it
  can be used as a drop-in replacement for the original M3A step loop.
  """

  def __init__(
      self,
      model_name: str = 'qwen-vl-plus',
      temperature: float = 0.0,
  ):
    self.model_name = model_name
    self.temperature = temperature

  def predict_mm(
      self,
      text_prompt: str,
      images: list[np.ndarray],
  ) -> tuple[str, Optional[bool], Any]:
    """Call multimodal LLM via AgentScope DashScopeChatModel."""
    from android_world.agents import agentscope_tools  # pylint: disable=import-outside-toplevel
    model = agentscope_config.get_model(
        model_backend='dashscope',
        model_name=self.model_name,
        temperature=self.temperature,
    )
    messages = agentscope_tools.build_multimodal_messages(
        system_prompt='',
        user_text=text_prompt,
        images=images,
        backend='dashscope',
    )
    try:
      response = _sync_model_call(model, messages)
      if response.content:
        text = response.content[0].get('text', '')
        return text, None, response
      return ERROR_CALLING_LLM, None, None
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f'Error calling AgentScope DashScope: {e}')
      return ERROR_CALLING_LLM, None, None


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
  """OpenAI GPT4 wrapper.

  Attributes:
    openai_api_key: The class gets the OpenAI api key either explicitly, or
      through env variable in which case just leave this empty.
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """

  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if 'OPENAI_API_KEY' not in os.environ:
      raise RuntimeError('OpenAI API key not set.')
    self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    """Call multimodal LLM via AgentScope OpenAIChatModel."""
    return _call_agentscope_openai(
        text_prompt=text_prompt,
        images=images,
        model_name=self.model,
        temperature=self.temperature,
    )
