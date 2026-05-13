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

"""Android action tools wrapped for AgentScope integration.

Provides utility functions to construct Android action messages compatible
with AgentScope's model interface, and to execute parsed actions on the
Android environment.
"""

import base64
import io
import json
from typing import Any, Optional

import numpy as np
from PIL import Image

from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils


def image_to_base64_url(image: np.ndarray) -> str:
  """Convert a numpy image array to a base64 data URL string.

  Args:
    image: Image as numpy ndarray (H, W, C) in RGB.

  Returns:
    Base64-encoded JPEG data URL (e.g., "data:image/jpeg;base64,...").
  """
  pil_image = Image.fromarray(image)
  buf = io.BytesIO()
  pil_image.save(buf, format="JPEG")
  buf.seek(0)
  img_bytes = buf.read()
  b64_str = base64.b64encode(img_bytes).decode("utf-8")
  return f"data:image/jpeg;base64,{b64_str}"


def build_multimodal_messages(
    system_prompt: str,
    user_text: str,
    images: list[np.ndarray],
    backend: str = "openai",
) -> list[dict[str, Any]]:
  """Build AgentScope-compatible multimodal messages.

  Constructs a list of message dicts in the correct format for the target
  model backend.

  Args:
    system_prompt: System-level instruction for the agent.
    user_text: The text content of the user message.
    images: List of images as numpy ndarrays to include in the message.
    backend: Model backend ("openai", "dashscope", "gemini"). Determines
      the image content format.

  Returns:
    List of message dicts with roles and multimodal content.
  """
  messages: list[dict[str, Any]] = []

  if system_prompt:
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

  content: list[dict[str, Any]] = [
      {"type": "text", "text": user_text},
  ]

  for image in images:
    b64_url = image_to_base64_url(image)
    if backend == "dashscope":
      # DashScope uses {"type": "image", "image": "..."}
      content.append({
          "type": "image",
          "image": b64_url,
      })
    else:
      # OpenAI / Gemini use {"type": "image_url", "image_url": {"url": "..."}}
      content.append({
          "type": "image_url",
          "image_url": {
              "url": b64_url,
          },
      })

  messages.append({
      "role": "user",
      "content": content,
  })

  return messages


def build_text_messages(
    system_prompt: str,
    user_text: str,
) -> list[dict[str, Any]]:
  """Build AgentScope-compatible text-only messages.

  Args:
    system_prompt: System-level instruction for the agent.
    user_text: The text content of the user message.

  Returns:
    List of message dicts with roles and text content.
  """
  messages: list[dict[str, Any]] = []

  if system_prompt:
    messages.append({
        "role": "system",
        "content": system_prompt,
    })

  messages.append({
      "role": "user",
      "content": user_text,
  })

  return messages


def execute_parsed_action(
    action_dict: dict[str, Any],
    env: interface.AsyncEnv,
) -> Optional[json_action.JSONAction]:
  """Convert a parsed action dict to JSONAction and execute on environment.

  Args:
    action_dict: Dictionary containing action_type and parameters.
    env: The Android environment to execute on.

  Returns:
    The JSONAction that was executed, or None if parsing failed.
  """
  try:
    converted = json_action.JSONAction(**action_dict)
  except (TypeError, ValueError) as e:
    print(f"Failed to create JSONAction from {action_dict}: {e}")
    return None

  try:
    env.execute_action(converted)
  except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"Failed to execute action {converted}: {e}")
    return None

  return converted


def get_available_actions_schema() -> str:
  """Return the list of available Android actions and their JSON formats.

  This mirrors the action descriptions used in M3A/T3A prompts but is
  centralized here for reuse across different AgentScope-based agents.

  Returns:
    Multi-line string describing available actions.
  """
  return """- Click/tap on an element: `{"action_type": "click", "index": <target_index>}`
- Long press on an element: `{"action_type": "long_press", "index": <target_index>}`
- Type text into a text field: `{"action_type": "input_text", "text": <text_input>, "index": <target_index>}`
- Press the Enter key: `{"action_type": "keyboard_enter"}`
- Navigate to the home screen: `{"action_type": "navigate_home"}`
- Navigate back: `{"action_type": "navigate_back"}`
- Scroll the screen: `{"action_type": "scroll", "direction": <up, down, left, right>, "index": <optional_target_index>}`
- Open an app: `{"action_type": "open_app", "app_name": <name>}`
- Wait for the screen to update: `{"action_type": "wait"}`
- Finish the task: `{"action_type": "status", "goal_status": "complete"}`
- Mark task as infeasible: `{"action_type": "status", "goal_status": "infeasible"}`
- Answer user's question: `{"action_type": "answer", "text": "<answer_text>"}`"""


def _describe_ui_element(
    ui_element: representation_utils.UIElement, index: int
) -> str:
  """Generate a description for a given UI element with important information.

  Args:
    ui_element: UI element to describe.
    index: The numeric index for the UI element.

  Returns:
    JSON-like description string for the UI element.
  """
  element_description = f'UI element {index}: {{"index": {index}, '
  if ui_element.text:
    element_description += f'"text": "{ui_element.text}", '
  if ui_element.content_description:
    element_description += (
        f'"content_description": "{ui_element.content_description}", '
    )
  if ui_element.hint_text:
    element_description += f'"hint_text": "{ui_element.hint_text}", '
  if ui_element.tooltip:
    element_description += f'"tooltip": "{ui_element.tooltip}", '
  element_description += (
      f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
  )
  element_description += (
      f'"is_long_clickable": {"True" if ui_element.is_long_clickable else "False"}, '
  )
  element_description += (
      f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
  )
  if ui_element.is_scrollable:
    element_description += '"is_scrollable": True, '
  if ui_element.is_focusable:
    element_description += '"is_focusable": True, '
  element_description += (
      f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
  )
  element_description += (
      f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
  )
  return element_description[:-2] + '}'


def get_ui_elements_description(
    ui_elements: list[representation_utils.UIElement],
    screen_size: tuple[int, int],
) -> str:
  """Generate a text description of all UI elements on screen.

  This is used by ReActAgent tools to describe the current screen state
  to the language model.

  Args:
    ui_elements: List of UI elements on the current screen.
    screen_size: Screen width and height in pixels.

  Returns:
    Multi-line string with one line per visible UI element.
  """
  description = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_size):
      description += _describe_ui_element(ui_element, index) + '\n'
  return description if description else '(No UI elements visible)'
