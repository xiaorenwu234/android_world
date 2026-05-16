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

"""Three-stage Pipeline M3A Agent for Android.

Pipeline:
  Stage 1 (Perception):  VLM analyzes screenshot  → hierarchical screen DSL
  Stage 2 (Reasoning):   LLM reads DSL + goal + full history → structured action intent
  Stage 3 (Translation): LLM converts intent       → executable JSON action
"""

import re
import time
from typing import Optional

from absl import logging
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils


# ── Stage 1: Perception Prompts ───────────────────────────────────────────────

PERCEPTION_PROMPT_TEMPLATE = (
    'You are an Android screen UI structure analyzer.\n'
    'Examine the two screenshots (first: raw screenshot; second: same screenshot'
    ' with bounding boxes and numeric indexes on UI elements) together with the'
    ' UI element list below.\n\n'
    'UI Elements:\n{ui_elements}\n\n'
    'Produce TWO sections in the EXACT format below (both sections are'
    ' REQUIRED, in this order, with the headers spelled exactly as shown):\n\n'
    '=== SCREEN_SUMMARY ===\n'
    '<A concise natural-language description of the current screen, 3–6'
    ' sentences. Cover ALL of the following when applicable:\n'
    '  * Which app / page this is (e.g. "Settings → Network & internet",'
    ' "Markor file list", "Home screen with app drawer open").\n'
    '  * The primary purpose of this page and what the user can DO here'
    ' (e.g. browse a list, edit a note, configure Wi‑Fi, pick a date).\n'
    '  * Major functional regions / features visible on the page (search'
    ' bar, tab switcher, action buttons, content list, FAB, dialog, etc.).\n'
    '  * Notable state information (current selection, toggle on/off,'
    ' empty state, error messages, modal/dialog open, keyboard visible,'
    ' loading spinner, etc.).\n'
    '  * Anything visually unusual or potentially blocking (permission'
    ' dialog, login required, offline banner, etc.).>\n\n'
    '=== SCREEN_DSL ===\n'
    'Screen {{\n'
    '  <Region> {{\n'
    '    <ElementType>("<label>", index=<n>, pos="<pos>", <attrs>)\n'
    '  }}\n'
    '}}\n\n'
    'DSL Rules:\n'
    '- Group by visual region: TopBar, ContentArea, BottomBar, Dialog,'
    ' NavigationDrawer, etc.\n'
    '- Element types: Button, TextField, TextView, Toggle, CheckBox,'
    ' RadioButton, Image, ScrollList, Spinner, Tab, MenuItem, etc.\n'
    '- pos values: top-left | top-center | top-right | center-left | center |'
    ' center-right | bottom-left | bottom-center | bottom-right\n'
    '- For EACH interactive element that HAS a bounding-box label in the'
    ' second screenshot (i.e. it appears in the UI Elements list above),'
    ' include `index=<n>` using EXACTLY the numeric index shown on its'
    ' label. NEVER reuse such a numeric index for anything else.\n'
    '- IMPORTANT — actively recover MISSED interactive elements. The'
    ' bounding-box layer (SOM) only marks elements that the accessibility'
    ' tree exposed as separate nodes. In practice many real, clickable'
    ' targets are MISSED, e.g.:\n'
    '    * a TextView whose parent container is the actual clickable'
    ' ("Use without an account", "Skip", "Maybe later", footer links);\n'
    '    * icons / FAB / chips / overflow buttons whose nodes were merged'
    ' into a parent;\n'
    '    * tappable empty regions inside a dialog / banner / card;\n'
    '    * cells of a grid / calendar / palette where only the container'
    ' got a single index.\n'
    '  Look at the FIRST (raw) screenshot and decide visually whether each'
    ' such element is plausibly tappable (button-like styling, link text,'
    ' icon affordance, list-row chevron, etc.). For EVERY such MISSED'
    ' interactive element, output it in the DSL with a VIRTUAL index'
    ' `vindex=v<k>` (k = 1, 2, 3 ... independently numbered, with the `v`'
    ' prefix REQUIRED) and you MUST include `bbox=<x1,y1,x2,y2>` in pixels'
    ' (your best visual estimate using neighboring labeled bboxes and the'
    ' screen size as anchors). Example:\n'
    '       TextView("Use without an account", vindex=v1, pos=bottom-center,'
    ' bbox=540,1850,1380,1930, clickable=true)\n'
    '  Rules for `vindex`:\n'
    '    * `vindex=v<k>` is RESERVED for visually-detected interactive'
    ' elements that have NO numeric SOM label. Do NOT mix it with `index`.\n'
    '    * `bbox` is MANDATORY for every `vindex` element so the next'
    ' stage can ground it to pixel coordinates.\n'
    '    * Be conservative: only assign `vindex` when the element is'
    ' clearly interactive in the raw screenshot. Purely decorative text /'
    ' images stay WITHOUT any index/vindex field.\n'
    '    * NEVER invent a numeric `index=N`. If the element has no SOM'
    ' label, it MUST use `vindex=v<k>` (or no index at all if not'
    ' interactive).\n'
    '- Include applicable attrs: clickable=true, editable=true,'
    ' scrollable=true, checked=true/false, selected=true/false,'
    ' state=<value>.\n'
    '- For controls whose INTERNAL POSITION carries semantics (SeekBar,'
    ' ProgressBar, slider, custom canvas/whiteboard, map view, etc.), ALSO'
    ' include `bbox=<x1,y1,x2,y2>` after pos=. The translation stage uses'
    ' this to interpolate precise drag points (e.g. "60% along the slider").'
    ' For ordinary clickable elements bbox is not required (unless it is'
    ' a `vindex` element, where bbox is mandatory as stated above).\n'
    '- Preserve top-to-bottom, left-to-right spatial order.\n\n'
    'Output ONLY the two sections above (with their headers), nothing else.\n'
)


# ── Stage 2: Reasoning Prompts ────────────────────────────────────────────────

REASONING_PROMPT_TEMPLATE = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on the user's goal/request, you may either answer back if the"
    ' request is a question/chat message, or complete the task by performing'
    ' actions (step by step) on the phone.\n\n'
    'At this step, you have already analyzed the current screenshot and'
    ' produced (1) a high-level natural-language summary of the screen and'
    ' (2) a hierarchical DSL describing its UI structure. You are also'
    ' given the full history of what you have done so far. Based on the'
    ' goal, the screen summary, the screen DSL and the history, you must'
    ' decide the SINGLE next action to perform from the action list below,'
    ' and explain your reasoning.\n\n'
    'User Goal: {goal}\n\n'
    'Step History (each past step is shown in detail; use it to avoid'
    ' repeating failed actions and to remember information across steps):\n'
    '{history}\n\n'
    'Current Screen Summary (what this page is and what the user can do'
    ' here):\n{screen_summary}\n\n'
    'Current Screen DSL (regions, elements, indexes and attributes):\n'
    '{screen_dsl}\n\n'
    'Available actions (action description followed by the intent format you'
    ' must output in ACTION_INTENT):\n'
    '- If you think the task has been completed, finish the task with:'
    ' `status complete`\n'
    "- If you think the task is not feasible (including cases like you don't"
    ' have enough information or can not perform some necessary actions),'
    ' finish with: `status infeasible`\n'
    "- Answer the user's question (use this BEFORE finishing for question-like"
    ' goals): `answer text="<answer_text>"`\n'
    '- Click/tap on an element on the screen. We have added marks (bounding'
    ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
    ' elements in the screenshot, use the numeric index to indicate which'
    ' element you want to click: `click index=<target_index>`. The'
    ' perception stage may ALSO emit virtual indexes `vindex=v<k>` in the'
    ' DSL for tappable elements that have no SOM label (the bbox is given'
    ' there). You may reference these too, e.g. `click index=v1`. The'
    ' next stage will ground `vindex` references into pixel coordinates'
    ' automatically.\n'
    '- Double tap on an element (e.g. to quickly select a word or zoom in):'
    ' `double_tap index=<target_index>` (also accepts `vindex=v<k>`).\n'
    '- Long press on an element on the screen, similar to click, use the'
    ' numeric label on the bounding box to indicate which element you want'
    ' to long press: `long_press index=<target_index>` (also accepts'
    ' `vindex=v<k>`).\n'
    '- Drag from a source location to a target location (e.g. reorder a'
    ' list item, move an icon to another spot, slide a SeekBar thumb to a'
    ' different position, drop a tile onto a target zone, draw on a'
    ' canvas). Form: `drag_and_drop from=<source> to=<target>`. Each of'
    ' <source>/<target> is a SHORT natural-language phrase. Whenever the'
    ' endpoint corresponds to a labeled UI element, you MUST reference it'
    ' by its index, e.g. `index=5` or `the SeekBar thumb (index=5)`. For'
    ' endpoints WITHOUT a labeled element (a free spot on a canvas/map, or'
    ' a precise position INSIDE one element such as 60% along a slider),'
    ' describe the location in words RELATIVE to a visible element or'
    ' region (e.g. `the right end of the SeekBar (index=5)`, `60% along'
    ' the SeekBar (index=5) from its left edge`, `the empty area near the'
    ' bottom-right corner of the canvas (index=8)`). DO NOT output pixel'
    ' coordinates here — the next stage will compute them.\n'
    '- Type text into a text field (this action contains clicking the text'
    ' field, typing in the text and pressing the enter, so no need to click'
    ' the target field first), use the numeric label on the bounding box to'
    ' indicate the target text field. If the field already contains text you'
    ' want to replace, add clear_text=true to wipe the existing content'
    ' first: `input_text index=<target_index> text="<text_input>"'
    ' [clear_text=true]`\n'
    '- Press the Enter key: `keyboard_enter`\n'
    '- Navigate to the home screen: `navigate_home`\n'
    '- Navigate back: `navigate_back`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the numeric index if you want to scroll a specific UI'
    ' element, leave it empty when scrolling the whole screen:'
    ' `scroll direction=<up|down|left|right> [index=<target_index>]`\n'
    '- Swipe the whole screen in one of the four directions (use this for'
    ' page/tab gestures; note the direction is OPPOSITE to scroll):'
    ' `swipe direction=<up|down|left|right>`\n'
    '- Open an app (nothing will happen if the app is not installed):'
    ' `open_app name="<app_name>"`\n'
    '- Wait for the screen to update: `wait`\n\n'
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due to'
    ' various reasons), sometimes a simple retry can solve the problem, but'
    " if it doesn't (you can see that from the history), SWITCH to other"
    ' solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks "what is my'
    ' schedule tomorrow", then you may want to open the calendar app (using'
    " the `open_app` action), look up information there, answer user's"
    ' question (using the `answer` action) and finish (using the `status`'
    ' action with complete).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish! Merely'
    ' displaying the answer on the screen is NOT sufficient (unless the goal'
    ' is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    '- Carefully review the Step History before deciding. Do NOT repeat an'
    ' action that has already been tried and failed in a similar context;'
    ' switch strategy instead.\n'
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app (nothing'
    ' will happen if the app is not installed), do not use the app drawer to'
    ' open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type something'
    ' (including password) instead of clicking characters on the keyboard'
    ' one by one. Sometimes there is some default text in the text field'
    ' you want to type in, remember to use clear_text=true to delete it'
    ' before typing.\n'
    '- For `click`, `double_tap`, `long_press`, `input_text` and indexed'
    ' `scroll`, the index you pick MUST appear in the screen DSL above —'
    ' either as a numeric `index=N` or as a virtual `vindex=v<k>`. Do NOT'
    ' invent an index that is absent from the DSL. If the element you'
    ' want is not in the DSL at all, scroll/wait first to reveal it.\n'
    '- Consider exploring the screen by using the `scroll` action with'
    ' different directions to reveal additional content.\n'
    '- Use `swipe` (not `scroll`) for whole-screen gestures like switching'
    ' between pages/tabs in a pager. Use `scroll` for revealing more'
    ' content inside a list or the current screen.\n'
    '- IMPORTANT — `scroll` direction semantics. The `direction` value of'
    ' `scroll` indicates WHERE THE ADDITIONAL CONTENT YOU WANT TO SEE LIES,'
    ' NOT where the finger moves. The underlying implementation translates'
    ' `scroll direction=X` into a finger drag toward the OPPOSITE side of'
    ' X. `swipe` is the inverse convention (its `direction` IS the finger'
    ' movement direction). Concretely:\n'
    '    * `scroll direction=down`  → finger drags from screen center'
    ' toward the TOP edge (a physical swipe-up gesture). Use this to'
    ' REVEAL CONTENT BELOW the current view — items further down in a'
    ' list, the next section beneath the fold, or the app drawer pulled'
    ' up from the home screen.\n'
    '    * `scroll direction=up`    → finger drags from screen center'
    ' toward the BOTTOM edge (a physical swipe-down gesture). Use this to'
    ' REVEAL CONTENT ABOVE the current view, to pull down the'
    ' notification shade, or to close the app drawer back to the home'
    ' screen.\n'
    '    * `scroll direction=left`  → finger drags from screen center'
    ' toward the RIGHT edge. Use this to REVEAL CONTENT TO THE LEFT.\n'
    '    * `scroll direction=right` → finger drags from screen center'
    ' toward the LEFT edge. Use this to REVEAL CONTENT TO THE RIGHT.\n'
    '    * Mnemonic: think "scroll direction = where more content is".'
    ' If unsure, you can also try the opposite direction once.\n'
    '- Concrete app-drawer rule: from the home screen, ALWAYS use'
    ' `scroll direction=down` to open the app drawer (NEVER `scroll'
    ' direction=up`, and NEVER `swipe`). To return to the home screen,'
    ' use `scroll direction=up` or `navigate_home`.\n'
    '- Use `double_tap` to quickly select a word, toggle zoom, or trigger'
    ' double-tap-specific UI behaviors.\n'
    '- When using `drag_and_drop`, your job is ONLY to specify WHAT to drag'
    ' and WHERE to drop it in plain language, anchored to elements/indexes'
    ' visible in the DSL. Do NOT think about pixel coordinates; the'
    ' translation stage will resolve indexes / coordinates automatically.'
    ' Do NOT use `drag_and_drop` as a substitute for `scroll` / `swipe`.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text'
    ' selection mode by long pressing the area where the text is, then some'
    ' of the words near the long press point will be selected (highlighted'
    ' with two pointers indicating the range) and usually a text selection'
    ' bar will also appear with options like `copy`, `paste`, `select all`,'
    ' etc. <ii> Select the exact text you need. Usually the text selected'
    ' from the previous step is NOT the one you want, you need to adjust'
    ' the range by dragging the two pointers. If you want to select all'
    ' text in the text field, simply click the `select all` button in the'
    ' bar.\n'
    "- At this point, you don't have the ability to drag arbitrary text"
    ' selection pointers, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). A faster approach is to use'
    ' `input_text` with clear_text=true to wipe the field, or first select'
    ' the text you want to delete then click backspace.\n'
    '- To copy some text: first select the exact text you want to copy,'
    ' which usually also brings up the text selection bar, then click the'
    ' `copy` button in the bar.\n'
    '- To paste text into a text box, first long press the text box, then'
    ' usually the text selection bar will appear with a `paste` button.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicates this is an enum field and'
    ' you should try to select the best match by clicking the corresponding'
    ' one in the list.\n'
    '{additional_guidelines}'
    '\nNow decide the SINGLE next action and respond in EXACTLY this format'
    ' (all four lines are required, do NOT output JSON here, the JSON'
    ' translation will be done in the next stage):\n'
    'OBSERVATION: <describe the current screen state relevant to the goal,'
    ' citing concrete elements/regions from the DSL>\n'
    'PLAN: <what you intend to do next and why, referencing history if'
    ' relevant; be critical about why previous attempts may have failed>\n'
    'TARGET: <element description with index when applicable, e.g.'
    ' Button("OK", index=3); use "none" for actions without a target such'
    ' as keyboard_enter, navigate_home, navigate_back, wait, open_app,'
    ' status, answer, or non-indexed scroll/swipe>\n'
    'ACTION_INTENT: <one of the action formats listed above, e.g.'
    ' `click index=3`, `input_text index=5 text="hello" clear_text=true`,'
    ' `scroll direction=down`, `open_app name="Settings"`,'
    ' `status complete`>\n'
)


# ── Stage 3: Translation Prompts ──────────────────────────────────────────────

TRANSLATION_PROMPT_TEMPLATE = (
    'You are a precise JSON action translator for Android automation.\n'
    'Convert the ACTION_INTENT below into a single valid JSON object.\n\n'
    'ACTION_INTENT: {action_intent}\n\n'
    'Screen size (pixels, width x height): {screen_size}\n'
    'Current UI elements (each with bbox=[x_min,y_min,x_max,y_max] and'
    ' center=[cx,cy] in pixels):\n{ui_elements}\n'
    'Current Screen DSL (regions, indexes and optional bbox):\n{screen_dsl}\n'
    'Virtual index map (vindex -> bbox & center, parsed by code from the'
    ' DSL above; ALWAYS use these numbers when the intent contains a'
    ' vindex reference, do NOT recompute):\n{vindex_map}\n\n'
    'For most actions you only need to mechanically convert the intent into'
    ' the JSON form shown below. For `drag_and_drop` you are also responsible'
    ' for GROUNDING the natural-language source/target into either a UI'
    ' element index or absolute pixel coordinates, using the UI elements and'
    ' screen size above.\n\n'
    'Virtual index (`vindex=v<k>`) grounding rules — applies to ALL actions'
    ' that take an index (`click`, `double_tap`, `long_press`, `input_text`,'
    ' indexed `scroll`, and `drag_and_drop` endpoints):\n'
    '  * A `vindex=v<k>` is a perception-only virtual marker. It does NOT'
    ' correspond to any real UI element index in the runtime; you MUST'
    ' translate it into pixel coordinates using the `bbox=<x1,y1,x2,y2>`'
    ' that the perception stage attached to that vindex element in the'
    ' Screen DSL above.\n'
    '  * For `click` / `double_tap` / `long_press`: read the bbox of that'
    ' vindex element from the DSL, compute its center'
    ' (cx, cy) = ((x1+x2)//2, (y1+y2)//2), and emit `"x": cx, "y": cy`'
    ' (do NOT emit an `index` field).\n'
    '  * For `input_text` referencing a vindex: emit the input_text JSON'
    ' WITHOUT an `index` field but WITH `"x": cx, "y": cy` so the runtime'
    ' taps the bbox center to focus the field before typing.\n'
    '  * For indexed `scroll vindex=v<k>`: convert to a non-indexed scroll'
    ' (drop the index) since the runtime cannot scroll a virtual element.\n'
    '  * For `drag_and_drop` endpoints: convert each `vindex=v<k>` endpoint'
    ' to `from_xy` / `to_xy` using the bbox center (or an interpolated'
    ' point if the natural-language phrase implies a relative position'
    ' inside it).\n'
    '  * NEVER place a `v<k>` string into the JSON `index` field. The'
    ' runtime expects an integer or no index.\n\n'
    'drag_and_drop grounding rules:\n'
    '  1) If the endpoint phrase clearly refers to a single element by'
    ' `index=N` and the intent is to drag THAT element AS A WHOLE, output'
    ' `from_index` / `to_index` (the runtime will use the element bbox'
    ' center).\n'
    '  2) If the phrase describes a RELATIVE position INSIDE one element'
    ' (e.g. "60% along (index=N)", "right end of (index=N)", "top-left of'
    ' (index=N)"), compute an absolute pixel point from that element bbox'
    ' and output `from_xy` / `to_xy`. Examples:\n'
    '       - "60% along the SeekBar (index=5) from its left edge" with'
    ' bbox=[100,300,500,360] -> x = 100 + 0.6*(500-100) = 340, y = 330,'
    ' so to_xy=[340,330].\n'
    '       - "right end of (index=5)" with the same bbox -> [500,330].\n'
    '  3) If the phrase refers to a position that is NOT on any element'
    ' (free area, blank canvas, map gesture), estimate a sensible pixel'
    ' point using nearby element bboxes and the screen size, then output'
    ' `from_xy` / `to_xy`.\n'
    '  4) Coordinates MUST stay within 0..W-1 and 0..H-1.\n'
    '  5) If you truly cannot determine a precise point, default to the'
    ' bbox center of the referenced element (i.e. emit `from_index` /'
    ' `to_index`).\n\n'
    'Format reference:\n'
    '  click index=<n>                    → {{"action_type": "click", "index": <n>}}\n'
    '  click index=v<k> (virtual)         → {{"action_type": "click", "x": <cx>, "y": <cy>}}    # cx,cy = bbox center of v<k>\n'
    '  double_tap index=<n>               → {{"action_type": "double_tap", "index": <n>}}\n'
    '  double_tap index=v<k> (virtual)    → {{"action_type": "double_tap", "x": <cx>, "y": <cy>}}\n'
    '  long_press index=<n>               → {{"action_type": "long_press", "index": <n>}}\n'
    '  long_press index=v<k> (virtual)    → {{"action_type": "long_press", "x": <cx>, "y": <cy>}}\n'
    '  input_text index=<n> text="<t>"    → {{"action_type": "input_text", "text": "<t>", "index": <n>}}\n'
    '  input_text index=v<k> text="<t>"   → {{"action_type": "input_text", "text": "<t>", "x": <cx>, "y": <cy>}}\n'
    '  input_text index=<n> text="<t>" clear_text=true → {{"action_type": "input_text", "text": "<t>", "index": <n>, "clear_text": true}}\n'
    '  keyboard_enter                     → {{"action_type": "keyboard_enter"}}\n'
    '  navigate_home                      → {{"action_type": "navigate_home"}}\n'
    '  navigate_back                      → {{"action_type": "navigate_back"}}\n'
    '  scroll direction=<d>               → {{"action_type": "scroll", "direction": "<d>"}}\n'
    '  scroll direction=<d> index=<n>     → {{"action_type": "scroll", "direction": "<d>", "index": <n>}}\n'
    '  swipe direction=<d>                → {{"action_type": "swipe", "direction": "<d>"}}\n'
    '  open_app name="<n>"                → {{"action_type": "open_app", "app_name": "<n>"}}\n'
    '  wait                               → {{"action_type": "wait"}}\n'
    '  status complete                    → {{"action_type": "status", "goal_status": "complete"}}\n'
    '  status infeasible                  → {{"action_type": "status", "goal_status": "infeasible"}}\n'
    '  answer text="<t>"                  → {{"action_type": "answer", "text": "<t>"}}\n'
    '  drag_and_drop (whole-element source AND target):\n'
    '       → {{"action_type": "drag_and_drop", "from_index": <a>, "to_index": <b>}}\n'
    '  drag_and_drop (whole-element source, intra-element target):\n'
    '       → {{"action_type": "drag_and_drop", "from_index": <a>, "to_xy": [<x>, <y>]}}\n'
    '  drag_and_drop (both endpoints need pixel computation):\n'
    '       → {{"action_type": "drag_and_drop", "from_xy": [<x1>, <y1>], "to_xy": [<x2>, <y2>]}}\n\n'
    'Output ONLY the JSON object, nothing else:\n'
)


# ── Summary Prompt ────────────────────────────────────────────────────────────

SUMMARY_PROMPT_TEMPLATE = (
    'User goal: {goal}\n'
    'Action intent executed: {action_intent}\n'
    'JSON action used: {action_json}\n\n'
    'You are given the screenshot BEFORE the action (labeled "before") and'
    ' AFTER the action (labeled "after").\n\n'
    'UI elements before:\n{before_elements}\n'
    'UI elements after:\n{after_elements}\n\n'
    'CRITICAL RULES - READ AND FOLLOW THESE STRICTLY:\n'
    '1. DEFAULT SUCCESS RULE: If a click/tap/swipe action was executed, '
    '   ALWAYS mark outcome as "succeeded" UNLESS there is CLEAR visual '
    '   evidence it failed (e.g., explicit error message, dialog still open '
    '   with same unfilled input, button still visibly untapped after click).\n'
    '2. INVISIBLE STATE CHANGES: Many actions succeed without visible UI '
    '   changes. Examples:\n'
    '   * JS internal counter incremented (UI may not reflect immediately)\n'
    '   * Tapping already-selected option / already-checked toggle\n'
    '   * Drawing thin stroke on canvas (invisible at this resolution)\n'
    '   * Drag-and-drop landing on same/equivalent slot\n'
    '   * "Save"/"Confirm" where toast already faded\n'
    '   * `wait`/`keyboard_enter`/`navigate_back` intentionally no visible diff\n'
    '   * SOM labeling differences from accessibility-tree node merging\n'
    '3. BUTTON DISAPPEARANCE = COMPLETION: If a button/element disappears '
    '   after clicking, this OFTEN means the action succeeded and triggered '
    '   a state transition (e.g., JS count completed, form submitted, page '
    '   navigated). NEVER mark this as "infeasible" or "failed" - it is '
    '   STRONG evidence of success.\n'
    '4. INCONCLUSIVE USAGE: Only use "inconclusive" when you genuinely cannot '
    '   tell if the action executed (e.g., screenshot unclear, timing issue). '
    '   DO NOT use it just because UI looks unchanged. Downstream reasoning '
    '   treats "inconclusive" as "probably failed" - misuse will break the task.\n'
    '5. INFEASIBLE PROHIBITION: NEVER mark outcome as "infeasible" just because '
    '   an element disappeared or UI changed unexpectedly. Element disappearance '
    '   usually means success, not failure. Only use "failed" for explicit errors.\n\n'
    'DECISION TREE:\n'
    '- Action executed + no error visible → "succeeded"\n'
    '- Action executed + element disappeared → "succeeded" (triggered transition)\n'
    '- Action executed + explicit error/failure visible → "failed"\n'
    '- Cannot determine if action executed at all → "inconclusive" (rare)\n\n'
    'Output STRICTLY a single-line JSON object with EXACTLY these keys'
    ' (no markdown fences, no extra text before/after):\n'
    '  {{"outcome": "succeeded" | "failed" | "inconclusive",\n'
    '   "summary": "<one line, < 50 words: what you intended, what'
    ' happened, any note for future steps>",\n'
    '   "evidence": "<one short phrase citing the visual/UI cue you'
    ' used to decide>"}}\n\n'
    'JSON: '
)


# ── Helper functions ──────────────────────────────────────────────────────────


def _generate_ui_elements_description(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate a concise text description list for UI elements.

  The output starts with a `Screen size: WxH` header and embeds each
  element's `bbox` and `center` coordinates so that the translation stage
  can ground high-level drag intents into pixel coordinates when needed.
  """
  w, h = screen_width_height_px
  tree_info = f'Screen size: {int(w)}x{int(h)}\n'
  for index, ui_element in enumerate(ui_elements):
    # Relaxed validation: include elements with bbox (skip is_visible check)
    # to match the SOM screenshot generation in perception stage.
    if ui_element.bbox_pixels:
      bbox = ui_element.bbox_pixels
      x_min, y_min = int(bbox.x_min), int(bbox.y_min)
      x_max, y_max = int(bbox.x_max), int(bbox.y_max)
      cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
      desc = f'UI element {index}: {{"index": {index}, '
      if ui_element.text:
        desc += f'"text": "{ui_element.text}", '
      if ui_element.content_description:
        desc += f'"content_description": "{ui_element.content_description}", '
      if ui_element.hint_text:
        desc += f'"hint_text": "{ui_element.hint_text}", '
      desc += f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
      desc += f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
      if ui_element.is_scrollable:
        desc += '"is_scrollable": True, '
      desc += f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
      desc += f'"bbox": [{x_min}, {y_min}, {x_max}, {y_max}], '
      desc += f'"center": [{cx}, {cy}]'
      desc += '}\n'
      tree_info += desc
  return tree_info


def _extract_vindex_map(
    screen_dsl: str,
) -> dict[str, tuple[int, int, int, int]]:
  """Parse perception DSL and return {v<k>: (x_min,y_min,x_max,y_max)}.

  The perception stage marks SOM-missed interactive elements with
  `vindex=v<k>` and a mandatory `bbox=<x1,y1,x2,y2>`. This helper extracts
  that map so downstream stages can ground the virtual references into
  pixel coordinates without relying on the LLM to do bbox arithmetic.
  """
  if not screen_dsl:
    return {}
  pattern = re.compile(
      r'vindex\s*=\s*(v\d+)[^\n)]*?bbox\s*=\s*\[?\s*'
      r'(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?'
  )
  result: dict[str, tuple[int, int, int, int]] = {}
  for m in pattern.finditer(screen_dsl):
    result[m.group(1)] = (
        int(m.group(2)), int(m.group(3)),
        int(m.group(4)), int(m.group(5)),
    )
  return result


def _vindex_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
  x_min, y_min, x_max, y_max = bbox
  return ((x_min + x_max) // 2, (y_min + y_max) // 2)


def _format_vindex_map(
    vindex_map: dict[str, tuple[int, int, int, int]],
) -> str:
  if not vindex_map:
    return '(none)'
  lines = []
  for k in sorted(vindex_map.keys(), key=lambda s: int(s[1:])):
    b = vindex_map[k]
    cx, cy = _vindex_center(b)
    lines.append(
        f'  {k}: bbox=[{b[0]},{b[1]},{b[2]},{b[3]}], center=[{cx},{cy}]'
    )
  return '\n'.join(lines)


def _is_vindex_ref(value) -> bool:
  return (
      isinstance(value, str)
      and len(value) >= 2
      and value[0] == 'v'
      and value[1:].isdigit()
  )


def _resolve_vindex_in_action_dict(
    action_dict: dict,
    vindex_map: dict[str, tuple[int, int, int, int]],
) -> dict:
  """Replace string vindex references inside a JSONAction dict by xy.

  Defensive post-processing for the translation LLM: even if the model
  emits {"index":"v1"} or {"from_index":"v1"}, this helper rewrites
  them into proper {"x":cx,"y":cy} / {"from_xy":[cx,cy]} forms based on
  the bbox map extracted from the perception DSL.
  """
  if not isinstance(action_dict, dict) or not vindex_map:
    return action_dict
  # Single-point actions: click / double_tap / long_press / input_text.
  idx = action_dict.get('index')
  if _is_vindex_ref(idx) and idx in vindex_map:
    cx, cy = _vindex_center(vindex_map[idx])
    action_dict['x'] = cx
    action_dict['y'] = cy
    action_dict.pop('index', None)
  # drag_and_drop endpoints.
  for ep_idx_key, ep_xy_key in (
      ('from_index', 'from_xy'),
      ('to_index', 'to_xy'),
  ):
    v = action_dict.get(ep_idx_key)
    if _is_vindex_ref(v) and v in vindex_map:
      cx, cy = _vindex_center(vindex_map[v])
      action_dict[ep_xy_key] = [cx, cy]
      action_dict.pop(ep_idx_key, None)
  # Defensive: handle from_xy / to_xy accidentally set to a vindex string.
  for xy_key in ('from_xy', 'to_xy'):
    v = action_dict.get(xy_key)
    if _is_vindex_ref(v) and v in vindex_map:
      cx, cy = _vindex_center(vindex_map[v])
      action_dict[xy_key] = [cx, cy]
  return action_dict


def _parse_perception_output(output: str) -> tuple[str, str]:
  """Parse Stage 1 output into (screen_summary, screen_dsl).

  The perception stage is asked to produce two sections separated by
  '=== SCREEN_SUMMARY ===' and '=== SCREEN_DSL ===' headers. This helper
  extracts both. If the headers are missing, the entire output is treated
  as the DSL and the summary is left empty (so reasoning still works).

  Args:
    output: The raw text output from the perception stage.

  Returns:
    A tuple of (screen_summary, screen_dsl).
  """
  if not output:
    return '', ''

  summary_match = re.search(
      r'={2,}\s*SCREEN_SUMMARY\s*={2,}\s*(.*?)(?=={2,}\s*SCREEN_DSL\s*={2,}|$)',
      output,
      re.DOTALL | re.IGNORECASE,
  )
  dsl_match = re.search(
      r'={2,}\s*SCREEN_DSL\s*={2,}\s*(.*)',
      output,
      re.DOTALL | re.IGNORECASE,
  )

  screen_summary = summary_match.group(1).strip() if summary_match else ''
  if dsl_match:
    screen_dsl = dsl_match.group(1).strip()
  else:
    # Fallback: treat the whole output as DSL when headers are missing.
    screen_dsl = output.strip()
  return screen_summary, screen_dsl


def _parse_reasoning_output(
    output: str,
) -> tuple[str, str, str, str]:
  """Parse Stage 2 output into (observation, plan, target, action_intent).

  Extracts the four required fields from the structured reasoning output.
  Each field value spans until the next field key or end of string.

  Args:
    output: The raw text output from the reasoning stage.

  Returns:
    A tuple of (observation, plan, target, action_intent). Any field not
    found in the output will be an empty string.
  """
  keys = ['OBSERVATION', 'PLAN', 'TARGET', 'ACTION_INTENT']
  results: dict[str, str] = {k: '' for k in keys}

  # Build a pattern that matches each key up to the next key or end-of-string.
  alternation = '|'.join(keys)
  for key in keys:
    pattern = rf'{key}:\s*(.*?)(?=(?:{alternation}):|$)'
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
    if match:
      results[key] = match.group(1).strip()

  return (
      results['OBSERVATION'],
      results['PLAN'],
      results['TARGET'],
      results['ACTION_INTENT'],
  )


# ── PipelineM3A Agent ─────────────────────────────────────────────────────────

class PipelineM3A(base_agent.EnvironmentInteractingAgent):
  """Three-stage Pipeline M3A Agent.

  Decomposes the single LLM call of standard M3A into three specialized stages:

  Stage 1 – Perception (perception_llm, multimodal):
    Receives raw screenshot + SOM-labeled screenshot + UI element list.
    Outputs a hierarchical DSL description of the current screen.

  Stage 2 – Reasoning (reasoning_llm, text):
    Receives screen DSL + goal + full step-by-step history.
    Each history entry includes the observation, action intent, and result
    summary from that step, giving the model full context to avoid repeating
    failed actions and make informed decisions.
    Outputs a structured analysis with OBSERVATION / PLAN / TARGET /
    ACTION_INTENT fields.

  Stage 3 – Translation (translation_llm, text):
    Receives the ACTION_INTENT string.
    Outputs the corresponding JSON action for execution.

  Each stage uses an independently configurable MultimodalLlmWrapper.
  Stages 2 and 3 are called with an empty image list (text-only).
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      perception_llm: Optional[infer.MultimodalLlmWrapper] = None,
      reasoning_llm: Optional[infer.MultimodalLlmWrapper] = None,
      translation_llm: Optional[infer.MultimodalLlmWrapper] = None,
      perception_model_name: str = 'qwen-vl-plus',
      reasoning_model_name: str = 'qwen-plus',
      translation_model_name: str = 'qwen-plus',
      name: str = 'PipelineM3A',
      wait_after_action_seconds: float = 2.0,
  ):
    """Initializes PipelineM3A.

    Stages 2 (reasoning) and 3 (translation) are TEXT-ONLY: they call
    `predict_mm` with an empty image list. So they should normally use a
    cheap text model (e.g. ``qwen-plus``) instead of a multimodal model.
    Only stage 1 (perception) and the per-step summary genuinely need a
    vision-capable model (e.g. ``qwen-vl-plus``).

    For convenience, when a wrapper argument is left as ``None`` this class
    auto-constructs an ``infer.DashScopeWrapper`` using the corresponding
    ``*_model_name``. The defaults are tuned for the recommended
    perception=``qwen-vl-plus`` + reasoning/translation=``qwen-plus``
    setup. To use a different backend (OpenAI / Gemini / etc.), build the
    wrapper instance yourself and pass it in explicitly — in that case the
    matching ``*_model_name`` argument is ignored.

    Args:
      env: The Android environment.
      perception_llm: Multimodal LLM for Stage 1 (screen → DSL) and the
        per-step summary. Defaults to a DashScope wrapper of
        ``perception_model_name``.
      reasoning_llm: LLM for Stage 2 (DSL + goal + history → action intent).
        Text-only call. Defaults to a DashScope wrapper of
        ``reasoning_model_name``.
      translation_llm: LLM for Stage 3 (action intent → JSON action).
        Text-only call. Defaults to a DashScope wrapper of
        ``translation_model_name``.
      perception_model_name: Default model name for stage 1 / summary when
        ``perception_llm`` is None. Must be a vision-capable model.
      reasoning_model_name: Default model name for stage 2 when
        ``reasoning_llm`` is None. A text model is sufficient.
      translation_model_name: Default model name for stage 3 when
        ``translation_llm`` is None. A text model is sufficient.
      name: Agent display name.
      wait_after_action_seconds: Seconds to wait after executing an action.
    """
    super().__init__(env, name)

    def _default_dashscope(
        model_name: str,
    ) -> infer.MultimodalLlmWrapper:
      return infer.DashScopeWrapper(model_name=model_name)

    self.perception_llm = perception_llm or _default_dashscope(
        perception_model_name
    )
    self.reasoning_llm = reasoning_llm or _default_dashscope(
        reasoning_model_name
    )
    self.translation_llm = translation_llm or _default_dashscope(
        translation_model_name
    )
    self.history: list[dict] = []
    self.additional_guidelines: Optional[list[str]] = None
    self.wait_after_action_seconds = wait_after_action_seconds

  def set_task_guidelines(self, task_guidelines: list[str]) -> None:
    self.additional_guidelines = task_guidelines

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.env.hide_automation_ui()
    self.history = []

  def step(self, goal: str) -> base_agent.AgentInteractionResult:  # pylint: disable=too-many-return-statements
    step_data = {
        'raw_screenshot': None,
        'before_screenshot_with_som': None,
        'after_screenshot_with_som': None,
        # Stage 1
        'perception_prompt': None,
        'perception_output': None,
        'screen_summary': None,
        'screen_dsl': None,
        # Stage 2
        'reasoning_prompt': None,
        'reasoning_output': None,
        'observation': None,
        'plan': None,
        'target': None,
        'action_intent': None,
        # Stage 3
        'translation_prompt': None,
        'translation_output': None,
        'action_output_json': None,
        # Summary
        'summary_prompt': None,
        'summary_output': None,
        'summary': None,
        # Three-state outcome judged by the summary stage:
        #   'succeeded' | 'failed' | 'inconclusive' | None
        # `None` means the step terminated before the summary stage ran
        # (e.g. early exit on parse / validation error).
        'outcome': None,
        'summary_evidence': None,
    }

    print('----------step ' + str(len(self.history) + 1))

    # ── Capture current state ────────────────────────────────────────────────
    state = self.get_post_transition_state()
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary

    before_ui_elements = state.ui_elements
    ui_elements_text = _generate_ui_elements_description(
        before_ui_elements, logical_screen_size
    )

    step_data['raw_screenshot'] = state.pixels.copy()
    before_screenshot = state.pixels.copy()

    # Build SOM (Set-of-Marks) labeled screenshot
    # Use relaxed validation for perception: include elements with valid bbox
    # even if is_visible=False (some apps mark visible elements incorrectly).
    for index, ui_element in enumerate(before_ui_elements):
      # Only check if bbox exists and is non-empty (skip is_visible check)
      if ui_element.bbox_pixels:
        m3a_utils.add_ui_element_mark(
            before_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
    step_data['before_screenshot_with_som'] = before_screenshot.copy()

    # ── Stage 1: Perception ──────────────────────────────────────────────────
    perception_prompt = PERCEPTION_PROMPT_TEMPLATE.format(
        ui_elements=ui_elements_text if ui_elements_text else 'Not available',
    )
    step_data['perception_prompt'] = perception_prompt

    screen_dsl, _, perception_response = self.perception_llm.predict_mm(
        perception_prompt,
        [step_data['raw_screenshot'], before_screenshot],
    )

    if not perception_response:
      step_data['summary'] = (
          'Error calling LLM in perception stage (Stage 1).'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    step_data['perception_output'] = screen_dsl
    screen_summary, screen_dsl = _parse_perception_output(screen_dsl)
    if not screen_summary:
      screen_summary = '(no screen summary produced)'
    step_data['screen_summary'] = screen_summary
    step_data['screen_dsl'] = screen_dsl
    vindex_map = _extract_vindex_map(screen_dsl)
    step_data['vindex_map'] = vindex_map
    logging.info('Screen summary:\n%s', screen_summary)
    logging.info('Screen DSL:\n%s', screen_dsl)
    if vindex_map:
      logging.info('Vindex map: %s', vindex_map)

    # ── Stage 2: Reasoning ───────────────────────────────────────────────────
    if self.history:
      history_parts = []
      for i, s in enumerate(self.history):
        lines = [f'--- Step {i + 1} ---']
        if s.get('observation'):
          lines.append(f'  OBSERVATION: {s["observation"]}')
        if s.get('action_intent'):
          lines.append(f'  ACTION: {s["action_intent"]}')
        if s.get('summary'):
          lines.append(f'  RESULT: {s["summary"]}')
        history_parts.append('\n'.join(lines))
      history_text = '\n'.join(history_parts)
    else:
      history_text = 'No actions taken yet.'

    if self.additional_guidelines:
      additional_guidelines_text = (
          'Additional task guidelines:\n'
          + '\n'.join(f'- {g}' for g in self.additional_guidelines)
          + '\n'
      )
    else:
      additional_guidelines_text = ''

    reasoning_prompt = REASONING_PROMPT_TEMPLATE.format(
        goal=goal,
        history=history_text,
        screen_summary=screen_summary,
        screen_dsl=screen_dsl,
        additional_guidelines=additional_guidelines_text,
    )
    step_data['reasoning_prompt'] = reasoning_prompt

    reasoning_output, _, reasoning_response = self.reasoning_llm.predict_mm(
        reasoning_prompt,
        [],  # text-only: no images needed for reasoning
    )

    if not reasoning_response:
      step_data['summary'] = (
          'Error calling LLM in reasoning stage (Stage 2).'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    step_data['reasoning_output'] = reasoning_output
    logging.info('Reasoning output:\n%s', reasoning_output)

    observation, plan, target, action_intent = _parse_reasoning_output(
        reasoning_output
    )
    step_data['observation'] = observation
    step_data['plan'] = plan
    step_data['target'] = target
    step_data['action_intent'] = action_intent

    if not action_intent:
      logging.info(
          'Reasoning output missing ACTION_INTENT field:\n%s', reasoning_output
      )
      step_data['summary'] = (
          'Reasoning stage output is not in the correct format:'
          ' ACTION_INTENT field not found.'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    logging.info('Action intent: %s', action_intent)

    # ── Stage 3: Translation ─────────────────────────────────────────────────
    translation_prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        action_intent=action_intent,
        screen_size=(
            f'{logical_screen_size[0]}x{logical_screen_size[1]}'
        ),
        ui_elements=(
            ui_elements_text if ui_elements_text else 'Not available'
        ),
        screen_dsl=screen_dsl if screen_dsl else 'Not available',
        vindex_map=_format_vindex_map(vindex_map),
    )
    step_data['translation_prompt'] = translation_prompt

    translation_output, _, translation_response = self.translation_llm.predict_mm(
        translation_prompt,
        [],  # text-only: no images needed for translation
    )

    if not translation_response:
      step_data['summary'] = (
          'Error calling LLM in translation stage (Stage 3).'
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    step_data['translation_output'] = translation_output
    logging.info('Translation output: %s', translation_output)

    # Parse the translated output into a JSONAction
    try:
      parsed_action = agent_utils.extract_json(translation_output)
      parsed_action = _resolve_vindex_in_action_dict(
          parsed_action, vindex_map
      )
      converted_action = json_action.JSONAction(**parsed_action)
      step_data['action_output_json'] = converted_action
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Failed to parse translation output to JSONAction: %s', e)
      step_data['summary'] = (
          'Translation stage failed to produce a valid JSON action. '
          'Output was: %s' % translation_output
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    # Validate index bounds
    action_index = converted_action.index
    num_ui_elements = len(before_ui_elements)
    if (
        converted_action.action_type
        in ['click', 'long_press', 'input_text', 'scroll']
        and action_index is not None
    ):
      if action_index >= num_ui_elements:
        logging.info(
            'Index out of range: predicted %s, but only %d elements exist.',
            action_index,
            num_ui_elements,
        )
        step_data['summary'] = (
            'Action index %d is out of range (only %d UI elements).'
            % (action_index, num_ui_elements)
        )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)

      # Highlight the target element on the raw screenshot
      m3a_utils.add_ui_element_mark(
          step_data['raw_screenshot'],
          before_ui_elements[action_index],
          action_index,
          logical_screen_size,
          physical_frame_boundary,
          orientation,
      )

    # Validate drag_and_drop endpoints (each side: index OR xy).
    if converted_action.action_type == 'drag_and_drop':
      w, h = logical_screen_size

      def _check(idx, xy, name):
        if idx is not None:
          if idx < 0 or idx >= num_ui_elements:
            return (
                f'{name}_index={idx} out of range (only {num_ui_elements}'
                ' UI elements).'
            )
          return None
        if xy is not None:
          x, y = xy
          if not (0 <= x < w and 0 <= y < h):
            return f'{name}_xy=({x},{y}) outside screen {w}x{h}.'
          return None
        return f'{name} endpoint missing (need index or xy).'

      for err in (
          _check(
              converted_action.from_index, converted_action.from_xy, 'from'
          ),
          _check(converted_action.to_index, converted_action.to_xy, 'to'),
      ):
        if err:
          step_data['summary'] = 'drag_and_drop invalid: ' + err
          self.history.append(step_data)
          return base_agent.AgentInteractionResult(False, step_data)

      # Visualize endpoints: index endpoints reuse add_ui_element_mark.
      if converted_action.from_index is not None:
        m3a_utils.add_ui_element_mark(
            step_data['raw_screenshot'],
            before_ui_elements[converted_action.from_index],
            converted_action.from_index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
      if converted_action.to_index is not None:
        m3a_utils.add_ui_element_mark(
            step_data['raw_screenshot'],
            before_ui_elements[converted_action.to_index],
            converted_action.to_index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

    # Handle terminal actions
    if converted_action.action_type == 'status':
      if converted_action.goal_status == 'infeasible':
        logging.info('Agent declared task infeasible.')
      step_data['summary'] = (
          'Agent declared the task complete/infeasible. '
          'Plan: %s' % plan
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(True, step_data)

    if converted_action.action_type == 'answer':
      logging.info('Agent answered: %s', converted_action.text)

    # Execute the action
    try:
      self.env.execute_action(converted_action)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.info('Failed to execute action: %s', e)
      step_data['summary'] = (
          'Failed to execute action %s: %s' % (converted_action, e)
      )
      self.history.append(step_data)
      return base_agent.AgentInteractionResult(False, step_data)

    time.sleep(self.wait_after_action_seconds)

    # ── Summary ──────────────────────────────────────────────────────────────
    state = self.env.get_state(wait_to_stabilize=False)
    logical_screen_size = self.env.logical_screen_size
    orientation = self.env.orientation
    physical_frame_boundary = self.env.physical_frame_boundary
    after_ui_elements = state.ui_elements
    after_ui_elements_text = _generate_ui_elements_description(
        after_ui_elements, logical_screen_size
    )
    after_screenshot = state.pixels.copy()
    for index, ui_element in enumerate(after_ui_elements):
      if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
        m3a_utils.add_ui_element_mark(
            after_screenshot,
            ui_element,
            index,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

    m3a_utils.add_screenshot_label(
        step_data['before_screenshot_with_som'], 'before'
    )
    m3a_utils.add_screenshot_label(after_screenshot, 'after')
    step_data['after_screenshot_with_som'] = after_screenshot.copy()

    summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
        goal=goal,
        action_intent=action_intent,
        action_json=str(converted_action),
        before_elements=ui_elements_text,
        after_elements=after_ui_elements_text,
    )
    step_data['summary_prompt'] = summary_prompt

    summary_output, _, summary_response = self.perception_llm.predict_mm(
        summary_prompt,
        [step_data['before_screenshot_with_som'], after_screenshot],
    )
    step_data['summary_output'] = summary_output

    # Parse the three-state JSON. Be tolerant: if the LLM falls back to
    # legacy free-form text, treat the whole text as the summary and
    # mark the outcome as `inconclusive` so reasoning keeps going
    # without retrying.
    outcome = 'inconclusive'
    summary_text = '(summary unavailable)'
    evidence = ''
    if not summary_response:
      logging.info('Error calling LLM in summary stage.')
    else:
      parsed = agent_utils.extract_json(summary_output or '')
      if isinstance(parsed, dict) and 'outcome' in parsed:
        raw_outcome = str(parsed.get('outcome', '')).strip().lower()
        if raw_outcome in ('succeeded', 'failed', 'inconclusive'):
          outcome = raw_outcome
        summary_text = str(
            parsed.get('summary') or parsed.get('evidence') or ''
        ).strip() or '(empty summary)'
        evidence = str(parsed.get('evidence', '')).strip()
      else:
        # Legacy / malformed output: keep raw text as summary, leave
        # outcome=inconclusive so we don't accidentally trigger retries.
        summary_text = (summary_output or '').strip() or '(empty summary)'

    step_data['outcome'] = outcome
    step_data['summary_evidence'] = evidence
    step_data['summary'] = 'Action intent: %s. [%s] %s' % (
        action_intent,
        outcome,
        summary_text,
    )
    logging.info('Outcome: %s | Summary: %s', outcome, summary_text)

    self.history.append(step_data)
    return base_agent.AgentInteractionResult(False, step_data)
