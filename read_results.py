#!/usr/bin/env python3
"""读取任务执行结果的脚本

用法:
  python read_results.py                        # 读取默认目录，仅摘要
  python read_results.py --dir=<run_dir>        # 指定目录
  python read_results.py --steps               # 同时输出每步摘要
  python read_results.py --pipeline            # 输出 PipelineM3A 三阶段详情
  python read_results.py --task=MarkorCreateNote  # 只看指定任务
"""

import argparse
import gzip
import io
import pickle
import textwrap
from pathlib import Path


# ── 常量 ─────────────────────────────────────────────────────────────────────

# PipelineM3A 独有字段（按阶段排列）
PIPELINE_STEP_FIELDS = [
    ('screen_dsl',        'Stage 1 DSL'),
    ('observation',       'Stage 2 观察'),
    ('plan',              'Stage 2 计划'),
    ('target',            'Stage 2 目标元素'),
    ('action_intent',     'Stage 2 动作意图'),
    ('translation_output','Stage 3 翻译输出'),
    ('action_output_json','Stage 3 JSONAction'),
    ('summary',           '摘要'),
]

# 通用步骤字段（M3A / PipelineM3A 共有）
COMMON_STEP_FIELDS = [
    ('action_output',     '动作输出'),
    ('summary',           '摘要'),
]


# ── 读取 ──────────────────────────────────────────────────────────────────────

def _load_pkl_gz(filepath):
    """读取 .pkl.gz 文件，兼容直接 gzip 和先读后解压两种格式。"""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
        try:
            data = pickle.load(gzip.open(io.BytesIO(raw), 'rb'))
        except Exception:
            data = pickle.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f'  读取失败: {e}')
        return None


# ── 输出工具 ──────────────────────────────────────────────────────────────────

def _wrap(text, width=100, indent=6):
    """对长文本自动换行缩进输出。"""
    if text is None:
        return ' ' * indent + 'N/A'
    text = str(text)
    lines = text.splitlines()
    result = []
    prefix = ' ' * indent
    for line in lines:
        if len(prefix + line) <= width:
            result.append(prefix + line)
        else:
            result.extend(
                textwrap.wrap(line, width=width - indent,
                              initial_indent=prefix,
                              subsequent_indent=prefix + '  ')
            )
    return '\n'.join(result) if result else prefix + '(空)'


def _print_step_field(label, value, width=100):
    print(f'      [{label}]')
    print(_wrap(value, width=width, indent=8))


# ── 核心输出 ──────────────────────────────────────────────────────────────────

def print_episode_summary(episode, task_name):
    """输出单个 episode 的摘要行。"""
    success = episode.get('is_successful', 'N/A')
    goal    = episode.get('goal', 'N/A')
    agent   = episode.get('agent_name', 'N/A')
    status  = '✓ 成功' if success else '✗ 失败'
    ep_data = episode.get('episode_data', {})
    n_steps = len(ep_data.get('summary', [])) if ep_data else 'N/A'
    print(f'  {status}  [{agent}]  步数={n_steps}  {task_name}')
    print(f'      目标: {goal}')
    if episode.get('exception_info'):
        print(f'      异常: {episode["exception_info"]}')


def print_steps(episode, pipeline=False):
    """输出每步详情。pipeline=True 时输出三阶段字段。"""
    ep_data = episode.get('episode_data', {})
    if not ep_data:
        print('      (无步骤数据)')
        return

    fields = PIPELINE_STEP_FIELDS if pipeline else COMMON_STEP_FIELDS
    # 步数以 summary 列表长度为准
    n_steps = len(ep_data.get('summary', []))
    if n_steps == 0:
        print('      (步骤数为 0)')
        return

    for i in range(n_steps):
        print(f'\n      ── Step {i + 1} ──')
        for key, label in fields:
            values = ep_data.get(key)
            if values is None:
                continue
            val = values[i] if i < len(values) else None
            _print_step_field(label, val)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='读取 AndroidWorld 任务结果')
    parser.add_argument('--dir', default=None,
                        help='运行目录（默认自动找最新目录）')
    parser.add_argument('--steps', action='store_true',
                        help='输出每步 summary')
    parser.add_argument('--pipeline', action='store_true',
                        help='输出 PipelineM3A 三阶段详情（包含 DSL/推理/转译）')
    parser.add_argument('--task', default=None,
                        help='只显示包含此字符串的任务文件名')
    args = parser.parse_args()

    # 确定运行目录
    runs_root = Path.home() / 'android_world/runs'
    if args.dir:
        run_dir = Path(args.dir)
    else:
        candidates = sorted(runs_root.glob('run_*'))
        if not candidates:
            print(f'未找到运行目录: {runs_root}')
            return
        run_dir = candidates[-1]  # 最新目录

    print(f'运行目录: {run_dir}')
    print('=' * 80)

    total, ok, fail = 0, 0, 0
    task_results = []

    for pkl_file in sorted(run_dir.glob('*.pkl.gz')):
        task_name = pkl_file.stem.replace('.pkl', '')
        if args.task and args.task.lower() not in pkl_file.name.lower():
            continue

        episodes = _load_pkl_gz(pkl_file)
        if episodes is None:
            continue

        print(f'\n文件: {pkl_file.name}')
        print('-' * 80)

        for episode in episodes:
            print_episode_summary(episode, task_name)
            if args.pipeline or args.steps:
                print_steps(episode, pipeline=args.pipeline)

            total += 1
            if episode.get('is_successful'):
                ok += 1
            else:
                fail += 1
            task_results.append({
                'name': task_name,
                'success': episode.get('is_successful', False),
            })

    # 统计
    print('\n' + '=' * 80)
    print('统计:')
    print('=' * 80)
    print(f'  总任务: {total}   成功: {ok}   失败: {fail}', end='')
    if total > 0:
        print(f'   成功率: {ok / total * 100:.1f}%')
    else:
        print()

    print()
    for i, r in enumerate(task_results, 1):
        status = '✓' if r['success'] else '✗'
        print(f'  {i:3}. {status}  {r["name"]}')


if __name__ == '__main__':
    main()
