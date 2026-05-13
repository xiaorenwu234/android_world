#!/usr/bin/env python3
"""读取任务执行结果的脚本"""

import gzip
import pickle
import sys
from pathlib import Path

def read_episode_data(filepath):
    """读取单个任务的执行数据"""
    try:
        with gzip.open(filepath, 'rb') as f:
            episodes = pickle.load(f)
        
        if isinstance(episodes, list):
            return episodes
        else:
            return [episodes]
    except Exception as e:
        print(f"  读取失败: {e}")
        return None

def main():
    # 最新的运行目录
    run_dir = Path.home() / "android_world/runs/run_20260513T211911363574"
    
    print(f"运行目录: {run_dir}")
    print("=" * 80)
    
    total_tasks = 0
    successful_tasks = 0
    failed_tasks = 0
    task_results = []
    
    for pkl_file in sorted(run_dir.glob("*.pkl.gz")):
        task_name = pkl_file.name.replace('.pkl.gz', '')
        
        episodes = read_episode_data(pkl_file)
        if episodes is None:
            continue
        
        for episode in episodes:
            print(f"\n任务文件: {pkl_file.name}")
            print("-" * 80)
            print(f"  任务目标: {episode.get('goal', 'N/A')}")
            print(f"  是否成功: {episode.get('is_successful', 'N/A')}")
            print(f"  任务模板: {episode.get('task_template', 'N/A')}")
            print(f"  智能体: {episode.get('agent_name', 'N/A')}")
            print(f"  运行时间: {episode.get('run_time', 'N/A')}")
            print(f"  步数: {episode.get('episode_length', 'N/A')}")
            if episode.get('exception_info'):
                print(f"  异常信息: {episode.get('exception_info')}")
            
            total_tasks += 1
            if episode.get('is_successful'):
                successful_tasks += 1
            else:
                failed_tasks += 1
            
            task_results.append({
                'name': task_name,
                'success': episode.get('is_successful', False)
            })
    
    print("\n" + "=" * 80)
    print("任务执行统计:")
    print("=" * 80)
    print(f"总任务数: {total_tasks}")
    print(f"成功: {successful_tasks}")
    print(f"失败: {failed_tasks}")
    if total_tasks > 0:
        print(f"成功率: {successful_tasks/total_tasks*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("详细结果:")
    print("=" * 80)
    for i, result in enumerate(task_results, 1):
        status = "✓ 成功" if result['success'] else "✗ 失败"
        print(f"  {i}. {result['name']} - {status}")

if __name__ == "__main__":
    main()
