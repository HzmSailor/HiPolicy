#!/usr/bin/env python3
"""
多卡训练性能测试脚本
用于验证性能优化效果，对比优化前后的训练速度

使用方法:
python test_training_efficiency.py --task block_hammer_beat --gpus 0,1,2,3 --epochs 2
"""

import os
import sys
import time
import argparse
import subprocess
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="多卡训练性能测试")
    parser.add_argument("--task", type=str, default="block_hammer_beat", 
                       help="任务名称")
    parser.add_argument("--head_camera_type", type=str, default="D435",
                       help="相机类型") 
    parser.add_argument("--expert_data_num", type=int, default=100,
                       help="专家数据数量")
    parser.add_argument("--gpus", type=str, default="0,1,2,3",
                       help="使用的GPU列表，逗号分隔")
    parser.add_argument("--epochs", type=int, default=2,
                       help="测试训练轮数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    return parser.parse_args()

def run_training_test(task_name, head_camera_type, expert_data_num, gpu_list, epochs, seed):
    """运行训练测试"""
    print(f"🚀 开始性能测试：{task_name} - GPU: {gpu_list}")
    
    # 设置测试环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu_list
    
    # 构建训练命令
    gpu_count = len(gpu_list.split(','))
    
    train_cmd = [
        "bash", "policy/Diffusion-Policy/train.sh",
        task_name, head_camera_type, str(expert_data_num),
        str(seed), gpu_list
    ]
    
    print(f"执行命令: {' '.join(train_cmd)}")
    print(f"GPU数量: {gpu_count}")
    print(f"测试轮数: {epochs}")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 运行训练
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            cwd='/root/RoboTwin-1-unet'
        )
        
        # 实时输出并收集日志
        performance_logs = []
        epoch_times = []
        
        for line in process.stdout:
            print(line, end='')
            
            # 解析性能信息
            if "性能分析报告:" in line:
                performance_logs.append(line.strip())
            elif "总时间:" in line and "平均batch:" in line:
                try:
                    # 提取epoch时间
                    parts = line.split("|")
                    if len(parts) >= 2:
                        time_part = parts[0].split(":")[-1].strip()
                        epoch_time = float(time_part.replace('s', ''))
                        epoch_times.append(epoch_time)
                except:
                    pass
        
        process.wait()
        total_time = time.time() - start_time
        
        return {
            'success': process.returncode == 0,
            'total_time': total_time,
            'epoch_times': epoch_times,
            'avg_epoch_time': sum(epoch_times) / len(epoch_times) if epoch_times else 0,
            'performance_logs': performance_logs,
            'gpu_count': gpu_count
        }
        
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        return {
            'success': False,
            'error': str(e),
            'total_time': time.time() - start_time,
            'gpu_count': gpu_count
        }

def generate_performance_report(results):
    """生成性能报告"""
    print("\n" + "="*80)
    print("🎯 多卡训练性能测试报告")
    print("="*80)
    
    if not results['success']:
        print(f"❌ 测试失败: {results.get('error', '未知错误')}")
        return
    
    gpu_count = results['gpu_count']
    avg_epoch = results['avg_epoch_time']
    total_time = results['total_time']
    epoch_count = len(results['epoch_times'])
    
    print(f"📊 基础统计信息:")
    print(f"   GPU数量: {gpu_count}")
    print(f"   总测试时间: {total_time:.2f}s")
    print(f"   完成轮数: {epoch_count}")
    print(f"   平均轮次时间: {avg_epoch:.2f}s")
    
    if epoch_count > 0:
        throughput = gpu_count / avg_epoch if avg_epoch > 0 else 0
        print(f"   训练吞吐量: {throughput:.3f} GPU/s")
        
        # 估算多卡加速比
        single_gpu_estimate = avg_epoch * gpu_count
        speedup = single_gpu_estimate / avg_epoch if avg_epoch > 0 else 0
        efficiency = speedup / gpu_count * 100 if gpu_count > 0 else 0
        
        print(f"\n🚀 多卡性能分析:")
        print(f"   理论加速比: {speedup:.2f}x")
        print(f"   并行效率: {efficiency:.1f}%")
        
        if efficiency < 70:
            print("   ⚠️  并行效率偏低，建议进一步优化通信和同步")
        elif efficiency < 85:
            print("   ✅ 并行效率良好")
        else:
            print("   🏆 并行效率优秀")
    
    # 显示详细性能日志
    if results['performance_logs']:
        print(f"\n📋 详细性能分析:")
        for log in results['performance_logs'][-3:]:  # 显示最后3条
            print(f"   {log}")
    
    print("="*80)

def main():
    args = parse_args()
    
    print(f"🎯 多卡训练性能测试")
    print(f"任务: {args.task}")
    print(f"相机: {args.head_camera_type}")
    print(f"数据量: {args.expert_data_num}")
    print(f"GPU: {args.gpus}")
    print(f"测试轮数: {args.epochs}")
    
    # 检查数据是否存在
    zarr_path = f"/DATA/disk0/data_robotwin_new/processed_data/{args.task}_{args.head_camera_type}_{args.expert_data_num}.zarr"
    if not os.path.exists(zarr_path):
        print(f"⚠️  数据不存在: {zarr_path}")
        print("正在生成数据...")
        
    # 运行性能测试
    results = run_training_test(
        args.task, 
        args.head_camera_type,
        args.expert_data_num,
        args.gpus, 
        args.epochs,
        args.seed
    )
    
    # 生成报告
    generate_performance_report(results)
    
    # 保存结果
    result_file = f"performance_test_{args.task}_{len(args.gpus.split(','))}gpu_{int(time.time())}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 详细结果已保存到: {result_file}")

if __name__ == "__main__":
    main()