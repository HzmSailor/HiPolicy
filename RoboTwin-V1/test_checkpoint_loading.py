#!/usr/bin/env python3
"""
测试checkpoint加载修复是否生效
"""
import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import torch
import os
from script.eval_policy_dp import get_policy

def test_checkpoint_loading():
    """测试checkpoint加载功能"""
    
    # 查找可用的checkpoint
    checkpoint_dir = "./policy/Diffusion-Policy/checkpoints"
    available_checkpoints = []
    
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_path = os.path.relpath(os.path.join(root, file), "./policy/Diffusion-Policy")
                available_checkpoints.append(checkpoint_path)
    
    if not available_checkpoints:
        print("未找到可测试的checkpoint文件")
        return False
    
    # 使用第一个找到的checkpoint进行测试
    test_checkpoint = available_checkpoints[0]
    print(f"测试checkpoint: {test_checkpoint}")
    
    try:
        print("=" * 60)
        print("开始测试checkpoint加载...")
        policy = get_policy(test_checkpoint, output_dir=None, device='cuda:0')
        print("=" * 60)
        print("✅ checkpoint加载成功！")
        print(f"模型类型: {type(policy).__name__}")
        print(f"模型参数数量: {sum(p.numel() for p in policy.parameters()):,}")
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ checkpoint加载失败: {e}")
        print("错误详情:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("RoboTwin Checkpoint加载修复测试")
    print("=" * 60)
    
    success = test_checkpoint_loading()
    
    if success:
        print("\n🎉 测试通过！修复方案生效。")
    else:
        print("\n💥 测试失败，需要进一步调试。")