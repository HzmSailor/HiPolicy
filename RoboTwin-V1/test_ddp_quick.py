#!/usr/bin/env python3
"""
快速DDP测试 - 直接测试上下文管理器问题
"""

import os
import sys
sys.path.insert(0, '/root/RoboTwin-1-unet/policy/Diffusion-Policy')

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

def test_context_managers():
    """测试上下文管理器的创建"""
    print("🔍 测试上下文管理器...")
    
    # 测试nullcontext
    try:
        ctx1 = nullcontext()
        print(f"✅ nullcontext() 创建成功: {type(ctx1)}")
        print(f"   有__enter__方法: {hasattr(ctx1, '__enter__')}")
        print(f"   有__exit__方法: {hasattr(ctx1, '__exit__')}")
        
        with ctx1:
            print("   ✅ nullcontext可以正常使用")
    except Exception as e:
        print(f"❌ nullcontext测试失败: {e}")
        return False
    
    # 测试DDP模型的no_sync
    try:
        # 创建简单模型
        model = torch.nn.Linear(10, 1)
        
        # 检查普通模型是否有no_sync（应该没有）
        if hasattr(model, 'no_sync'):
            print(f"⚠️  普通模型居然有no_sync方法")
        else:
            print(f"✅ 普通模型正常，没有no_sync方法")
            
        # 测试DDP包装
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model = model.to(device)
            
            # 初始化进程组（单机测试）
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', rank=0, world_size=1)
            
            # 创建DDP模型
            ddp_model = DDP(model, device_ids=[0])
            
            if hasattr(ddp_model, 'no_sync'):
                print(f"✅ DDP模型有no_sync方法")
                
                # 测试no_sync的返回值
                sync_ctx = ddp_model.no_sync()
                print(f"   no_sync()返回类型: {type(sync_ctx)}")
                print(f"   有__enter__方法: {hasattr(sync_ctx, '__enter__')}")
                print(f"   有__exit__方法: {hasattr(sync_ctx, '__exit__')}")
                
                # 测试使用
                with sync_ctx:
                    print("   ✅ DDP no_sync()可以正常使用")
            else:
                print(f"❌ DDP模型没有no_sync方法")
                return False
                
        else:
            print("⚠️  CUDA不可用，跳过DDP测试")
            
    except Exception as e:
        print(f"❌ DDP测试失败: {e}")
        return False
    
    return True

def main():
    print("🎯 快速DDP上下文管理器测试")
    print("=" * 50)
    
    success = test_context_managers()
    
    print("\n" + "=" * 50)
    if success:
        print("🏆 所有上下文管理器测试通过！")
    else:
        print("❌ 上下文管理器测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)