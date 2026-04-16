#!/usr/bin/env python3
"""
DDP Bug修复验证测试脚本
快速测试单卡和多卡训练是否能正常启动
"""

import os
import sys
import subprocess
import time

def test_single_gpu():
    """测试单卡训练"""
    print("🔍 测试单卡训练...")
    
    cmd = [
        "bash", "policy/Diffusion-Policy/train.sh",
        "container_place", "D435", "100", "0", "0"
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, cwd='/root/RoboTwin-1-unet'
        )
        
        # 等待10秒看是否有AttributeError
        start_time = time.time()
        while time.time() - start_time < 10:
            if process.poll() is not None:
                break
            time.sleep(0.1)
        
        # 杀死进程
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        
        stdout, stderr = process.communicate()
        
        # 检查是否有AttributeError
        if "AttributeError" in stderr or "AttributeError" in stdout:
            print("❌ 单卡测试失败: 仍有AttributeError")
            print("错误信息:", stderr[-500:])  # 显示最后500字符
            return False
        else:
            print("✅ 单卡测试通过: 没有AttributeError")
            return True
            
    except Exception as e:
        print(f"❌ 单卡测试异常: {e}")
        return False

def test_multi_gpu():
    """测试多卡训练"""
    print("🔍 测试多卡训练...")
    
    cmd = [
        "bash", "policy/Diffusion-Policy/train.sh", 
        "container_place", "D435", "100", "0", "0,1"
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=True, cwd='/root/RoboTwin-1-unet'
        )
        
        # 等待15秒看是否有AttributeError
        start_time = time.time()
        while time.time() - start_time < 15:
            if process.poll() is not None:
                break
            time.sleep(0.1)
        
        # 杀死进程
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
            
        stdout, stderr = process.communicate()
        
        # 检查是否有AttributeError
        if "AttributeError" in stderr or "AttributeError" in stdout:
            print("❌ 多卡测试失败: 仍有AttributeError")
            print("错误信息:", stderr[-500:])  # 显示最后500字符  
            return False
        else:
            print("✅ 多卡测试通过: 没有AttributeError")
            return True
            
    except Exception as e:
        print(f"❌ 多卡测试异常: {e}")
        return False

def main():
    print("🎯 DDP AttributeError Bug修复验证测试")
    print("=" * 50)
    
    # 检查数据是否存在
    zarr_path = "/DATA/disk0/data_robotwin_new/processed_data/container_place_D435_100.zarr"
    if not os.path.exists(zarr_path):
        print(f"⚠️  测试数据不存在: {zarr_path}")
        print("请先生成测试数据或更换其他任务")
        return False
    
    single_ok = test_single_gpu()
    multi_ok = test_multi_gpu()
    
    print("\n" + "=" * 50)
    print("🎯 测试结果总结:")
    print(f"   单卡训练: {'✅ 通过' if single_ok else '❌ 失败'}")
    print(f"   多卡训练: {'✅ 通过' if multi_ok else '❌ 失败'}")
    
    if single_ok and multi_ok:
        print("🏆 所有测试通过！DDP AttributeError bug已修复")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)