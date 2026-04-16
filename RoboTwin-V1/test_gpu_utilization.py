#!/usr/bin/env python3
"""
GPU利用率监控脚本
用于测试多卡训练的性能优化效果

使用方法：
在训练过程中运行: python test_gpu_utilization.py

或者后台运行: nohup python test_gpu_utilization.py > gpu_monitor.log 2>&1 &
"""
import subprocess
import time
import signal
import sys
from datetime import datetime

class GPUMonitor:
    def __init__(self, interval=2):
        self.interval = interval
        self.running = True
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print("\n正在停止GPU监控...")
        self.running = False
        sys.exit(0)
    
    def get_gpu_stats(self):
        """获取GPU统计信息"""
        try:
            # 获取GPU利用率、显存使用率和温度
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
            else:
                return None
        except Exception as e:
            print(f"获取GPU状态失败: {e}")
            return None
    
    def monitor(self):
        """开始监控"""
        print("开始监控GPU利用率...")
        print("按Ctrl+C停止监控")
        print("-" * 80)
        print(f"{'时间':<19} {'GPU':<4} {'型号':<20} {'GPU利用率':<8} {'显存利用率':<9} {'显存使用':<15} {'温度':<6}")
        print("-" * 80)
        
        low_utilization_count = 0
        total_checks = 0
        
        while self.running:
            stats = self.get_gpu_stats()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if stats:
                for line in stats:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            gpu_id = parts[0]
                            name = parts[1][:18]  # 截断名称
                            gpu_util = int(parts[2])
                            mem_util = int(parts[3])
                            mem_used = int(parts[4])
                            mem_total = int(parts[5])
                            temp = parts[6]
                            
                            mem_usage = f"{mem_used}MB/{mem_total}MB"
                            
                            # 标记低利用率
                            util_status = f"{gpu_util:>3}%"
                            if gpu_util < 20:
                                util_status += " ⚠️"
                                low_utilization_count += 1
                            
                            print(f"{current_time} GPU{gpu_id:<3} {name:<20} {util_status:<8} {mem_util:>3}%     "
                                  f"{mem_usage:<15} {temp:>3}°C")
                            
                            total_checks += 1
            else:
                print(f"{current_time} 无法获取GPU状态")
            
            time.sleep(self.interval)
        
        # 显示统计结果
        if total_checks > 0:
            low_util_percentage = (low_utilization_count / total_checks) * 100
            print(f"\n监控统计:")
            print(f"总检查次数: {total_checks}")
            print(f"低利用率(<20%)次数: {low_utilization_count}")
            print(f"低利用率占比: {low_util_percentage:.1f}%")
            
            if low_util_percentage > 50:
                print("⚠️  GPU利用率较低，建议检查数据加载和模型配置")
            elif low_util_percentage > 20:
                print("ℹ️  GPU利用率中等，还有优化空间")
            else:
                print("✅ GPU利用率良好")

def main():
    print("=== GPU利用率监控工具 ===")
    print("此工具将持续监控GPU利用率，帮助诊断多卡训练性能问题")
    print()
    
    monitor = GPUMonitor(interval=2)  # 每2秒检查一次
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main()