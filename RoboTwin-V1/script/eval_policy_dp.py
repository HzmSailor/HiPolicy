import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def convert_state_dict_keys(checkpoint_state_dict, target_keys):
    """
    智能转换checkpoint中的键名以匹配目标模型的键名格式
    处理torch.compile导致的_orig_mod前缀问题和normalizer参数
    """
    converted_dict = {}
    checkpoint_keys = set(checkpoint_state_dict.keys())
    
    # 检查是否需要添加或移除_orig_mod前缀
    target_has_orig_mod = any(key.startswith('_orig_mod.') for key in target_keys)
    checkpoint_has_orig_mod = any('_orig_mod' in key for key in checkpoint_keys)
    
    print(f"目标模型是否有_orig_mod前缀: {target_has_orig_mod}")
    print(f"checkpoint是否有_orig_mod前缀: {checkpoint_has_orig_mod}")
    
    # 统计normalizer相关键
    target_normalizer_keys = [k for k in target_keys if 'normalizer.params_dict' in k]
    checkpoint_normalizer_keys = [k for k in checkpoint_keys if 'normalizer.params_dict' in k]
    print(f"目标模型normalizer键数量: {len(target_normalizer_keys)}")
    print(f"checkpoint normalizer键数量: {len(checkpoint_normalizer_keys)}")
    
    if target_has_orig_mod and not checkpoint_has_orig_mod:
        # 目标模型有前缀，checkpoint没有，需要添加前缀
        print("正在为checkpoint键名添加_orig_mod前缀...")
        for key, value in checkpoint_state_dict.items():
            new_key = f"_orig_mod.{key}"
            if new_key in target_keys:
                converted_dict[new_key] = value
            else:
                # 如果添加前缀后仍然不在目标中，保持原键名
                converted_dict[key] = value
                
    elif not target_has_orig_mod and checkpoint_has_orig_mod:
        # 目标模型没有前缀，checkpoint有，需要移除前缀
        print("正在从checkpoint键名移除_orig_mod前缀...")
        for key, value in checkpoint_state_dict.items():
            new_key = key
            
            # 按顺序移除所有可能的前缀
            if new_key.startswith('module._orig_mod.'):
                # 同时包含module和_orig_mod前缀
                new_key = new_key[len('module._orig_mod.'):]
                print(f"同时移除module._orig_mod前缀: {key} -> {new_key}")
            elif new_key.startswith('_orig_mod.'):
                # 只有_orig_mod前缀
                new_key = new_key[len('_orig_mod.'):]
                print(f"移除_orig_mod前缀: {key} -> {new_key}")
            elif new_key.startswith('module.'):
                # 只有module前缀
                new_key = new_key[len('module.'):]
                print(f"移除module前缀: {key} -> {new_key}")
            
            if new_key in target_keys:
                converted_dict[new_key] = value
            else:
                # 特别处理normalizer参数：即使不在目标键中也要保留转换后的键
                if 'normalizer.params_dict' in key:
                    print(f"发现normalizer参数键: {key} -> {new_key}")
                    # 保留转换后的键，让load_state_dict尝试加载
                    converted_dict[new_key] = value
                else:
                    # 非normalizer键保持原键名
                    converted_dict[key] = value
    else:
        # 目标和checkpoint的前缀情况一致，但仍需要检查键匹配
        print("键名格式一致，检查键匹配情况...")
        for key, value in checkpoint_state_dict.items():
            if key in target_keys:
                converted_dict[key] = value
            else:
                # 保留不匹配的键，以防其他逻辑需要
                converted_dict[key] = value
    
    # 检查转换结果
    matched_keys = set(converted_dict.keys()) & target_keys
    matched_normalizer_keys = [k for k in matched_keys if 'normalizer.params_dict' in k]
    print(f"成功匹配的键数量: {len(matched_keys)} / {len(target_keys)}")
    print(f"成功匹配的normalizer键数量: {len(matched_normalizer_keys)}")
    
    if len(matched_keys) < len(target_keys) * 0.8:  # 如果匹配率低于80%，警告
        missing_keys = target_keys - set(converted_dict.keys())
        missing_normalizer_keys = [k for k in missing_keys if 'normalizer.params_dict' in k]
        print(f"警告: 仍有较多键名未匹配，缺失的键: {list(missing_keys)[:10]}...")  # 只显示前10个
        if missing_normalizer_keys:
            print(f"缺失的normalizer键: {missing_normalizer_keys[:5]}...")
    
    return converted_dict

def get_policy(checkpoint, output_dir, device):
    
    # load checkpoint
    payload = torch.load(open('./policy/Diffusion-Policy/'+checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # 临时禁用torch.compile以避免推理时的参数键名不匹配问题
    original_torch_compile = cfg.training.get('torch_compile', True)
    cfg.training.torch_compile = False
    print(f"推理模式：临时禁用torch.compile (原值: {original_torch_compile})")
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    
    try:
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    except RuntimeError as e:
        error_msg = str(e)
        print(f"加载checkpoint时出现错误: {error_msg}")
        
        # 处理torch.compile相关的_orig_mod前缀问题
        if "Missing key(s) in state_dict" in error_msg and "_orig_mod" in error_msg:
            print("检测到torch.compile相关的参数键名不匹配问题，尝试智能修复...")
            
            # 获取当前模型和checkpoint的state_dict键
            model_keys = set(workspace.model.state_dict().keys())
            checkpoint_keys = set(payload['state_dicts']['model'].keys())
            
            print(f"模型期望的键数量: {len(model_keys)}")
            print(f"checkpoint中的键数量: {len(checkpoint_keys)}")
            
            # 尝试键名转换
            converted_state_dict = convert_state_dict_keys(payload['state_dicts']['model'], model_keys)
            
            try:
                workspace.model.load_state_dict(converted_state_dict, strict=False)
                if 'ema_model' in payload['state_dicts'] and workspace.ema_model is not None:
                    ema_converted = convert_state_dict_keys(payload['state_dicts']['ema_model'], 
                                                           set(workspace.ema_model.state_dict().keys()))
                    workspace.ema_model.load_state_dict(ema_converted, strict=False)
                
                # 特别检查normalizer是否正确加载
                if hasattr(workspace.model, 'normalizer') and hasattr(workspace.model.normalizer, 'params_dict'):
                    print(f"模型normalizer参数键: {list(workspace.model.normalizer.params_dict.keys())}")
                    if len(workspace.model.normalizer.params_dict) == 0:
                        print("警告: 模型normalizer参数为空，检查checkpoint中是否存在normalizer参数")
                        # 尝试在checkpoint中搜索normalizer相关键
                        normalizer_keys_in_checkpoint = [k for k in payload['state_dicts']['model'].keys() 
                                                       if 'normalizer' in k.lower()]
                        print(f"checkpoint中发现的normalizer相关键: {normalizer_keys_in_checkpoint[:10]}")
                        
                        # 检查是否有_orig_mod前缀的normalizer键
                        orig_mod_normalizer_keys = [k for k in payload['state_dicts']['model'].keys() 
                                                  if '_orig_mod' in k and 'normalizer' in k]
                        print(f"checkpoint中_orig_mod.normalizer键: {orig_mod_normalizer_keys[:5]}")
                    else:
                        print("✓ normalizer参数加载成功")
                print("torch.compile键名转换成功")
                
                # 手动加载其他组件
                if 'optimizer' in payload['state_dicts']:
                    try:
                        workspace.optimizer.load_state_dict(payload['state_dicts']['optimizer'])
                    except:
                        print("优化器状态加载失败，跳过")
                if 'lr_scheduler' in payload['state_dicts']:
                    try:
                        workspace.lr_scheduler.load_state_dict(payload['state_dicts']['lr_scheduler'])
                    except:
                        print("学习率调度器状态加载失败，跳过")
                if 'global_step' in payload:
                    workspace.global_step = payload['global_step']
                if 'epoch' in payload:
                    workspace.epoch = payload['epoch']
            except Exception as conv_e:
                print(f"键名转换失败: {conv_e}")
                raise e
                
        elif "Missing key(s) in state_dict" in error_msg and any(key in error_msg for key in [".mean", ".std"]):
            print(f"警告: 检查点缺少ImageNet normalization参数，尝试忽略这些键: {e}")
            print("这通常发生在使用新版本代码评估旧版本训练的模型时")
            
            # 获取当前模型的state_dict键
            model_keys = set(workspace.model.state_dict().keys())
            checkpoint_keys = set(payload['state_dicts']['model'].keys())
            
            # 找到缺失的键
            missing_keys = model_keys - checkpoint_keys
            print(f"缺失的键: {missing_keys}")
            
            # 过滤掉ImageNet normalization相关的缺失键
            imagenet_missing_keys = [key for key in missing_keys if key.endswith(('.mean', '.std'))]
            if imagenet_missing_keys:
                print(f"忽略ImageNet normalization缺失键: {imagenet_missing_keys}")
                # 使用strict=False加载，允许缺失键
                workspace.model.load_state_dict(payload['state_dicts']['model'], strict=False)
                if 'ema_model' in payload['state_dicts']:
                    workspace.ema_model.load_state_dict(payload['state_dicts']['ema_model'], strict=False)
                # 手动加载其他组件
                if 'optimizer' in payload['state_dicts']:
                    workspace.optimizer.load_state_dict(payload['state_dicts']['optimizer'])
                if 'lr_scheduler' in payload['state_dicts']:
                    workspace.lr_scheduler.load_state_dict(payload['state_dicts']['lr_scheduler'])
                if 'global_step' in payload:
                    workspace.global_step = payload['global_step']
                if 'epoch' in payload:
                    workspace.epoch = payload['epoch']
            else:
                raise e
        else:
            raise e
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    # 检查normalizer状态
    if hasattr(policy, 'normalizer') and hasattr(policy.normalizer, 'params_dict'):
        print(f"最终policy normalizer参数键: {list(policy.normalizer.params_dict.keys())}")
        if len(policy.normalizer.params_dict) == 0:
            print("严重警告: policy normalizer参数为空！")
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

class DP:
    def __init__(self, task_name, head_camera_type: str, checkpoint_num: int, expert_data_num: int, seed: int, ensemble: bool = False, no_hierarchical_cond: bool = False):
        self.policy = get_policy(f'checkpoints/{task_name}_{head_camera_type}_{expert_data_num}_{seed}/{checkpoint_num}.ckpt', None, 'cuda:0')
        self.policy.ensemble = ensemble
        self.policy.no_hierarchical_cond = no_hierarchical_cond
        # 从 policy 中获取正确的 n_obs_steps 值
        n_obs_steps = self.policy.n_obs_steps
        self.runner = DPRunner(output_dir=None, n_obs_steps=n_obs_steps)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def test_policy(task_name, Demo_class, args, dp: DP, st_seed, test_num=20):
    expert_check = True
    print("Task name: ", args["task_name"])


    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    

    now_seed = st_seed
    while succ_seed < test_num:
        render_freq = args['render_freq']
        args['render_freq'] = 0
        
        if expert_check:
            try:
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
                Demo_class.play_once()
                Demo_class.close()
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                args['render_freq'] = render_freq
                print('error occurs !')
                continue

        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            continue


        args['render_freq'] = render_freq

        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        Demo_class.apply_dp(dp, args)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        dp.runner.reset_obs()
        print(f"{task_name} success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1

    return now_seed, Demo_class.suc

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, '../task_config/_camera_config.yml')

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def main(usr_args):
    task_name = usr_args.task_name
    head_camera_type = usr_args.head_camera_type
    checkpoint_num = usr_args.checkpoint_num
    seed = usr_args.seed
    ensemble = usr_args.ensemble
    no_hierarchical_cond = usr_args.no_hierarchical_cond

    with open(f'./task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['head_camera_type'] = head_camera_type 
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # output camera config
    print('============= Camera Config =============\n')
    print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '\n    fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '\n    camera_h: ' + str(args['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '\n    fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '\n    camera_h: ' + str(args['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '\n    fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '\n    camera_h: ' + str(args['front_camera_h']))
    print('\n=======================================')

    args['expert_seed'] = seed
    args['expert_data_num'] = usr_args.expert_data_num

    task = class_decorator(args['task_name'])

    st_seed = 100000 * (1+seed)
    suc_nums = []
    test_num = 100 
    topk = 1

    dp = DP(task_name, head_camera_type, checkpoint_num, usr_args.expert_data_num, seed, ensemble, no_hierarchical_cond)

    st_seed, suc_num = test_policy(task_name, task, args, dp, st_seed, test_num=test_num)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    save_dir = Path(f'eval_result/dp/{task_name}_{usr_args.head_camera_type}/{usr_args.expert_data_num}')
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f'ckpt_{checkpoint_num}_seed_{seed}.txt'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, 'w') as file:
        file.write(f'Timestamp: {current_time}\n\n')

        file.write(f'Checkpoint Num: {checkpoint_num}\n')
        
        file.write('Successful Rate of Diffenent checkpoints:\n')
        file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate (every):\n')
        file.write('\n'.join(map(str, np.array(topk_success_rate) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate:\n')
        file.write(f'\n'.join(map(str, np.array(topk_success_rate) / (topk * test_num))))
        file.write('\n\n')

    print(f'Data has been saved to {file_path}')



if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    
    parser = ArgumentParser()
    parser.add_argument('task_name', type=str, default='block_hammer_beat')
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('expert_data_num', type=int, default=20)
    parser.add_argument('checkpoint_num', type=int, default=1000)
    parser.add_argument('seed', type=int, default=0)
    parser.add_argument('--ensemble', action='store_true', default=False, help='是否使用ensemble')
    parser.add_argument('--no_hierarchical_cond', type=str, default='false', help='是否使用no_hierarchical_cond模式')
    usr_args = parser.parse_args()
    
    # 将字符串转换为布尔值
    usr_args.no_hierarchical_cond = usr_args.no_hierarchical_cond.lower() == 'true' or usr_args.no_hierarchical_cond == '1'
    
    main(usr_args)
