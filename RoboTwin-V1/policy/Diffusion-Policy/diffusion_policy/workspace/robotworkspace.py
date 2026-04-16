if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import tqdm
import numpy as np
from torch import amp
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
import wandb
import time

class NoSyncContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return None

OmegaConf.register_new_resolver("eval", eval, replace=True)

class RobotWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # 初始化DDP - 只有在真正的分布式环境中才启用
        has_dist_env = all(env in os.environ for env in ['RANK', 'WORLD_SIZE'])
        self.use_ddp = cfg.training.get('use_ddp', False) and has_dist_env
        if self.use_ddp:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.local_rank = 0
            self.world_size = 1
            self.rank = 0
            self.device = torch.device(cfg.training.device)

        # 性能后端设置（Ampere+ 建议开启）
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # Debug: CUDA/self device mapping overview
        if cfg.training.get('debug', False):
            # print(f"[DEBUG] cuda.is_available={torch.cuda.is_available()}, use_ddp={self.use_ddp}, world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}")
            if torch.cuda.is_available():
                try:
                    current_idx = torch.cuda.current_device()
                    device_name = torch.cuda.get_device_name(current_idx)
                except Exception:
                    current_idx, device_name = 'N/A', 'N/A'
                # print(f"[DEBUG] device={self.device}, current_device={current_idx}, device_name={device_name}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")

        # set seed (确保每个进程有不同的随机种子)
        seed = cfg.training.seed + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.model.to(self.device)

        # 可选：channels_last 与 torch.compile 优化 - 保守策略
        if cfg.training.get('channels_last', False):  # 默认关闭以避免维度问题
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
                if cfg.training.get('debug', False) and hasattr(self, 'rank') and self.rank == 0:
                    print("[INFO] channels_last optimization enabled")
            except Exception as e:
                if cfg.training.get('debug', False) and hasattr(self, 'rank') and self.rank == 0:
                    print(f"[WARNING] channels_last optimization failed: {e}, using default format")
        if cfg.training.get('torch_compile', True) and hasattr(torch, 'compile'):  # 默认启用
            try:
                # 多卡训练时使用更保守的编译模式
                compile_mode = 'reduce-overhead' if not self.use_ddp else 'default'
                self.model = torch.compile(self.model, mode=compile_mode)
                if cfg.training.get('debug', False) and self.rank == 0:
                    print(f"[INFO] torch.compile enabled with mode: {compile_mode}")
            except Exception as e:
                if cfg.training.get('debug', False) and self.rank == 0:
                    print(f"[WARNING] torch.compile failed: {e}, fallback to eager.")

        # if cfg.training.get('debug', False):
        #     try:
        #         # print(f"[DEBUG] model device: {next(self.model.parameters()).device}")
        #     except Exception as e:
        #         print(f"[DEBUG] model device inspection failed: {e}")

        # 如果使用DDP，包装模型并优化通信
        if self.use_ddp:
            # 高级DDP通信优化配置
            # 根据模型大小和GPU数量动态调整bucket大小（更激进的优化）
            bucket_size_mb = max(50, min(128, 32 * self.world_size))
            
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                gradient_as_bucket_view=True,  # 减少内存使用并加速通信
                bucket_cap_mb=bucket_size_mb,  # 动态调整bucket提升通信效率
                find_unused_parameters=False,  # 对于Diffusion Policy通常不需要
                broadcast_buffers=True,  # 同步BN等缓冲区
                static_graph=True,  # 静态图优化，适用于固定模型结构
            )

            # 设置DDP进程组通信后端优化
            if hasattr(dist, 'get_backend') and dist.get_backend() == 'nccl':
                # NCCL特定优化 - 更激进的多卡通信配置
                os.environ['NCCL_BUFFSIZE'] = '67108864'   # 64MB缓冲区，降低延迟
                os.environ['NCCL_P2P_DISABLE'] = '0'       # 启用P2P以提升通信效率
                os.environ['NCCL_IB_DISABLE'] = '0'        # 启用InfiniBand
                os.environ['NCCL_NET_GDR_LEVEL'] = '3'     # 启用GPU Direct RDMA
                os.environ['NCCL_ALGO'] = 'Tree,Ring'      # 优化通信算法
                os.environ['NCCL_MAX_NCHANNELS'] = '16'    # 增加通信通道数
                os.environ['NCCL_MIN_NCHANNELS'] = '8'     # 设置最小通道数

            self.policy_model = self.model.module
        else:
            self.policy_model = self.model

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.policy_model)
            self.ema_model.to(self.device)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # 初始化混合精度训练
        self.use_amp = cfg.training.get('use_amp', True)
        if self.use_amp:
            self.scaler = amp.GradScaler('cuda')
        else:
            self.scaler = None

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.horizon = self.policy_model.horizon
        self.n_obs_steps = self.policy_model.n_obs_steps

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed
        head_camera_type = cfg.head_camera_type

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # 创建训练数据加载器（支持DDP）
        if self.use_ddp:
            train_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=cfg.dataloader.shuffle)
            train_dataloader_cfg = copy.deepcopy(cfg.dataloader)
            train_dataloader_cfg.shuffle = False  # sampler已处理shuffle
            train_dataloader = create_dataloader_ddp(dataset, sampler=train_sampler, **train_dataloader_cfg)
        else:
            train_dataloader = create_dataloader(dataset, **cfg.dataloader)

        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        if self.use_ddp:
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            val_dataloader_cfg = copy.deepcopy(cfg.val_dataloader)
            val_dataloader_cfg.shuffle = False
            val_dataloader = create_dataloader_ddp(val_dataset, sampler=val_sampler, **val_dataloader_cfg)
        else:
            val_dataloader = create_dataloader(val_dataset, **cfg.val_dataloader)

        self.policy_model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging (仅主进程记录)
        if not self.use_ddp or self.rank == 0:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )
        else:
            wandb_run = None

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer (已在__init__中完成)
        if not hasattr(self, 'device'):
            self.device = torch.device(cfg.training.device)

        optimizer_to(self.optimizer, self.device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # 设置DDP sampler的epoch
                if self.use_ddp:
                    train_dataloader.sampler.set_epoch(self.epoch)

                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    if self.use_ddp:
                        self.model.module.obs_encoder.eval()
                        self.model.module.obs_encoder.requires_grad_(False)
                    else:
                        self.policy_model.obs_encoder.eval()
                        self.policy_model.obs_encoder.requires_grad_(False)

                train_losses = list()
                train_gripper_losses = list()
                epoch_start_time = time.time()
                
                # 详细性能监控变量
                data_load_time = 0.0
                data_transfer_time = 0.0
                forward_time = 0.0
                backward_time = 0.0
                optimizer_time = 0.0
                ddp_sync_time = 0.0
                total_compute_time = 0.0
                gpu_util_samples = []
                
                # 性能计数器
                batch_count = 0

                # 仅主进程显示进度条
                if not self.use_ddp or self.rank == 0:
                    train_iter = tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                            leave=False, mininterval=cfg.training.tqdm_interval_sec)
                else:
                    train_iter = train_dataloader

                for batch_idx, batch in enumerate(train_iter):
                    batch_start_time = time.time()
                    batch_count += 1
                    
                    # 性能监控 - 数据传输时间
                    transfer_start = time.time()
                    batch = dataset.postprocess(batch, self.device)
                    data_transfer_time += time.time() - transfer_start

                    # Debug: Inspect batch tensor devices on first few steps
                    if cfg.training.get('debug', False) and (batch_idx < 3) and (self.rank == 0):
                        try:
                            model_device = next(self.policy_model.parameters()).device
                            obs_devices = {}
                            if isinstance(batch.get('obs'), dict):
                                for k, v in batch['obs'].items():
                                    if torch.is_tensor(v):
                                        obs_devices[k] = str(v.device)
                            action_device = str(batch['action'].device) if torch.is_tensor(batch.get('action')) else 'N/A'
                            # print(f"[DEBUG][batch] model_device={model_device}, action_device={action_device}, obs_devices={obs_devices}")
                        except Exception as e:
                            pass

                    # GPU利用率监控 (仅采样部分batch以减少开销)
                    if (cfg.training.get('debug', False) or (batch_idx % 10 == 0)) and self.rank == 0:
                        try:
                            import pynvml
                            if not hasattr(self, '_nvml_initialized'):
                                pynvml.nvmlInit()
                                self._nvml_initialized = True
                            # Prefer current CUDA device if available
                            index = self.local_rank
                            if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                                try:
                                    index = torch.cuda.current_device()
                                except Exception:
                                    index = self.local_rank
                            # Map visible index to absolute NVML index using CUDA_VISIBLE_DEVICES
                            abs_index = index
                            try:
                                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                                if visible:
                                    mapping = [int(x) for x in visible.split(',') if x != '']
                                    if 0 <= index < len(mapping):
                                        abs_index = mapping[index]
                            except Exception:
                                abs_index = index
                            handle = pynvml.nvmlDeviceGetHandleByIndex(abs_index)
                            # synchronize to reflect recent kernels
                            if torch.cuda.is_available():
                                try:
                                    torch.cuda.synchronize()
                                except Exception:
                                    pass
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_util_samples.append(util.gpu)
                            if cfg.training.get('debug', False) and batch_idx < 5:
                                pass
                        except Exception:
                            pass

                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # 混合精度训练优化 - 梯度累积
                    # 保持原始梯度累积设置，不根据GPU数量调整
                    base_grad_acc = int(cfg.training.gradient_accumulate_every)
                    # 修复：保持原始梯度累积设置，避免loss随GPU数量变化
                    grad_acc = base_grad_acc
                    is_sync_step = ((self.global_step + 1) % grad_acc == 0)
                    
                    # 简化DDP同步控制 - 避免兼容性问题
                    ddp_sync_ctx = NoSyncContext()
                    
                    # DDP同步时间监控
                    sync_start_time = time.time() if self.use_ddp and is_sync_step else None

                    if self.use_amp and self.scaler is not None:
                        # 详细前向传播计时分析
                        forward_start = time.time()
                        
                        # 数据传输到GPU计时
                        data_transfer_start = time.time()
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True) if hasattr(x, 'to') else x)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        data_transfer_time = time.time() - data_transfer_start
                        
                        # 前向传播主体计时
                        compute_start = time.time()
                        try:
                            with ddp_sync_ctx:
                                with amp.autocast('cuda'):
                                    loss, gripper_loss = self.policy_model.compute_loss(batch)
                                    # 修复：为了梯度累积，loss需要缩放，但记录时要恢复原始值
                                    loss = loss / grad_acc
                        except Exception as e:
                            if cfg.training.get('debug', False) and self.rank == 0:
                                print(f"[ERROR] Forward pass failed: {e}")
                            raise
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        compute_time = time.time() - compute_start
                        total_forward_time = time.time() - forward_start
                        
                        forward_time += total_forward_time
                        
                        # 收集详细性能数据
                        if not hasattr(self, 'detailed_timing'):
                            self.detailed_timing = {
                                'data_transfer_times': [],
                                'compute_times': [],
                                'total_forward_times': []
                            }
                        
                        self.detailed_timing['data_transfer_times'].append(data_transfer_time)
                        self.detailed_timing['compute_times'].append(compute_time)
                        self.detailed_timing['total_forward_times'].append(total_forward_time)
                        
                        # 反向传播计时
                        backward_start = time.time()
                        try:
                            with ddp_sync_ctx:
                                self.scaler.scale(loss).backward()
                        except Exception as e:
                            if cfg.training.get('debug', False) and self.rank == 0:
                                print(f"[ERROR] Backward pass failed: {e}")
                            raise
                        backward_time += time.time() - backward_start
                        
                        # DDP同步时间计算
                        if sync_start_time is not None:
                            ddp_sync_time += time.time() - sync_start_time

                        # 优化器步骤计时
                        if is_sync_step:
                            optimizer_start = time.time()
                            # 梯度裁剪 - AMP版本
                            self.scaler.unscale_(self.optimizer)
                            if self.use_ddp:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            optimizer_time += time.time() - optimizer_start
                    else:
                        # 详细前向传播计时分析（FP32）
                        forward_start = time.time()
                        
                        # 数据传输到GPU计时
                        data_transfer_start = time.time()
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True) if hasattr(x, 'to') else x)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        data_transfer_time = time.time() - data_transfer_start
                        
                        # 前向传播主体计时
                        compute_start = time.time()
                        try:
                            with ddp_sync_ctx:
                                loss, gripper_loss = self.policy_model.compute_loss(batch)
                                # 修复：为了梯度累积，loss需要缩放，但记录时要恢复原始值
                                loss = loss / grad_acc
                        except Exception as e:
                            if cfg.training.get('debug', False) and self.rank == 0:
                                print(f"[ERROR] Forward pass (FP32) failed: {e}")
                            raise
                        
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        compute_time = time.time() - compute_start
                        total_forward_time = time.time() - forward_start
                        
                        forward_time += total_forward_time
                        
                        # 收集详细性能数据
                        if not hasattr(self, 'detailed_timing'):
                            self.detailed_timing = {
                                'data_transfer_times': [],
                                'compute_times': [],
                                'total_forward_times': []
                            }
                        
                        self.detailed_timing['data_transfer_times'].append(data_transfer_time)
                        self.detailed_timing['compute_times'].append(compute_time)
                        self.detailed_timing['total_forward_times'].append(total_forward_time)
                        
                        # 反向传播计时（FP32）
                        backward_start = time.time()
                        try:
                            with ddp_sync_ctx:
                                loss.backward()
                        except Exception as e:
                            if cfg.training.get('debug', False) and self.rank == 0:
                                print(f"[ERROR] Backward pass (FP32) failed: {e}")
                            raise
                        backward_time += time.time() - backward_start
                        
                        # DDP同步时间计算（FP32）
                        if sync_start_time is not None:
                            ddp_sync_time += time.time() - sync_start_time

                        # 优化器步骤计时（FP32）
                        if is_sync_step:
                            optimizer_start = time.time()
                            if self.use_ddp:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            else:
                                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            optimizer_time += time.time() - optimizer_start

                    # update ema
                    ema_start = time.time()
                    if cfg.training.use_ema:
                        if self.use_ddp:
                            ema.step(self.model.module)
                        else:
                            ema.step(self.model)
                    ema_time = time.time() - ema_start
                    
                    # 计算单个batch的总计算时间
                    batch_compute_time = time.time() - batch_start_time
                    total_compute_time += batch_compute_time

                    # 计算后再采样一次NVML以更贴近计算期负载
                    if (cfg.training.get('debug', False) or (batch_idx % 10 == 0)) and self.rank == 0:
                        try:
                            import pynvml
                            if not hasattr(self, '_nvml_initialized'):
                                pynvml.nvmlInit()
                                self._nvml_initialized = True
                            index = self.local_rank
                            if torch.cuda.is_available() and str(self.device).startswith('cuda'):
                                try:
                                    index = torch.cuda.current_device()
                                except Exception:
                                    index = self.local_rank
                            abs_index = index
                            try:
                                visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                                if visible:
                                    mapping = [int(x) for x in visible.split(',') if x != '']
                                    if 0 <= index < len(mapping):
                                        abs_index = mapping[index]
                            except Exception:
                                abs_index = index
                            # 同步确保上一轮kernel完成再取值
                            if torch.cuda.is_available():
                                try:
                                    torch.cuda.synchronize()
                                except Exception:
                                    pass
                            handle = pynvml.nvmlDeviceGetHandleByIndex(abs_index)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_util_samples.append(util.gpu)
                            if cfg.training.get('debug', False) and batch_idx < 5:
                                pass
                        except Exception:
                            pass

                    # logging with performance metrics
                    # 修复：记录未缩放的loss值以保持一致性
                    raw_loss_cpu = (loss * grad_acc).item()
                    if not self.use_ddp or self.rank == 0:
                        # 更新进度条，显示详细性能信息
                        avg_forward = (forward_time / batch_count) * 1000
                        avg_backward = (backward_time / batch_count) * 1000 
                        avg_transfer = (data_transfer_time / batch_count) * 1000
                        postfix_info = {
                            'loss': raw_loss_cpu,
                            'fwd_ms': f"{avg_forward:.1f}",
                            'bwd_ms': f"{avg_backward:.1f}",
                            'trans_ms': f"{avg_transfer:.1f}",
                            'gpu_util': f"{np.mean(gpu_util_samples):.0f}%" if gpu_util_samples else "N/A"
                        }
                        train_iter.set_postfix(postfix_info, refresh=False)

                    raw_gripper_loss_cpu = gripper_loss.item()
                    train_losses.append(raw_loss_cpu)
                    train_gripper_losses.append(raw_gripper_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'train_gripper_loss': raw_gripper_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch and (not self.use_ddp or self.rank == 0):
                        # log of last step is combined with validation and rollout
                        json_logger.log(step_log)
                        if wandb_run is not None:
                            wandb.log(step_log, step=self.global_step)

                    self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                train_gripper_loss = np.mean(train_gripper_losses)
                step_log['train_loss'] = train_loss
                step_log['train_gripper_loss'] = train_gripper_loss

                # 详细性能分析报告
                epoch_time = time.time() - epoch_start_time
                avg_gpu_util = np.mean(gpu_util_samples) if gpu_util_samples else 0
                
                # 计算各组件时间占比
                data_transfer_ratio = (data_transfer_time / epoch_time) * 100
                forward_ratio = (forward_time / epoch_time) * 100
                backward_ratio = (backward_time / epoch_time) * 100
                optimizer_ratio = (optimizer_time / epoch_time) * 100
                ddp_sync_ratio = (ddp_sync_time / epoch_time) * 100 if self.use_ddp else 0
                compute_ratio = (total_compute_time / epoch_time) * 100

                step_log.update({
                    'epoch_time_sec': epoch_time,
                    'total_batches': batch_count,
                    'avg_batch_time_ms': (epoch_time / batch_count) * 1000 if batch_count > 0 else 0,
                    'data_transfer_time_sec': data_transfer_time,
                    'forward_time_sec': forward_time,
                    'backward_time_sec': backward_time,
                    'optimizer_time_sec': optimizer_time,
                    'ddp_sync_time_sec': ddp_sync_time if self.use_ddp else 0,
                    'total_compute_time_sec': total_compute_time,
                    'data_transfer_ratio_%': data_transfer_ratio,
                    'forward_ratio_%': forward_ratio,
                    'backward_ratio_%': backward_ratio,
                    'optimizer_ratio_%': optimizer_ratio,
                    'ddp_sync_ratio_%': ddp_sync_ratio,
                    'compute_ratio_%': compute_ratio,
                    'avg_gpu_util_%': avg_gpu_util,
                    'effective_batch_size': cfg.dataloader.batch_size * cfg.training.gradient_accumulate_every * (self.world_size if self.use_ddp else 1)
                })

                # if not self.use_ddp or self.rank == 0:
                #     # 获取数据预处理性能统计
                #     dataset_stats = dataset.get_performance_stats() if hasattr(dataset, 'get_performance_stats') else {}
                    
                #     # 计算详细前向传播统计
                #     detailed_stats = ""
                #     if hasattr(self, 'detailed_timing') and self.detailed_timing['total_forward_times']:
                #         avg_data_transfer = np.mean(self.detailed_timing['data_transfer_times']) * 1000
                #         avg_compute = np.mean(self.detailed_timing['compute_times']) * 1000
                #         avg_total_forward = np.mean(self.detailed_timing['total_forward_times']) * 1000
                        
                #         data_transfer_pct = (avg_data_transfer / avg_total_forward) * 100
                #         compute_pct = (avg_compute / avg_total_forward) * 100
                        
                #         detailed_stats = f"\n    前向传播详细分析:"
                #         detailed_stats += f"\n      数据传输: {avg_data_transfer:.1f}ms ({data_transfer_pct:.1f}%)"
                #         detailed_stats += f"\n      模型计算: {avg_compute:.1f}ms ({compute_pct:.1f}%)"
                #         detailed_stats += f"\n      计算/传输比: {avg_compute/avg_data_transfer:.1f}x"
                        
                #         # 添加模型内部详细统计
                #         if hasattr(self.policy_model, 'encoder_timing') and self.policy_model.encoder_timing['total_times']:
                #             avg_encoder_total = np.mean(self.policy_model.encoder_timing['total_times']) * 1000
                #             avg_encoder_long = np.mean(self.policy_model.encoder_timing['long_times']) * 1000
                #             avg_encoder_mid = np.mean(self.policy_model.encoder_timing['mid_times']) * 1000
                #             avg_encoder_short = np.mean(self.policy_model.encoder_timing['short_times']) * 1000
                            
                #             encoder_pct = (avg_encoder_total / avg_compute) * 100 if avg_compute > 0 else 0
                            
                #             detailed_stats += f"\n    模型内部分析:"
                #             detailed_stats += f"\n      图像编码总计: {avg_encoder_total:.1f}ms ({encoder_pct:.1f}%)"
                #             detailed_stats += f"\n        - Long编码: {avg_encoder_long:.1f}ms"
                #             detailed_stats += f"\n        - Mid编码: {avg_encoder_mid:.1f}ms"  
                #             detailed_stats += f"\n        - Short编码: {avg_encoder_short:.1f}ms"
                            
                #         if hasattr(self.policy_model, 'diffusion_timing') and self.policy_model.diffusion_timing['times']:
                #             avg_diffusion = np.mean(self.policy_model.diffusion_timing['times']) * 1000
                #             diffusion_pct = (avg_diffusion / avg_compute) * 100 if avg_compute > 0 else 0
                #             detailed_stats += f"\n      扩散模型: {avg_diffusion:.1f}ms ({diffusion_pct:.1f}%)"
                            
                #         # 添加compute_loss内部详细统计 - 找出消失的时间
                #         if hasattr(self.policy_model, 'detailed_compute_timing'):
                #             dct = self.policy_model.detailed_compute_timing
                #             if 'total' in dct and dct['total']:
                #                 detailed_stats += f"\n    compute_loss内部详细分析:"
                #                 if 'normalize' in dct and dct['normalize']:
                #                     avg_norm = np.mean(dct['normalize']) * 1000
                #                     detailed_stats += f"\n      normalize: {avg_norm:.1f}ms"
                #                 if 'trajectory_processing' in dct and dct['trajectory_processing']:
                #                     avg_traj = np.mean(dct['trajectory_processing']) * 1000  
                #                     detailed_stats += f"\n      轨迹处理: {avg_traj:.1f}ms"
                #                 if 'obs_processing_total' in dct and dct['obs_processing_total']:
                #                     avg_obs = np.mean(dct['obs_processing_total']) * 1000
                #                     detailed_stats += f"\n      观察处理总计: {avg_obs:.1f}ms"
                #                 if 'mask_generation' in dct and dct['mask_generation']:
                #                     avg_mask = np.mean(dct['mask_generation']) * 1000
                #                     detailed_stats += f"\n      mask生成: {avg_mask:.1f}ms"
                #                 if 'noise_diffusion' in dct and dct['noise_diffusion']:
                #                     avg_noise = np.mean(dct['noise_diffusion']) * 1000
                #                     detailed_stats += f"\n      噪声和扩散: {avg_noise:.1f}ms"
                #                 if 'loss_computation' in dct and dct['loss_computation']:
                #                     avg_loss = np.mean(dct['loss_computation']) * 1000
                #                     detailed_stats += f"\n      损失计算: {avg_loss:.1f}ms"
                    
                #     print(f"\nEpoch {self.epoch} 性能分析报告:")
                #     print(f"  总时间: {epoch_time:.2f}s | 平均batch: {(epoch_time/batch_count)*1000:.1f}ms | 平均loss: {train_loss:.6f}")
                #     print(f"  数据传输: {data_transfer_time:.2f}s ({data_transfer_ratio:.1f}%)")
                #     if dataset_stats:
                #         print(f"  数据预处理: 平均 {dataset_stats['avg_postprocess_time_ms']:.1f}ms")
                #     print(f"  前向传播: {forward_time:.2f}s ({forward_ratio:.1f}%){detailed_stats}")  
                #     print(f"  反向传播: {backward_time:.2f}s ({backward_ratio:.1f}%)")
                #     print(f"  优化器步骤: {optimizer_time:.2f}s ({optimizer_ratio:.1f}%)")
                #     if self.use_ddp and ddp_sync_time > 0:
                #         print(f"  DDP同步: {ddp_sync_time:.2f}s ({ddp_sync_ratio:.1f}%)")
                #     print(f"  GPU利用率: {avg_gpu_util:.1f}% | 有效批量大小: {step_log['effective_batch_size']}")

                # ========= eval for this epoch ==========
                policy = self.policy_model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation - 优化验证策略减少同步开销
                should_validate = (self.epoch % cfg.training.val_every) == 0
                if should_validate:
                    if self.use_ddp:
                        # 多卡验证优化：仅主进程验证，其他进程继续训练
                        if self.rank == 0:
                            with torch.no_grad():
                                val_losses = list()
                                val_gripper_losses = list()
                                
                                for batch_idx, batch in enumerate(val_dataloader):
                                    batch = dataset.postprocess(batch, self.device)

                                    # 使用AMP加速验证过程
                                    if self.use_amp:
                                        with amp.autocast('cuda'):
                                            val_loss, val_gripper_loss = self.policy_model.compute_loss(batch)
                                    else:
                                        val_loss, val_gripper_loss = self.policy_model.compute_loss(batch)

                                    val_losses.append(val_loss.detach().cpu())
                                    val_gripper_losses.append(val_gripper_loss.detach().cpu())

                                if len(val_losses) > 0:
                                    val_loss = torch.mean(torch.tensor(val_losses)).item()
                                    val_gripper_loss = torch.mean(torch.tensor(val_gripper_losses)).item()
                                    step_log['val_loss'] = val_loss
                                    step_log['val_gripper_loss'] = val_gripper_loss
                        # 不在这里同步，让其他GPU继续工作
                    else:
                        # 单卡验证（保持原逻辑）
                        with torch.no_grad():
                            val_losses = list()
                            val_gripper_losses = list()
                            val_iter = tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec)

                            for batch_idx, batch in enumerate(val_iter):
                                batch = dataset.postprocess(batch, self.device)

                                if self.use_amp:
                                    with amp.autocast('cuda'):
                                        val_loss, val_gripper_loss = self.policy_model.compute_loss(batch)
                                else:
                                    val_loss, val_gripper_loss = self.policy_model.compute_loss(batch)

                                val_losses.append(val_loss.detach().cpu())
                                val_gripper_losses.append(val_gripper_loss.detach().cpu())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                val_gripper_loss = torch.mean(torch.tensor(val_gripper_losses)).item()
                                step_log['val_loss'] = val_loss
                                step_log['val_gripper_loss'] = val_gripper_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        # 确保batch数据在正确设备上
                        batch = dataset.postprocess(batch, self.device)
                        obs_dict = batch['obs']
                        gt_action = batch['action'][:,self.n_obs_steps:self.n_obs_steps+8]
                        # print("gt_action.shape: ", gt_action.shape)

                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        # print("pred_action.shape: ", pred_action.shape)

                        # 确保gt_action和pred_action在同一设备上
                        gt_action = gt_action.to(pred_action.device, non_blocking=True)
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint 优化 - 进一步减少同步频率
                should_checkpoint = ((self.epoch + 1) % cfg.training.checkpoint_every) == 0
                if should_checkpoint and (not self.use_ddp or self.rank == 0):
                    # 仅主进程保存检查点，异步进行不阻塞其他GPU
                    save_name = pathlib.Path(self.cfg.task.dataset.zarr_path).stem
                    # 添加 short_interval, mid_interval, long_interval 参数到文件夹名称
                    short_interval = cfg.get('short_interval', 1)
                    mid_interval = cfg.get('mid_interval', 2) 
                    long_interval = cfg.get('long_interval', 6)
                    checkpoint_dir = f'checkpoints/{save_name}_{short_interval}_{mid_interval}_{long_interval}_{seed}'
                    self.save_checkpoint(f'{checkpoint_dir}/{self.epoch + 1}.ckpt')
                    # 同时保存为latest.ckpt以支持resume功能
                    self.save_checkpoint(tag='latest')
                
                # 延迟同步 - 仅在必要时才进行barrier同步（每10个checkpoint一次）
                if self.use_ddp and should_checkpoint and ((self.epoch + 1) % (cfg.training.checkpoint_every * 10) == 0):
                    dist.barrier()

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch (仅主进程记录)
                if not self.use_ddp or self.rank == 0:
                    # log of last step is combined with validation and rollout
                    json_logger.log(step_log)
                    if wandb_run is not None:
                        wandb.log(step_log, step=self.global_step)

                self.global_step += 1
                self.epoch += 1

        # finish wandb run at the end of training (仅主进程)
        if wandb_run is not None:
            wandb_run.finish()


class BatchSampler:
    def __init__(self, data_size: int, batch_size: int, shuffle: bool = False, seed: int = 0, drop_last: bool = True):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch

def create_dataloader(dataset, *, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int = 2, seed: int = 0):
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)
    def collate(x):
        assert len(x) == 1
        return x[0]
    dataloader = DataLoader(dataset, collate_fn=collate, sampler=batch_sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor if num_workers > 0 else 2)
    return dataloader

def create_dataloader_ddp(dataset, *, sampler, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int = 2, seed: int = 0):
    def collate(batch):
        # DDP版本：正确处理批量数据
        # batch是一个包含batch_size个样本的列表
        if len(batch) == 1:
            return batch[0]

        # 处理多个样本的情况，需要将它们合并成一个批次
        def collate_recursive(items):
            """递归处理嵌套的字典和张量"""
            if isinstance(items[0], torch.Tensor):
                # 在CPU上堆叠，之后由postprocess传输到GPU
                return torch.stack(items)
            elif isinstance(items[0], dict):
                collated_dict = {}
                for key in items[0].keys():
                    collated_dict[key] = collate_recursive([item[key] for item in items])
                return collated_dict
            elif isinstance(items[0], (list, tuple)):
                return [item for item in items]
            else:
                return [item for item in items]

        return collate_recursive(batch)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor if num_workers > 0 else 2)
    return dataloader

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
