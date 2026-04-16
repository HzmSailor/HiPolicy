# 
task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
seed=${4}
gpu_id_list=${5}
no_hierarchical_cond=${6:-false}  # 新增参数，默认为false

DEBUG=False
save_ckpt=True

alg_name=robot_dp
# task choices: See TASK.md
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

# 解析GPU列表并计算GPU数量
IFS=',' read -ra GPU_ARRAY <<< "$gpu_id_list"
num_gpus=${#GPU_ARRAY[@]}
master_gpu=${GPU_ARRAY[0]}

echo -e "\033[33mgpu id list (to use): ${gpu_id_list}\033[0m"
echo -e "\033[33mnum gpus: ${num_gpus}\033[0m"
echo -e "\033[33mmaster gpu: ${master_gpu}\033[0m"
echo -e "\033[33mno_hierarchical_cond: ${no_hierarchical_cond}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

if [ ! -d "/DATA/disk0/data_robotwin_new/processed_data/${task_name}_${head_camera_type}_${expert_data_num}.zarr" ]; then
    echo "zarr does not exist, run pkl2zarr_dp.py"
    cd ../..
    expert_data_num_minus_one=$((expert_data_num - 1))
    if [ ! -d "/DATA/disk0/data_robotwin_new/${task_name}_${head_camera_type}_pkl/episode${expert_data_num_minus_one}" ]; then
        echo "error: expert data does not exist"
        exit 1
    else
        python script/pkl2zarr_dp.py ${task_name} ${head_camera_type} ${expert_data_num}
        cd policy/Diffusion-Policy
    fi
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id_list}

# 根据GPU数量选择训练方式
if [ ${num_gpus} -eq 1 ]; then
    # 单卡训练
    echo -e "\033[33m单卡训练模式\033[0m"
    python train.py --config-name=${config_name}.yaml \
                                task.name=${task_name} \
                                task.dataset.zarr_path="/DATA/disk0/data_robotwin_new/processed_data/${task_name}_${head_camera_type}_${expert_data_num}.zarr" \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device="cuda:0" \
                                training.max_train_steps=null \
                                exp_name=${exp_name} \
                                logging.mode=${wandb_mode} \
                                head_camera_type=${head_camera_type} \
                                expert_data_num=${expert_data_num} \
                                policy.no_hierarchical_cond=${no_hierarchical_cond}
else
    # 多卡训练
    echo -e "\033[33m多卡训练模式\033[0m"
    # 使用随机端口避免端口冲突，并确保工作目录正确
    MASTER_PORT=$((29500 + RANDOM % 1000))
    cd policy/Diffusion-Policy
    torchrun --nproc_per_node=${num_gpus} --master_port=${MASTER_PORT} \
        train.py --config-name=${config_name}.yaml \
                                task.name=${task_name} \
                                task.dataset.zarr_path="/DATA/disk0/data_robotwin_new/processed_data/${task_name}_${head_camera_type}_${expert_data_num}.zarr" \
                                training.debug=$DEBUG \
                                training.seed=${seed} \
                                training.device="cuda" \
                                training.use_ddp=True \
                                training.world_size=${num_gpus} \
                                exp_name=${exp_name} \
                                logging.mode=${wandb_mode} \
                                head_camera_type=${head_camera_type} \
                                expert_data_num=${expert_data_num} \
                                policy.no_hierarchical_cond=${no_hierarchical_cond}
fi