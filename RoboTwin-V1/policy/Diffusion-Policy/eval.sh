DEBUG=False

task_name=${1}
head_camera_type=${2}
expert_data_num=${3}
checkpoint_num=${4}
seed=${5}
gpu_id=${6}
ensemble=${7}
no_hierarchical_cond=${8:-false}  # 新增参数，默认为false

echo -e "\033[33mensemble: ${ensemble}\033[0m"
echo -e "\033[33mno_hierarchical_cond: ${no_hierarchical_cond}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..
if [ -z "$ensemble" ]; then
    python ./script/eval_policy_dp.py "$task_name" "$head_camera_type" "$expert_data_num" "$checkpoint_num" "$seed" --no_hierarchical_cond "$no_hierarchical_cond"
else
    python ./script/eval_policy_dp.py "$task_name" "$head_camera_type" "$expert_data_num" "$checkpoint_num" "$seed" --ensemble --no_hierarchical_cond "$no_hierarchical_cond"
fi