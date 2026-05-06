GPUS=(0)

# transformer_engine locates libnvrtc via CUDA_HOME before falling back to ldconfig.
# With uv-managed CUDA packages there is no system CUDA install, so point CUDA_HOME
# at the venv's nvidia/cuda_nvrtc package so the shared library is found at import time.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VENV="${SCRIPT_DIR}/../../model/.venv"
export CUDA_HOME="${MODEL_VENV}/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

fifo="/tmp/gpuq.$$"
mkfifo "$fifo"
exec 3<>"$fifo"
rm -f "$fifo"

for i in $(seq 0 $(( 1 * ${#GPUS[@]} - 1 ))); do
  g=$((i % ${#GPUS[@]}))
  echo "$g" >&3
done

launch() {
  local gpu
  read -u 3 gpu || exit 1
  (
    export CUDA_VISIBLE_DEVICES=${gpu}
    export SAPIEN_RENDER_CUDA_ORDINAL=0
    export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_device_select
    export PYTHONPATH=LIBERO
    export TOKENIZERS_PARALLELISM=false
    "$@"
    rc=$?
    echo "$gpu" >&3
    exit $rc
  ) &
}

checkpoint_dir=/home/ethrc/Desktop/mimic-video/model/checkpoints

declare -A goal_half=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_goal_half_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_goal_half_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128_iter_000050022.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_goal_half.json
  [suite]=libero_goal
)
declare -A goal_tenth=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_goal_tenth_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_goal_tenth_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128_iter_000040014.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_goal_tenth.json
  [suite]=libero_goal
)
declare -A goal_one=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_goal_one_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_goal_one_v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_goal_one.json
  [suite]=libero_goal
)

declare -A object_full=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_object_full_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_object_full_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128_iter_000050274.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_object_full.json
  [suite]=libero_object
)
declare -A object_half=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_object_half_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_object_half_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128_iter_000030090.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_object_half.json
  [suite]=libero_object
)
declare -A object_tenth=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_object_tenth_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_object_tenth_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128_iter_000039984.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_object_tenth.json
  [suite]=libero_object
)
declare -A object_one=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_object_one_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_object_one_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused_lr1.000e-04_layer20_bsz128_iter_000029997.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_object_one.json
  [suite]=libero_object
)

declare -A spatial_full=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_spatial_full_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_spatial_full_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000051212.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_spatial_full.json
  [suite]=libero_spatial
)
declare -A spatial_tenth=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_spatial_tenth_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_spatial_tenth_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000030012.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_spatial_tenth.json
  [suite]=libero_spatial
)
declare -A spatial_one=(
  [img_h]=5
  [lowdim_h]=1
  [experiment_name]=w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128
  [action_model]=${checkpoint_dir}/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt
  [video_model]=${checkpoint_dir}/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt
  [stats]=${checkpoint_dir}/dataset_statistics/libero_spatial_one.json
  [suite]=libero_spatial
)

models=(object_full)

execute_actions=(5)
stop_steps=(0 20 35)

for execute_steps in "${execute_actions[@]}"; do
  for stop_after in "${stop_steps[@]}"; do
    for model_name in "${models[@]}"; do
      declare -n MODEL="${model_name}"
      echo "=== Evaluating ${model_name} | stop=${stop_after} | execute=${execute_steps} | suite=${MODEL[suite]} ===";

      launch python run.py \
        --vam_experiment_name "${MODEL[experiment_name]}" \
        --vam_video_model_path "${MODEL[video_model]}" \
        --vam_action_model_path "${MODEL[action_model]}" \
        --vam_dataset_statistics_path "${MODEL[stats]}" \
        --vam_img_horizon "${MODEL[img_h]}" \
        --vam_lowdim_horizon "${MODEL[lowdim_h]}" \
        --vam_stop_video_denoising_step "${stop_after}" \
        --vam_num_execute_actions "${execute_steps}" \
        --task_suite_name "${MODEL[suite]}" \
        --eval_rank 0 \
        --eval_world_size 5;
      launch python run.py \
        --vam_experiment_name "${MODEL[experiment_name]}" \
        --vam_video_model_path "${MODEL[video_model]}" \
        --vam_action_model_path "${MODEL[action_model]}" \
        --vam_dataset_statistics_path "${MODEL[stats]}" \
        --vam_img_horizon "${MODEL[img_h]}" \
        --vam_lowdim_horizon "${MODEL[lowdim_h]}" \
        --vam_stop_video_denoising_step "${stop_after}" \
        --vam_num_execute_actions "${execute_steps}" \
        --task_suite_name "${MODEL[suite]}" \
        --eval_rank 1 \
        --eval_world_size 5;
      launch python run.py \
        --vam_experiment_name "${MODEL[experiment_name]}" \
        --vam_video_model_path "${MODEL[video_model]}" \
        --vam_action_model_path "${MODEL[action_model]}" \
        --vam_dataset_statistics_path "${MODEL[stats]}" \
        --vam_img_horizon "${MODEL[img_h]}" \
        --vam_lowdim_horizon "${MODEL[lowdim_h]}" \
        --vam_stop_video_denoising_step "${stop_after}" \
        --vam_num_execute_actions "${execute_steps}" \
        --task_suite_name "${MODEL[suite]}" \
        --eval_rank 2 \
        --eval_world_size 5;
      launch python run.py \
        --vam_experiment_name "${MODEL[experiment_name]}" \
        --vam_video_model_path "${MODEL[video_model]}" \
        --vam_action_model_path "${MODEL[action_model]}" \
        --vam_dataset_statistics_path "${MODEL[stats]}" \
        --vam_img_horizon "${MODEL[img_h]}" \
        --vam_lowdim_horizon "${MODEL[lowdim_h]}" \
        --vam_stop_video_denoising_step "${stop_after}" \
        --vam_num_execute_actions "${execute_steps}" \
        --task_suite_name "${MODEL[suite]}" \
        --eval_rank 3 \
        --eval_world_size 5;
      launch python run.py \
        --vam_experiment_name "${MODEL[experiment_name]}" \
        --vam_video_model_path "${MODEL[video_model]}" \
        --vam_action_model_path "${MODEL[action_model]}" \
        --vam_dataset_statistics_path "${MODEL[stats]}" \
        --vam_img_horizon "${MODEL[img_h]}" \
        --vam_lowdim_horizon "${MODEL[lowdim_h]}" \
        --vam_stop_video_denoising_step "${stop_after}" \
        --vam_num_execute_actions "${execute_steps}" \
        --task_suite_name "${MODEL[suite]}" \
        --eval_rank 4 \
        --eval_world_size 5;
    done
  done
done
