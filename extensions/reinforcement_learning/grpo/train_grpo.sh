#!/usr/bin/env bash
set -euo pipefail

# Optional IFlow Agent GRPO training entrypoint. This script never runs as part
# of the normal Agent runtime and requires an explicit user invocation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PACKAGE_PARENT="$(cd "${PROJECT_ROOT}/.." && pwd)"

: "${VERL_DIR:?Set VERL_DIR to a local verl checkout}"
: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the model or checkpoint directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a verl-compatible parquet file}"

VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-iflow_main_agent_grpo_${TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}"
REWARD_FILE="${SCRIPT_DIR}/reward.py"

for required_path in "${VERL_DIR}" "${BASE_MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[iflow-grpo] required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-12}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-24576}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"
ROLLOUT_GROUP_SIZE="${ROLLOUT_GROUP_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-50}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\"]}"

export PYTHONPATH="${PACKAGE_PARENT}:${VERL_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VERL_FILE_LOGGER_PATH="${OUTPUT_DIR}/metrics.jsonl"

cd "${VERL_DIR}"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_GROUP_SIZE}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKENS_PER_GPU}" \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger="${TRAINER_LOGGERS}" \
    trainer.project_name=iflow_agent_grpo \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.save_freq="${SAVE_FREQUENCY}" \
    trainer.test_freq=-1 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
    "$@"
