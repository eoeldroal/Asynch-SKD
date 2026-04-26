#!/usr/bin/env bash
# GRPO training with VeOmniEngine using locally copied APSKD Qwen3.5-9B model/data.
#
# Default assets:
#   - model: $HOME/model/Qwen3.5-9B
#   - train: $HOME/data/apskd_math/train.parquet
#   - validation: $HOME/data/apskd_math/test.parquet and $HOME/data/apskd_math/aime-2024.parquet
#
# Environment:
#   - transformers==5.3.0
#   - sglang==0.5.9
#   - flash-linear-attention==0.4.1
#   - veomni==0.1.9a1

set -xeuo pipefail

# /tmp is mounted noexec on this host, so disable SGLang's NUMA bind v2
# wrapper that creates executable scripts under /tmp.
export SGLANG_NUMA_BIND_V2=${SGLANG_NUMA_BIND_V2:-0}
export SGLANG_ENABLE_TORCH_INFERENCE_MODE=${SGLANG_ENABLE_TORCH_INFERENCE_MODE:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

data_path=${data_path:-$HOME/data/apskd_math}
model_path=${model_path:-$HOME/model/Qwen3.5-9B}
reward_function_path=${reward_function_path:-${REPO_ROOT}/examples/grpo_trainer/reward_fn_math_verify.py}
usp_size=${usp_size:-1}
nnodes=${nnodes:-1}
n_gpus_per_node=${n_gpus_per_node:-8}

backend=fsdp2
model_engine=veomni
project_name=${project_name:-'verl_grpo_qwen3_5_9b_apskd_math_veomni'}
exp_name=${exp_name:-'qwen3_5_9b_veomni_apskd_math_sp1'}


# ===================================== Algorithm =====================================
adv_estimator=grpo
loss_mode=gspo

# reference policy
use_kl_in_reward=False
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001

clip_ratio_low=3e-4
clip_ratio_high=4e-4

actor_lr=1e-6
critic_lr=2e-6
gae_gamma=1.0
gae_lam=0.95
critic_warmup=0

# ===================================== Data/Model =====================================
train_files=${train_files:-$data_path/train.parquet}
test_files=${test_files:-"['$data_path/test.parquet','$data_path/aime-2024.parquet']"}

actor_model_path=$model_path

max_prompt_length=${max_prompt_length:-$((1024 * 1))}
max_response_length=${max_response_length:-$((1024 * 8))}

train_batch_size=${train_batch_size:-128}
ppo_mini_batch_size=${ppo_mini_batch_size:-32}
n_resp_per_prompt=${n_resp_per_prompt:-8}
n_resp_per_prompt_val=${n_resp_per_prompt_val:-1}

use_remove_padding=True
use_dynamic_bsz=${use_dynamic_bsz:-True}

# ===================================== Training =====================================
# In dynamic batch mode, token budgets drive micro-batch packing. The sample
# micro-batch size below is kept for compatibility and non-dynamic overrides.
actor_max_token_len_per_gpu=${actor_max_token_len_per_gpu:-32768}
log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu:-49152}
ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu:-1}

# VeOmni config
ACTOR_VEOMNI_CONFIG="
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.veomni.enable_full_shard=True \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=$usp_size \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2"

# Actor model config
ACTOR_CONFIG="
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.model.path=$actor_model_path \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu"

CONFIG_NAME=ppo_trainer
ACTOR_CONFIG="$ACTOR_CONFIG $ACTOR_VEOMNI_CONFIG"
CRITIC_CONFIG=""

# ===================================== Inference =====================================
rollout_name=${rollout_name:-sglang}
infer_tp=${infer_tp:-1}
infer_dp=${infer_dp:-1}
infer_ep=${infer_ep:-1}
gpu_memory_utilization=${gpu_memory_utilization:-0.6}

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.data_parallel_size=$infer_dp \
    actor_rollout_ref.rollout.expert_parallel_size=$infer_ep \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu"

python -m verl.trainer.main_ppo \
    --config-path=./config \
    --config-name=$CONFIG_NAME \
    model_engine=$model_engine \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    algorithm.gamma=$gae_gamma \
    algorithm.lam=$gae_lam \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    reward.custom_reward_function.path="$reward_function_path" \
    reward.custom_reward_function.name=compute_score_math_verify \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.val_before_train=False \
    trainer.log_val_generations=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=500 \
    $ACTOR_CONFIG \
    $CRITIC_CONFIG \
    $ROLLOUT_CONFIG \
    "$@"
