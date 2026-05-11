#!/usr/bin/env bash
set -xeuo pipefail

cd /home/sogang_nlpy/verl

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ROLLOUT_DATA_DIR=/home/sogang_nlpy/verl/logs/rollout_data/qwen35_webgym_fully_async_tool_veomni_${RUN_TIMESTAMP}
# The referenced parquet files are the fully async RL copies generated from:
#   /home/sogang_nlpy/goonco/surfgym/tasks/tasks_subset.json
# with localhost tasks included, via:
#   /home/sogang_nlpy/verl/WebOSWorld/webgym_rl/create_webgym_rl_dataset.py
WEBGYM_ASYNC_RL_DATASET_DIR=/home/sogang_nlpy/verl/data/webgym_rl
WEBGYM_TOOL_CONFIG_PATH=/home/sogang_nlpy/verl/WebOSWorld/config/tool_config/webgym_rl_tool_config_bundled.yaml
WEBGYM_SYSTEM_PROMPT_PATH="${WEBGYM_SYSTEM_PROMPT_PATH:-/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/system_prompt_webgym_rl.txt}"

SGLANG_NUMA_BIND_V2=0 \
SGLANG_ENABLE_TORCH_INFERENCE_MODE=1 \
HYDRA_FULL_ERROR=1 \
WEB_OSGYM_UNIT_TRACE=1 \
WEB_OSGYM_TOOL_TRACE_DIR="${ROLLOUT_DATA_DIR}" \
python -m verl.experimental.fully_async_policy.fully_async_main \
    model_engine=veomni \
    "data.train_files=['${WEBGYM_ASYNC_RL_DATASET_DIR}/train.parquet']" \
    "data.val_files=['${WEBGYM_ASYNC_RL_DATASET_DIR}/val.parquet']" \
    data.prompt_key=prompt \
    data.truncation=error \
    data.max_prompt_length=3072 \
    data.max_response_length=56000 \
    data.filter_overlong_prompts=False \
    data.filter_overlong_prompts_workers=64 \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.shuffle=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=/home/sogang_nlpy/verl/checkpoints/verl_async_skd_qwen35_webgym/qwen35_9b_to_27b_async_skd_webgym_counter_tool/global_step_40/actor/huggingface \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=cispo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=10 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_single_actor_mini_batch=True \
    actor_rollout_ref.actor.veomni.param_offload=False \
    actor_rollout_ref.actor.veomni.optimizer_offload=False \
    actor_rollout_ref.actor.veomni.enable_full_shard=True \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=4 \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=38401 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.max_model_len=76800 \
    actor_rollout_ref.rollout.max_num_batched_tokens=76800 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=fa4 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.grammar_backend=xgrammar \
    actor_rollout_ref.rollout.skip_tokenizer_init=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=38401 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=50 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=5 \
    actor_rollout_ref.rollout.multi_turn.web_osgym_window_enable=True \
    actor_rollout_ref.rollout.multi_turn.web_osgym_window_history_n=5 \
    actor_rollout_ref.rollout.multi_turn.web_osgym_window_max_images_per_sample=6 \
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${WEBGYM_TOOL_CONFIG_PATH}" \
    "actor_rollout_ref.rollout.multi_turn.system_prompt_path=${WEBGYM_SYSTEM_PROMPT_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    +actor_rollout_ref.rollout.custom.enable_qwen3_coder_structured_output=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=web_tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.max_concurrent_samples_per_gpu=16 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    reward.custom_reward_function.path=/home/sogang_nlpy/verl/WebOSWorld/webgym_rl/reward_fn_webgym_rl.py \
    reward.custom_reward_function.name=compute_score_webgym_rl \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=verl_fully_async_qwen35_webgym_tool_veomni \
    trainer.experiment_name=qwen35_9b_fully_async_webgym_tool \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.resume_mode=auto \
    trainer.default_local_dir=/home/sogang_nlpy/verl/checkpoints/verl_fully_async_qwen35_webgym_tool_veomni/qwen35_9b_fully_async_webgym_tool \
    "trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}" \
    +ray_kwargs.ray_init.object_store_memory=320000000000 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=4 \
    rollout.total_rollout_steps=51200 \
    trainer.total_epochs=100 \
    async_training.staleness_threshold=1.0 \
    async_training.trigger_parameter_sync_step=2 \
    async_training.require_batches=1 \
    async_training.partial_rollout=True \
    async_training.use_trainer_do_validate=False \
    "$@"