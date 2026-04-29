#!/usr/bin/env bash
set -xeuo pipefail

cd /home/sogang_nlpy/verl

# Fully async RL Web/OSGym smoke run.
#
# This path intentionally does not enable SKD/distillation. It uses the fully
# async policy trainer and routes each Web/OSGym trajectory through
# web_tool_agent, which keeps one environment session alive across all actions
# and fetches the environment reward at rollout termination.

SGLANG_NUMA_BIND_V2=0 \
SGLANG_ENABLE_TORCH_INFERENCE_MODE=1 \
HYDRA_FULL_ERROR=1 \
python -m verl.experimental.fully_async_policy.fully_async_main \
    model_engine=veomni \
    "data.train_files=['/home/sogang_nlpy/verl/data/mock_web_osgym_fully_async_rl/train.parquet']" \
    "data.val_files=['/home/sogang_nlpy/verl/data/mock_web_osgym_fully_async_rl/val.parquet']" \
    data.prompt_key=prompt \
    data.truncation=error \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.shuffle=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=/home/sogang_nlpy/verl/models/Qwen3.5-9B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=True \
    actor_rollout_ref.actor.veomni.enable_full_shard=True \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=1 \
    actor_rollout_ref.actor.veomni.expert_parallel_size=1 \
    actor_rollout_ref.actor.veomni.attn_implementation=flash_attention_2 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.max_model_len=10241 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=triton \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_attention_backend=triton_attn \
    +actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=30720 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/home/sogang_nlpy/verl/examples/sglang_multiturn/config/tool_config/web_osgym_tool_config_webgym_rl.yaml \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.agent.default_agent_loop=web_tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.agent.max_concurrent_samples_per_gpu=16 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    reward.custom_reward_function.path=/home/sogang_nlpy/verl/WebOSWorld/mock_server/reward_fn_mock_web_osgym.py \
    reward.custom_reward_function.name=compute_score_mock_web_osgym \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=verl_fully_async_qwen35_web_mock_tool_fsdp \
    trainer.experiment_name=qwen35_9b_fully_async_web_mock_tool \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir=/home/sogang_nlpy/verl/checkpoints/verl_fully_async_qwen35_web_mock_tool_fsdp/qwen35_9b_fully_async_web_mock_tool \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=4 \
    rollout.total_rollout_steps=51200 \
    trainer.total_epochs=10 \
    async_training.staleness_threshold=0.1 \
    async_training.trigger_parameter_sync_step=2 \
    async_training.require_batches=1 \
    async_training.partial_rollout=True \
    async_training.use_trainer_do_validate=False \
    "$@"
