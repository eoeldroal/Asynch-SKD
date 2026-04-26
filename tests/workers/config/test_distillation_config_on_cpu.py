from omegaconf import OmegaConf

from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationConfig,
    DistillationLossConfig,
    DistillationTeacherModelConfig,
    RolloutConfig,
    SkdConfig,
)


def _make_distillation_config(*, teacher_system_prompt_path: str | None) -> DistillationConfig:
    return DistillationConfig(
        enabled=True,
        n_gpus_per_node=1,
        nnodes=1,
        teacher_models={
            "teacher_model": DistillationTeacherModelConfig(
                model_path="Qwen/Qwen3-8B",
                inference=RolloutConfig(
                    name="sglang",
                    tensor_model_parallel_size=1,
                    prompt_length=512,
                    response_length=8192,
                    max_model_len=8705,
                    max_num_batched_tokens=8705,
                ),
            )
        },
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk",
            topk=128,
            use_policy_gradient=False,
        ),
        skd=SkdConfig(
            chunk_size=128,
            verify_top_k=3,
            max_chunks_per_sample=256,
            teacher_system_prompt_path=teacher_system_prompt_path,
        ),
    )


def test_skd_config_accepts_fields_from_hydra_config():
    cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.SkdConfig",
            "chunk_size": 128,
            "verify_top_k": 3,
            "max_chunks_per_sample": 256,
            "teacher_system_prompt_path": "/tmp/teacher_prompt.txt",
        }
    )

    skd_cfg = omega_conf_to_dataclass(cfg, SkdConfig)

    assert skd_cfg.chunk_size == 128
    assert skd_cfg.verify_top_k == 3
    assert skd_cfg.max_chunks_per_sample == 256
    assert skd_cfg.teacher_system_prompt_path == "/tmp/teacher_prompt.txt"


def test_distillation_teacher_budget_stays_unchanged_without_teacher_prompt():
    distill_cfg = _make_distillation_config(teacher_system_prompt_path=None)

    teacher = distill_cfg.teacher_models["default"]
    assert teacher.inference.max_model_len == 8705
    assert teacher.inference.max_num_batched_tokens == 8705
    assert teacher.inference.prompt_length == 8704
    assert teacher.inference.response_length == 1


def test_distillation_teacher_budget_adds_margin_when_teacher_prompt_enabled():
    distill_cfg = _make_distillation_config(teacher_system_prompt_path="/tmp/teacher_prompt.txt")

    teacher = distill_cfg.teacher_models["default"]
    assert teacher.inference.max_model_len == 9217
    assert teacher.inference.max_num_batched_tokens == 9217
    assert teacher.inference.prompt_length == 9216
    assert teacher.inference.response_length == 1


def test_distillation_config_accepts_forward_kl_topk_impl_flag():
    cfg = OmegaConf.create(
        {
            "_target_": "verl.workers.config.DistillationLossConfig",
            "loss_mode": "forward_kl_topk",
            "topk": 32,
            "use_policy_gradient": False,
            "forward_kl_topk_impl": "logsumexp_gather",
        }
    )

    loss_cfg = omega_conf_to_dataclass(cfg, DistillationLossConfig)

    assert loss_cfg.forward_kl_topk_impl == "logsumexp_gather"
