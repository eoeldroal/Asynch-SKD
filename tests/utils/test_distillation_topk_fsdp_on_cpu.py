import torch
import torch.nn.functional as F

from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk
from verl.workers.config import DistillationConfig, DistillationLossConfig, DistillationTeacherModelConfig, RolloutConfig


def _to_nested(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nested.as_nested_tensor([tensor[i] for i in range(tensor.shape[0])], layout=torch.jagged)


def _make_cfg(impl: str) -> DistillationConfig:
    return DistillationConfig(
        enabled=True,
        teacher_models={
            "teacher_model": DistillationTeacherModelConfig(
                model_path="dummy",
                key="teacher_model",
                num_replicas=1,
                inference=RolloutConfig(
                    name="sglang",
                    tensor_model_parallel_size=1,
                    prompt_length=16,
                    response_length=16,
                    max_model_len=64,
                    max_num_batched_tokens=64,
                ),
            )
        },
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk",
            topk=32,
            use_policy_gradient=False,
            forward_kl_topk_impl=impl,
            log_prob_min_clamp=-10.0,
        ),
    )


def test_fsdp_forward_kl_topk_logsumexp_gather_matches_default_outputs_and_grads():
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 7
    vocab_size = 53
    topk = 32

    teacher_full_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_full_logps = F.log_softmax(teacher_full_logits, dim=-1)
    teacher_topk_logps, teacher_topk_ids = torch.topk(teacher_full_logps, k=topk, dim=-1)
    teacher_topk_logps = _to_nested(teacher_topk_logps)
    teacher_topk_ids = _to_nested(teacher_topk_ids)

    student_logits_default = torch.randn(1, batch_size * seq_len, vocab_size, requires_grad=True)
    student_logits_new = student_logits_default.detach().clone().requires_grad_(True)

    default_out = compute_forward_kl_topk(
        student_logits=student_logits_default,
        teacher_topk_log_probs=teacher_topk_logps,
        teacher_topk_ids=teacher_topk_ids,
        config=_make_cfg("log_softmax"),
        data_format="thd",
    )
    new_out = compute_forward_kl_topk(
        student_logits=student_logits_new,
        teacher_topk_log_probs=teacher_topk_logps,
        teacher_topk_ids=teacher_topk_ids,
        config=_make_cfg("logsumexp_gather"),
        data_format="thd",
    )

    for key in ("distillation_losses", "student_mass", "teacher_mass"):
        torch.testing.assert_close(default_out[key], new_out[key], atol=1e-5, rtol=1e-5)

    default_out["distillation_losses"].sum().backward()
    new_out["distillation_losses"].sum().backward()
    torch.testing.assert_close(student_logits_default.grad, student_logits_new.grad, atol=1e-5, rtol=1e-5)
