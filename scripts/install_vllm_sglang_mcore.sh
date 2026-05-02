#!/bin/bash
set -euo pipefail

USE_MEGATRON=${USE_MEGATRON:-1}
USE_SGLANG=${USE_SGLANG:-1}
USE_VLLM=${USE_VLLM:-1}
VLLM_VERSION=${VLLM_VERSION:-0.14.1}
FLASH_ATTN_WHEEL_URL=${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}
FLASH_ATTN_WHEEL_FILE=${FLASH_ATTN_WHEEL_FILE:-flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}

export MAX_JOBS=32

# Qwen3.5 / VeOmni baseline used by examples/grpo_trainer/run_qwen3_5-35b-a3b_veomni.sh:
#   - transformers==5.3.0
#   - sglang==0.5.10
#   - vllm==0.14.1 (--no-deps when co-installed with SGLang)
#   - flash-attn-4==4.0.0b11
#   - nvidia-cutlass-dsl==4.4.2
#   - flash-linear-attention==0.4.2
#   - veomni==0.1.9a1

echo "0. install uv"
pip install uv

echo "1. install inference frameworks and pytorch they need"
if [ "$USE_SGLANG" -eq 1 ]; then
    uv pip install --prerelease allow "sglang[all]==0.5.10"
    uv pip install torch-memory-saver
    # Keep the FA4 vision-attention backend reproducible for SGLang multimodal
    # runs. SGLang depends on FA4 transitively, but the WebGym/Qwen3.5 scripts
    # explicitly set mm_attention_backend=fa4, so pin the beta FA4 package and
    # its CUTLASS DSL provider here instead of relying on resolver drift.
    # Pin FA2 to a torch 2.9-compatible wheel, then reinstall FA4 after it.
    # FA2 and FA4 both write into the flash_attn namespace; FA4 must be last so
    # flash_attn.cute comes from flash-attn-4, not the FA2 wheel.
    wget -nv -O "${FLASH_ATTN_WHEEL_FILE}" "${FLASH_ATTN_WHEEL_URL}"
    uv pip install "${FLASH_ATTN_WHEEL_FILE}"
    uv pip install --prerelease allow --reinstall-package flash-attn-4 "flash-attn-4==4.0.0b11" "nvidia-cutlass-dsl==4.4.2"
    python - <<'PY'
import flash_attn
import flash_attn.flash_attn_interface
import flash_attn.cute
import cutlass
import sglang.jit_kernel.flash_attention_v4
print("FA2/FA4 import smoke test passed")
PY
fi
if [ "$USE_VLLM" -eq 1 ]; then
    if [ "$USE_SGLANG" -eq 1 ]; then
        # Keep the SGLang 0.5.10 dependency set authoritative. vLLM is useful
        # across the repo, but installing its dependencies in this shared env can
        # downgrade torch/flashinfer/xgrammar and break SGLang. Pick a torch
        # 2.9.1-era vLLM and install the package itself only.
        uv pip install --no-deps "vllm==${VLLM_VERSION}"
    else
        uv pip install "vllm==${VLLM_VERSION}"
    fi
fi

echo "2. install basic packages"
uv pip install "transformers[hf_xet]==5.3.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler cachetools \
    pytest py-spy pre-commit ruff tensorboard flash-linear-attention==0.4.2 veomni==0.1.9a1

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"


echo "3. install FlashAttention and FlashInfer"
if [ "$USE_SGLANG" -eq 1 ]; then
    echo "SGLang path uses flash-attn==2.8.3, flash-attn-4==4.0.0b11, and SGLang's pinned flashinfer."
else
    echo "No extra FlashAttention/FlashInfer pin is applied outside the SGLang path."
fi


if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "4. install TransformerEngine and Megatron"
    echo "Notice that TransformerEngine installation can take very long time, please be patient"
    uv pip install "onnxscript==0.3.1"
    NVTE_FRAMEWORK=pytorch uv pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
    uv pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
fi


echo "5. May need to fix opencv"
uv pip install --no-deps "opencv-python==4.10.0.84" "numpy<2.0.0"
uv pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"


if [ "$USE_MEGATRON" -eq 1 ]; then
    echo "6. Install cudnn python package (avoid being overridden)"
    uv pip install nvidia-cudnn-cu12==9.10.2.21
fi

echo "7. Pin CuPy for numpy<2.0 compatibility"
# cupy-cuda12x 14.x requires numpy>=2.0, while this environment pins
# numpy<2.0 for verl and related training dependencies. Keep numpy fixed and
# install the newest CuPy 13.x wheel that supports Python 3.12 and CUDA 12.x.
uv pip install --no-deps "cupy-cuda12x==13.6.0" "fastrlock==0.8.3"

echo "Successfully installed all packages"
