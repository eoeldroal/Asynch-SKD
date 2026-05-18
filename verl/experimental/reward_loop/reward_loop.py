# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import os
import threading

import aiohttp
import numpy as np
import ray
import torch
from omegaconf import DictConfig, open_dict
from PIL import Image
from ray.actor import ActorHandle
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_tokenizer
from verl.utils.experimental.reward_utils import pil_image_to_base64, prepare_query_for_multi_modal
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import load_extern_object
from verl.utils.ray_utils import get_event_loop

from .reward_model import RewardModelManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_custom_reward_kwargs(config: DictConfig) -> dict:
    reward_cfg = config.get("reward") or {}
    custom_cfg = reward_cfg.get("custom_reward_function") or {}
    reward_kwargs = custom_cfg.get("reward_kwargs") or {}
    return dict(reward_kwargs)


def _llm_judge_only_zerogroup_enabled(config: DictConfig) -> bool:
    reward_kwargs = _get_custom_reward_kwargs(config)
    only_zerogroup = reward_kwargs.get("llm_judge_only_zerogroup", False)
    return bool(reward_kwargs.get("llm_judge_enable", False)) and bool(only_zerogroup)


def _expected_llm_judge_group_size(config: DictConfig, *, validate: bool) -> int | None:
    rollout_cfg = config.get("actor_rollout_ref", {}).get("rollout", {})
    if validate:
        expected = rollout_cfg.get("val_kwargs", {}).get("n", None)
    else:
        expected = rollout_cfg.get("n", None)
    try:
        expected_int = int(expected)
    except (TypeError, ValueError):
        return None
    return expected_int if expected_int > 0 else None


def _reward_score_from_env(output: dict) -> float | None:
    reward_extra_info = output.get("reward_extra_info", {})
    if not isinstance(reward_extra_info, dict):
        return None
    try:
        return float(reward_extra_info.get("web_osgym_env_reward_score"))
    except (TypeError, ValueError):
        return None


def _load_zero_group_compare_fn(config: DictConfig):
    reward_kwargs = _get_custom_reward_kwargs(config)
    if not (bool(reward_kwargs.get("llm_judge_enable", False)) and bool(reward_kwargs.get("llm_judge_only_zerogroup", False))):
        return None
    reward_fn_config = config.reward.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None
    return load_extern_object(module_path=module_path, object_name="compare_zero_group_webgym_rl")


def _load_zero_group_compare_async_fn(config: DictConfig):
    reward_kwargs = _get_custom_reward_kwargs(config)
    if not (bool(reward_kwargs.get("llm_judge_enable", False)) and bool(reward_kwargs.get("llm_judge_only_zerogroup", False))):
        return None
    reward_fn_config = config.reward.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None
    try:
        return load_extern_object(module_path=module_path, object_name="compare_zero_group_webgym_rl_async")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load compare_zero_group_webgym_rl_async "
            f"from reward.custom_reward_function.path={module_path!r}"
        ) from exc


def _llm_judge_max_concurrency(config: DictConfig) -> int:
    reward_kwargs = _get_custom_reward_kwargs(config)
    value = reward_kwargs.get("llm_judge_max_concurrency", 2)
    try:
        concurrency = int(value)
    except (TypeError, ValueError):
        return 2
    return max(concurrency, 1)


def _run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def _runner():
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive bridge
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def _load_reward_extra_info_merge_fn(config: DictConfig):
    reward_fn_config = config.reward.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None
    return load_extern_object(module_path=module_path, object_name="merge_webgym_reward_extra_info")


def _load_reward_extra_info_pack_fn(config: DictConfig):
    reward_fn_config = config.reward.get("custom_reward_function") or {}
    module_path = reward_fn_config.get("path")
    if not module_path:
        return None
    return load_extern_object(module_path=module_path, object_name="pack_webgym_reward_extra_infos")


def migrate_legacy_reward_impl(config):
    """
    Migrate the legacy reward model implementation to the new one.
    """
    # 1. reward workers migration
    # config.reward_model.num_workers -> config.reward.num_workers
    if config.reward_model.num_workers is not None:
        config.reward.num_workers = config.reward_model.num_workers

    # 2. reward manager migration
    # config.reward_model.reward_manager -> config.reward.reward_manager
    if config.reward_model.reward_manager is not None:
        config.reward.reward_manager.name = config.reward_model.reward_manager
    if config.reward_model.reward_loop_source is not None:
        config.reward.reward_manager.source = config.reward_model.reward_loop_source
        config.reward.reward_manager.module.path = config.reward_model.reward_loop_module_path
        config.reward.reward_manager.module.name = config.reward_model.reward_loop_class_name

    # 3. custom reward function migration
    # config.custom_reward_function -> config.reward.custom_reward_function
    if not all(v is None for v in config.custom_reward_function.values()):
        config.reward.custom_reward_function = config.custom_reward_function

    # 4. reward model migration
    # config.reward_model -> config.reward.reward_model
    for key in ["enable", "enable_resource_pool", "n_gpus_per_node", "nnodes"]:
        if config.reward_model.get(key) is not None:
            config.reward.reward_model[key] = config.reward_model[key]
    if config.reward_model.model.path is not None:
        config.reward.reward_model.model_path = config.reward_model.model.path
    # config.reward_model.reward_kwargs -> config.reward.reward_kwargs (for dapo algo)
    if config.reward_model.get("reward_kwargs") is not None:
        with open_dict(config.reward):
            config.reward["reward_kwargs"] = config.reward_model["reward_kwargs"]
    # config.reward_model.rollout -> config.reward.reward_model.rollout
    legacy_rollout = config.reward_model.rollout
    for key in legacy_rollout.keys():
        if legacy_rollout[key] is not None:
            config.reward.reward_model.rollout[key] = legacy_rollout[key]

    # 5. sandbox_fusion migration
    # config.sandbox_fusion -> reward.sandbox_fusion
    if not all(v is None for v in config.sandbox_fusion.values()):
        config.reward.sandbox_fusion = config.sandbox_fusion

    # 6. delete legacy config from configs
    with open_dict(config):
        del config.reward_model
        del config.custom_reward_function
        del config.sandbox_fusion

    return config


class RewardLoopWorker:
    """
    RewardLoopWork can tackle reward computation:
    (1) rule-based reward computation
    (2) reward model-based reward computation (both disrm and genrm)
    (3) high-flexible user-customized reward function (can access rm by posting requests to reward_model_router)

    Reward Computation Logic:
    - if user-customized reward function is provided:
        -> directly use user-customized reward function
    - if user-customized reward function is not provided:
        -> rm is not enabled: use default rule-based reward function
        -> rm is disrm: compute reward score using disrm
        -> rm is genrm: raise error (user-costomized reward func must be provided)
    """

    def __init__(self, config: DictConfig, reward_router_address: str = None):
        """
        Args:
            config: DictConfig, the config for reward loop worker.
            reward_router_address: str, the address of reward router.
        """
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()
        self.loop = get_event_loop()

    def _init_reward_fn(self):
        input_tokenizer_path = self.config.actor_rollout_ref.model.tokenizer_path
        if input_tokenizer_path is None:
            input_tokenizer_path = self.config.actor_rollout_ref.model.path
        input_tokenizer_local_path = copy_to_local(input_tokenizer_path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        self.reward_model_tokenizer = None
        if self.config.reward.reward_model.enable:
            reward_model_tokenizer_local_path = copy_to_local(self.config.reward.reward_model.model_path)
            self.reward_model_tokenizer = hf_tokenizer(reward_model_tokenizer_local_path, trust_remote_code=True)

        self.reward_manager = load_reward_manager(
            self.config,
            self.input_tokenizer,
            reward_router_address=self.reward_router_address,
            reward_model_tokenizer=self.reward_model_tokenizer,
        )

    async def compute_score_batch(self, data: DataProto) -> list[dict]:
        tasks = []
        for i in range(len(data)):
            tasks.append(asyncio.create_task(self.compute_score(data[i : i + 1])))
        outputs = await asyncio.gather(*tasks)
        return outputs

    async def compute_score(self, data: DataProto) -> dict:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        if self.config.reward.custom_reward_function.path is not None:
            # directly use user-customized reward function
            return await self.reward_manager.run_single(data)
        else:
            if self.config.reward.reward_model.enable:
                # we assume the rm is disrm
                # genrm must set custom_reward_function
                return await self.compute_score_disrm(data)
            else:
                return await self.reward_manager.run_single(data)

    async def _post_request(self, payload: dict, endpoint: str, max_retries: int = 16):
        url = f"http://{self.reward_router_address}/{endpoint}"
        last_exception = None
        for attempt in range(max_retries):
            try:
                # It's safer to have a timeout instead of None, which can hang indefinitely.
                timeout = aiohttp.ClientTimeout(total=None)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientResponseError as e:
                # Do not retry on 4xx client errors, but retry on 5xx server errors.
                if 400 <= e.status < 500:
                    logger.error(f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying.")
                    raise
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. "
                    "Retrying..."
                )
            except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
                last_exception = e
                logger.warning(f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying...")
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. "
                    "Retrying..."
                )

            if attempt < max_retries - 1:
                # Using exponential backoff is generally better than a fixed sleep.
                backoff_seconds = 2**attempt
                await asyncio.sleep(min(backoff_seconds, 30))

        logger.error(f"Max retries ({max_retries}) reached for request to {url}.")
        if last_exception:
            raise last_exception

    async def _preprocess_reward_inputs(self, data: DataProto) -> str:
        assert len(data) == 1, "RewardLoopWorker only support single data item"
        data_item = data[0]
        assert "raw_prompt" in data_item.non_tensor_batch

        # extract raw prompt
        chat: list = list(data_item.non_tensor_batch["raw_prompt"])

        # extract response
        response = data_item.batch["responses"]
        if response.ndim == 3:
            # handling multi-modal response
            response_image = response
            if isinstance(response_image, torch.Tensor):
                response_image = response_image.float().permute(1, 2, 0).cpu().numpy()
            assert response_image.shape[-1] == 3, "must be in HWC format"
            response_image = (response_image * 255).round().clip(0, 255).astype(np.uint8)
            response_image = Image.fromarray(response_image)

            image_base64 = await self.loop.run_in_executor(None, pil_image_to_base64, response_image)
            query = prepare_query_for_multi_modal(image_base64)

            chat.append({"role": "assistant", "content": query})
        else:
            response_ids = response
            response_length = response_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            rollout_response = self.input_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            rollout_response = rollout_response.replace(self.input_tokenizer.eos_token, "")

            chat.append({"role": "assistant", "content": rollout_response})

        rm_prompt = self.reward_model_tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=False,
        )

        # llama tokenizer will add bos token by default
        # will be removed in vllm >= 0.11.2, where we can add "add_special_tokens" = False
        if self.reward_model_tokenizer.bos_token is not None and rm_prompt.startswith(
            self.reward_model_tokenizer.bos_token
        ):
            rm_prompt = rm_prompt[len(self.reward_model_tokenizer.bos_token) :]

        return rm_prompt

    async def compute_score_disrm(self, data: DataProto) -> dict:
        disrm_prompt = await self._preprocess_reward_inputs(data)
        engine_name = self.config.reward.reward_model.rollout.name
        model_name = self.config.reward.reward_model.model_path
        if engine_name == "vllm":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
                "use_activation": False,
            }
            output = await self._post_request(payloads, "classify")
            rm_score = output["data"][-1]["probs"][-1]
        elif engine_name == "sglang":
            payloads = {
                "model": model_name,
                "input": disrm_prompt,
            }
            output = await self._post_request(payloads, "v1/embeddings")
            rm_score = output["data"][-1]["embedding"][-1]
        elif engine_name == "trtllm":
            # TODO: remove this once TRT-LLM switches to TorchSampler
            raise ValueError("TensorRT-LLM backend does not support reward models currently.")

            payloads = {
                "model": model_name,
                "prompt": disrm_prompt,
                "return_context_logits": True,
            }
            output = await self._post_request(payloads, "v1/completions")
            rm_score = output["choices"][0]["context_logits"]
            assert isinstance(rm_score, list) and len(rm_score) > 0, (
                "TensorRT-LLM OpenAI server response for reward score is not in the expected format."
            )

            rm_score = float(rm_score[0][0])
            logger.debug(f"rm score: {rm_score}")
        else:
            raise NotImplementedError(f"RewardLoopManager does not support {engine_name}")

        return {"reward_score": rm_score}


class RewardLoopManager:
    """
    RewardLoopManager run in single controller.
    This class will create reward loop workers and manage them.
    """

    def __init__(self, config: DictConfig, rm_resource_pool: RayResourcePool = None):
        self.config = config
        if self.config.reward.reward_model.enable:
            self.reward_model_manager = RewardModelManager(config.reward.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()
        else:
            self.reward_model_manager = None
            self.reward_router_address = None

        self.reward_loop_workers_class = ray.remote(RewardLoopWorker)
        self._init_reward_loop_workers()
        self.zero_group_compare_fn = _load_zero_group_compare_fn(config)
        self.zero_group_compare_async_fn = _load_zero_group_compare_async_fn(config)
        self.reward_extra_info_merge_fn = _load_reward_extra_info_merge_fn(config)
        self.reward_extra_info_pack_fn = _load_reward_extra_info_pack_fn(config)
        print(
            "[RewardLoopManager] "
            f"llm_judge_enable={bool(_get_custom_reward_kwargs(self.config).get('llm_judge_enable', False))} "
            f"llm_judge_only_zerogroup={bool(_get_custom_reward_kwargs(self.config).get('llm_judge_only_zerogroup', False))} "
            f"streaming_reward={'disabled' if _llm_judge_only_zerogroup_enabled(self.config) else 'enabled'}"
        )

    @property
    def reward_loop_worker_handles(self) -> list[ActorHandle]:
        """Return worker handles for agent loop worker to compute reward score.

        Only return worker handles when reward computation can be parallelized with rollout:
        (1) rule-based reward without reward model
        (2) reward model with extra resource pool
        """
        if _llm_judge_only_zerogroup_enabled(self.config):
            return None
        if not self.config.reward.reward_model.enable or self.config.reward.reward_model.enable_resource_pool:
            return self.reward_loop_workers
        return None

    def _init_reward_loop_workers(self):
        self.reward_loop_workers = []
        num_workers = self.config.reward.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]

        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]

            self.reward_loop_workers.append(
                self.reward_loop_workers_class.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=True,
                    ),
                ).remote(self.config, self.reward_router_address)
            )

    def _run_reward_workers(self, data: DataProto) -> list[dict]:
        if len(data) == 0:
            return []
        num_chunks = min(len(data), len(self.reward_loop_workers))
        split_size = (len(data) + num_chunks - 1) // num_chunks
        chunks = data.split(split_size)
        outputs = ray.get(
            [
                worker.compute_score_batch.remote(chunk)
                for worker, chunk in zip(self.reward_loop_workers[: len(chunks)], chunks, strict=True)
            ]
        )
        return [item for sublist in outputs for item in sublist]

    def _maybe_apply_zerogroup_llm_judge(self, data: DataProto, outputs_flat: list[dict]) -> list[dict]:
        if not _llm_judge_only_zerogroup_enabled(self.config):
            return outputs_flat

        uids = data.non_tensor_batch.get("uid")
        if not isinstance(uids, np.ndarray) or len(uids) != len(outputs_flat):
            return outputs_flat
        extra_infos = data.non_tensor_batch.get("extra_info")
        if not isinstance(extra_infos, np.ndarray) or len(extra_infos) != len(outputs_flat):
            return outputs_flat

        validate = bool(data.meta_info.get("validate", False))
        expected_group_size = _expected_llm_judge_group_size(self.config, validate=validate)

        grouped_indices: dict[str, list[int]] = {}
        for index, uid in enumerate(uids.tolist()):
            grouped_indices.setdefault(str(uid), []).append(index)

        selected_groups: list[list[int]] = []
        selected_group_count = 0
        skipped_incomplete_groups = 0
        skipped_missing_env_groups = 0
        skipped_missing_judge_standard_groups = 0
        skipped_invalid_compare_input_groups = 0
        for indices in grouped_indices.values():
            if expected_group_size is not None and len(indices) != expected_group_size:
                skipped_incomplete_groups += 1
                continue
            env_scores = [_reward_score_from_env(outputs_flat[index]) for index in indices]
            if any(score is None for score in env_scores):
                skipped_missing_env_groups += 1
                continue
            if not all(
                isinstance(extra_infos[index], dict)
                and isinstance(extra_infos[index].get("judge_standard"), list)
                and any(
                    isinstance(item, dict)
                    and str(item.get("id", "")).strip()
                    and str(item.get("text", "")).strip()
                    for item in extra_infos[index]["judge_standard"]
                )
                for index in indices
            ):
                skipped_missing_judge_standard_groups += 1
                continue
            if all(float(score) <= 0.0 for score in env_scores):
                selected_group_count += 1
                selected_groups.append(indices)

        if selected_group_count > 0 or skipped_incomplete_groups > 0 or skipped_missing_judge_standard_groups > 0:
            print(
                "[RewardLoopManager][ZeroGroup] "
                f"validate={validate} batch_size={len(data)} uid_groups={len(grouped_indices)} "
                f"expected_group_size={expected_group_size} selected_groups={selected_group_count} "
                f"selected_samples={sum(len(indices) for indices in selected_groups)} skipped_incomplete_groups={skipped_incomplete_groups} "
                f"skipped_missing_env_groups={skipped_missing_env_groups} "
                f"skipped_missing_judge_standard_groups={skipped_missing_judge_standard_groups}"
            )

        if not selected_groups:
            return outputs_flat

        compare_fn = getattr(self, "zero_group_compare_fn", None) or _load_zero_group_compare_fn(self.config)
        compare_async_fn = getattr(self, "zero_group_compare_async_fn", None) or _load_zero_group_compare_async_fn(self.config)
        if compare_fn is None and compare_async_fn is None:
            raise RuntimeError("compare_zero_group_webgym_rl is not available for zerogroup judge")
        merge_fn = getattr(self, "reward_extra_info_merge_fn", None) or _load_reward_extra_info_merge_fn(self.config)

        raw_prompts = data.non_tensor_batch.get("raw_prompt")
        group_payloads: list[tuple[list[int], list[dict], list[str | None]]] = []
        for indices in selected_groups:
            group_extra_infos = []
            group_task_instructions = []
            for index in indices:
                item = extra_infos[index]
                group_extra_infos.append(dict(item) if isinstance(item, dict) else {})
                task_instruction = None
                if isinstance(raw_prompts, np.ndarray) and index < len(raw_prompts):
                    prompt = raw_prompts[index]
                    if isinstance(prompt, list):
                        for message in prompt:
                            if isinstance(message, dict) and message.get("role") == "user":
                                content = message.get("content")
                                if isinstance(content, str) and content.strip():
                                    task_instruction = content.strip()
                                    break
                group_task_instructions.append(task_instruction)
            group_payloads.append((indices, group_extra_infos, group_task_instructions))

        if compare_async_fn is not None:
            judged_group_results = _run_coroutine_sync(
                self._judge_selected_groups_async(compare_async_fn, group_payloads)
            )
        else:
            judged_group_results = []
            for indices, group_extra_infos, group_task_instructions in group_payloads:
                try:
                    judged_outputs = compare_fn(
                        extra_infos=group_extra_infos,
                        task_instructions=group_task_instructions,
                        **_get_custom_reward_kwargs(self.config),
                    )
                    judged_group_results.append((indices, judged_outputs, False))
                except Exception:
                    judged_group_results.append((indices, None, True))

        for indices, judged_outputs, failed in judged_group_results:
            if failed or judged_outputs is None:
                skipped_invalid_compare_input_groups += 1
                continue
            for original_index, judged_output in zip(indices, judged_outputs, strict=True):
                if merge_fn is not None:
                    merged_reward_extra_info = merge_fn(
                        outputs_flat[original_index].get("reward_extra_info", {}),
                        judged_output.get("reward_extra_info", {}),
                    )
                else:
                    merged_reward_extra_info = dict(outputs_flat[original_index].get("reward_extra_info", {}))
                    merged_reward_extra_info.update(judged_output.get("reward_extra_info", {}))
                outputs_flat[original_index] = {
                    "reward_score": judged_output["reward_score"],
                    "reward_extra_info": merged_reward_extra_info,
                }
        if skipped_invalid_compare_input_groups > 0:
            print(
                "[RewardLoopManager][ZeroGroupInvalidInput] "
                f"skipped_invalid_compare_input_groups={skipped_invalid_compare_input_groups}"
            )
        return outputs_flat

    async def _judge_selected_groups_async(self, compare_async_fn, group_payloads):
        semaphore = asyncio.Semaphore(_llm_judge_max_concurrency(self.config))
        reward_kwargs = _get_custom_reward_kwargs(self.config)

        async def _judge_single(indices, group_extra_infos, group_task_instructions):
            async with semaphore:
                try:
                    judged_outputs = await compare_async_fn(
                        extra_infos=group_extra_infos,
                        task_instructions=group_task_instructions,
                        **reward_kwargs,
                    )
                    return indices, judged_outputs, False
                except Exception:
                    return indices, None, True

        return await asyncio.gather(
            *[
                _judge_single(indices, group_extra_infos, group_task_instructions)
                for indices, group_extra_infos, group_task_instructions in group_payloads
            ]
        )

    def compute_rm_score(self, data: DataProto) -> DataProto:
        if self.reward_model_manager is not None:
            self.reward_model_manager.wake_up()

        outputs_flat = self._run_reward_workers(data)
        outputs_flat = self._maybe_apply_zerogroup_llm_judge(data, outputs_flat)

        # compute rm score
        scores = [item["reward_score"] for item in outputs_flat]
        if self.config.reward.reward_manager.name == "visual":
            # visual reward only has one score for the whole response
            rm_scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
        else:
            prompt_length = data.batch["prompts"].size(1)
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1)
            rm_scores = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            rm_scores[torch.arange(rm_scores.size(0)), valid_response_length - 1] = torch.tensor(
                scores, dtype=torch.float32
            )
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))

        reward_extra_infos = [output.get("reward_extra_info", {}) for output in outputs_flat]
        pack_fn = getattr(self, "reward_extra_info_pack_fn", None) or _load_reward_extra_info_pack_fn(self.config)
        if pack_fn is not None:
            non_tensor_batch, reward_extra_keys = pack_fn(
                reward_extra_infos,
                template_non_tensor_batch=data.non_tensor_batch,
            )
        else:
            reward_extra_keys = sorted(
                {key for reward_extra_info in reward_extra_infos if isinstance(reward_extra_info, dict) for key in reward_extra_info}
            )
            non_tensor_batch = {}
            for key in reward_extra_keys:
                existing_value = data.non_tensor_batch.get(key)
                target_dtype = existing_value.dtype if isinstance(existing_value, np.ndarray) else object
                non_tensor_batch[key] = np.array(
                    [info.get(key) if isinstance(info, dict) else None for info in reward_extra_infos],
                    dtype=target_dtype,
                )

        if self.reward_model_manager is not None:
            self.reward_model_manager.sleep()

        return DataProto(
            batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"reward_extra_keys": reward_extra_keys}
        )

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all())
