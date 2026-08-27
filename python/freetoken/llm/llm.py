from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from freetoken.core import SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.message import (
    BaseBackendMsg,
    BaseTokenizerMsg,
    DetokenizeMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    pass


@dataclass
class RequestStatus:
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.mm_embeds_map: Dict[int, torch.Tensor] = {}
        self.mrope_map: Dict[int, Tuple[torch.Tensor, int]] = {}
        self.counter = 0

    @torch.inference_mode()
    def encode_images(
        self, pixel_values: torch.Tensor, image_metadata: torch.Tensor
    ) -> torch.Tensor:
        """Run the vision tower + projector on processor outputs, returning the
        ``[num_image_tokens, hidden]`` soft-token embeddings (on device)."""
        model = self.engine.model
        if not hasattr(model, "encode_images"):
            raise RuntimeError(f"{type(model).__name__} does not support image inputs")
        return model.encode_images(
            pixel_values.to(self.device), image_metadata.to(self.device)
        )

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(
                UserMsg(
                    uid=uid,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    mm_embeds=self.mm_embeds_map.get(uid),
                    mrope_position_ids=(
                        self.mrope_map[uid][0] if uid in self.mrope_map else None
                    ),
                    mrope_position_delta=(
                        self.mrope_map[uid][1] if uid in self.mrope_map else 0
                    ),
                )
            )
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[BaseTokenizerMsg]) -> None:
        for msg in reply:
            if isinstance(msg, PromptAdmittedMsg):
                # PromptAdmittedMsg feeds the online server's global accounting. Offline
                # generation already owns its inputs and has no FrontendManager stats sink.
                continue
            assert isinstance(msg, DetokenizeMsg)
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token in self.eos_token_ids):
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
        mm_inputs: List[Dict[str, torch.Tensor] | None] | None = None,
    ) -> List[Dict[str, str | List[int]]]:
        """Offline generation.

        ``mm_inputs`` (optional) is aligned with ``prompts``; each entry is either
        ``None`` (text-only) or a dict with ``pixel_values`` ``[N, P, 3*patch**2]`` and
        either ``image_position_ids`` (Gemma) or ``image_grid_thw`` (Qwen) from
        the HF processor. For multimodal
        prompts pass token-id ``prompts`` containing ``image_token_id`` placeholders.
        """
        self.pending_requests = []
        self.status_map = {}
        self.mm_embeds_map = {}
        self.mrope_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        if mm_inputs is not None:
            for uid, mm in enumerate(mm_inputs):
                if mm is not None:
                    metadata = mm.get("image_grid_thw", mm.get("image_position_ids"))
                    if metadata is None:
                        raise ValueError(
                            "multimodal input needs image_grid_thw or image_position_ids"
                        )
                    self.mm_embeds_map[uid] = self.encode_images(
                        mm["pixel_values"], metadata
                    )
                    if "image_grid_thw" in mm:
                        mm_types = mm.get("mm_token_type_ids")
                        if mm_types is None:
                            raise ValueError(
                                "Qwen multimodal input needs mm_token_type_ids from the processor"
                            )
                        from freetoken.models.qwen4_exp.mrope import build_mrope_positions

                        prompt_ids = self._tokenize_one(prompts[uid])
                        merge_size = int(self.config.model_config.vision_config.spatial_merge_size)
                        self.mrope_map[uid] = build_mrope_positions(
                            prompt_ids,
                            mm_types,
                            mm["image_grid_thw"],
                            merge_size,
                        )
            torch.cuda.synchronize(self.device)
        try:
            self.run_forever()
        except RequestAllFinished:
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
