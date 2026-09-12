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
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams, List[bytes] | None]] = []
        self.status_map: Dict[int, RequestStatus] = {}
        self.counter = 0

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
        for tokens_or_prompt, sampling_params, images in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break
            input_ids = self._tokenize_one(tokens_or_prompt)
            msg = UserMsg(uid=0, input_ids=input_ids, sampling_params=sampling_params)
            if images:
                from freetoken.mm.processor import get_mm_processor

                processor = get_mm_processor(self.config.model_path)
                if processor is None:
                    raise ValueError("image input is not supported for this model")
                r = processor.apply(input_ids, images, max_pixels=None)
                input_ids = r.input_ids
                msg = UserMsg(
                    uid=0,
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    mm_items=r.mm_items,
                    mrope_positions=r.mrope_positions,
                    mrope_delta=r.mrope_delta,
                )
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            msg.uid = uid
            results.append(msg)
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=input_ids.tolist(),
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
        images: List[List[bytes] | None] | None = None,
    ) -> List[Dict[str, str | List[int]]]:
        """Offline generation; images is aligned with prompts: the raw image files of each prompt in placeholder order, or None."""
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        if images is None:
            images = [None] * len(prompts)
        for prompt, sp, imgs in zip(prompts, sampling_params, images, strict=True):
            self.pending_requests.append((prompt, sp, imgs))
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
