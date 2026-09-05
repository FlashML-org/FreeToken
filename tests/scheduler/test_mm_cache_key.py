"""Image prompts take part in the prefix cache through ``cache_ids``: the same image hits, a
different image (same placeholder ids) matches only the text prefix, and a text prompt never
matches a placeholder run."""

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.utils import PendingReq

IMAGE = 9
KEY = 1 << 30


def _ids(*toks):
    return torch.tensor(toks, dtype=torch.int32)


def _finish(cm, uid, input_ids, cache_ids, table_idx):
    """Admit, 'prefill', and commit one request into the tree."""
    sp = SamplingParams(max_tokens=1)
    mr = cm.match_req(PendingReq(uid, input_ids, sp, cache_ids=cache_ids))
    req = Req(
        input_ids=input_ids,
        table_idx=table_idx,
        cached_len=mr.cuda_handle.cached_len,
        output_len=1,
        uid=uid,
        sampling_params=sp,
        cache_handle=mr.cuda_handle,
        mm_embeds=None if cache_ids is None else torch.zeros(3, 2),
        cache_ids=cache_ids,
    )
    cm.lock(mr.cuda_handle)
    cm.allocate_paged([req])
    req.complete_one()
    cm.cache_req(req, finished=True)
    return req


def _cached(cm, input_ids, cache_ids=None):
    return cm.match_req(
        PendingReq(99, input_ids, SamplingParams(max_tokens=1), cache_ids=cache_ids)
    ).cuda_handle.cached_len


def test_image_prompt_hits_only_for_the_same_image():
    cm = CacheManager(64, 1, torch.zeros(4, 64, dtype=torch.int32), "radix")
    prompt = _ids(1, 2, IMAGE, IMAGE, IMAGE, 3)
    same = _ids(1, 2, KEY + 7, KEY + 8, KEY + 9, 3)
    other = _ids(1, 2, KEY + 70, KEY + 71, KEY + 72, 3)
    _finish(cm, 1, prompt, same, table_idx=0)
    # match_req keys on all but the last token, so a follow-up turn hits the whole prompt
    follow = torch.cat([prompt, _ids(4, 5)])
    assert _cached(cm, follow, torch.cat([same, _ids(4, 5)])) == len(prompt)
    # a different image behind the same placeholder ids only matches the text before it
    assert _cached(cm, follow, torch.cat([other, _ids(4, 5)])) == 2
    # a text prompt with the raw placeholder ids never matches a keyed run
    assert _cached(cm, follow) == 2
    cm.check_integrity()
