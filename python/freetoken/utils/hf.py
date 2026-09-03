import functools
import json
import os
from typing import Any

from typing import FrozenSet

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from freetoken.utils.logger import init_logger

logger = init_logger(__name__)


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    from freetoken.models.gguf.reader import gguf_config_source

    if (gguf_src := gguf_config_source(model_path)) is not None:
        from freetoken.models.gguf.tokenizer import load_gguf_tokenizer

        return load_gguf_tokenizer(gguf_src)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


def load_eos_token_ids(
    model_path: str, tokenizer: PreTrainedTokenizerBase
) -> FrozenSet[int]:
    """Return the full set of stop-token ids for generation.

    Chat models often terminate a turn with a token other than ``tokenizer.eos_token``
    (e.g. Gemma uses ``<end_of_turn>``), and list every stop id in
    ``generation_config.json``'s ``eos_token_id``. Honor that list (unioned with the
    tokenizer's eos) so generation halts correctly instead of running to ``max_tokens``.
    """
    from freetoken.models.gguf.reader import gguf_config_source

    if (gguf_src := gguf_config_source(model_path)) is not None:
        from freetoken.models.gguf.tokenizer import gguf_eos_token_ids

        return frozenset(gguf_eos_token_ids(gguf_src, tokenizer))

    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    try:
        gen_eos = GenerationConfig.from_pretrained(model_path).eos_token_id
    except Exception:
        gen_eos = None
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        ids.update(int(x) for x in gen_eos)
    return frozenset(ids)


def load_toolcall_anchor_id(
    tokenizer: PreTrainedTokenizerBase, opener: str | None
) -> int | None:
    """The single token id of ``opener`` -- the wire format's unique tool-call opening
    marker, declared by the model's detector (``BaseFormatDetector.toolcall_opener``).
    None when there is no opener or the tokenizer spells it with more than one token:
    the scheduler's special-token checkpoint matches sampled ids one at a time, so only
    a one-token opener can anchor."""
    if not opener:
        return None
    try:
        ids = tokenizer.encode(opener, add_special_tokens=False)
    except Exception:
        return None
    return int(ids[0]) if len(ids) == 1 else None


def load_generation_sampling(model_path: str) -> dict[str, Any]:
    """Recommended sampling defaults from ``generation_config.json`` (sglang's
    ``sampling_defaults='model'``). Returns ``{temperature, top_k, top_p}`` for the keys
    present; if the model recommends greedy (``do_sample=false``) returns greedy
    (``temperature=0``). Empty dict if there is no generation config / no sampling info.

    Many reasoning models (e.g. Qwen3.5: temp 1.0, top_k 20, top_p 0.95) ship these here;
    applying them avoids the greedy/unfiltered repetition loops these models fall into.
    """
    # GGUF carries the recommended sampling in metadata (general.sampling.*), not a
    # generation_config.json -- read it directly so --sampling-defaults=model works.
    from freetoken.models.gguf.reader import gguf_config_source

    if (gguf_src := gguf_config_source(model_path)) is not None:
        from freetoken.models.gguf.reader import load_gguf_metadata

        meta = load_gguf_metadata(gguf_src)
        out: dict[str, Any] = {}
        if (v := meta.get("general.sampling.temp")) is not None:
            out["temperature"] = float(v)
        if (v := meta.get("general.sampling.top_k")) is not None:
            out["top_k"] = int(v)
        if (v := meta.get("general.sampling.top_p")) is not None:
            out["top_p"] = float(v)
        return out

    try:
        gc = GenerationConfig.from_pretrained(model_path)
    except Exception:
        return {}
    if getattr(gc, "do_sample", None) is False:
        return {"temperature": 0.0}
    out: dict[str, Any] = {}
    for key in ("temperature", "top_k", "top_p"):
        val = getattr(gc, key, None)
        if val is not None:
            out[key] = val
    return out


class RawConfigShim:
    """Attribute view over a checkpoint's raw ``config.json``.

    Fallback for model types NEWER than the installed transformers (AutoConfig raises
    on an unknown ``model_type`` when the checkpoint ships no ``auto_map``, e.g.
    Muse-Glimmer on transformers < 5.15). FreeToken only reads config fields
    (parse_config) and never instantiates modeling code, so the raw JSON is enough.
    ``*_config`` sub-dicts are wrapped for attribute access, matching
    PretrainedConfig's nested-config behavior; every other value is served raw
    (``rope_parameters`` and friends stay plain dicts, as they do on the real class).
    """

    def __init__(self, data: dict | None = None, **kwargs: Any) -> None:
        self._data = {**(data or {}), **kwargs}

    def __getattr__(self, name: str) -> Any:
        if name == "_name_or_path":
            # PretrainedConfig carries the checkpoint path under this name and
            # DSV4's parse_config reads it (to find inference/config.json), so it
            # must survive the underscore guard below.
            return self.__dict__.get("_data", {}).get("_name_or_path", "")
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(name) from None
        if name.endswith("_config") and isinstance(value, dict):
            return RawConfigShim(value)
        return value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._data))  # deep copy, callers may mutate


def _raw_config_json(model_path: str) -> dict:
    if os.path.isdir(model_path):
        path = os.path.join(model_path, "config.json")
    else:
        path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    # trust_remote_code: checkpoints that ship a custom config class via ``auto_map``
    # (e.g. MiniMax-M2) refuse to load without it. FreeToken only reads config fields
    # (parse_config) and never instantiates the checkpoint's modeling code.
    try:
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except ValueError as exc:
        # Unknown model_type on this transformers version: serve off the raw JSON.
        # Anything else (bad path, malformed JSON) stays fatal.
        if "model type" not in str(exc):
            raise
        return RawConfigShim(_raw_config_json(model_path), _name_or_path=model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    # A .gguf file (or an FTW dir converted from one) carries its own metadata (no HF
    # config.json); return a shim the model registry dispatches on instead of a
    # PretrainedConfig.
    from freetoken.models.gguf.reader import gguf_config_source

    if (gguf_src := gguf_config_source(model_path)) is not None:
        from freetoken.models.gguf.config import build_gguf_shim

        return build_gguf_shim(gguf_src)
    config = _load_hf_config(model_path)
    if isinstance(config, RawConfigShim):
        return RawConfigShim(config.to_dict())
    return type(config)(**config.to_dict())


def _weight_allow_patterns(repo_id: str) -> list[str]:
    try:
        index = hf_hub_download(repo_id, SAFE_WEIGHTS_INDEX_NAME, tqdm_class=DisabledTqdm)
        with open(index, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    except Exception as e:
        logger.warning(
            "no usable %s for %s (%s); falling back to *.safetensors",
            SAFE_WEIGHTS_INDEX_NAME, repo_id, e,
        )
        return ["*.safetensors"]
    return shards or ["*.safetensors"]


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=_weight_allow_patterns(model_path),
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )