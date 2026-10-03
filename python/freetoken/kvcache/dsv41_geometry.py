"""DSV41Geometry: everything the KV pool and its cost model need to know about a
Compressed-Sparse-Attention-2 model (DeepSeek-V4.1), independent of the model package.

Every layer has a sliding window (``window`` tokens) of layer-local KV. Layers with
``compress_ratio > 0`` also attend over the global compressed KV, which only the
``kv_source_layer_ids`` produce (one latent per ``ratio`` tokens, plus the indexer key
projected from it); every other DSV41 layer reads the most recent source's pools. The pool
therefore allocates the global tiers PER SOURCE and aliases consumers onto them.

Built by ``dsv41_geometry`` from ``ModelConfig.dsv41_args``, read by attribute like DSV4's pool reads
``dsv4_args``: the kvcache package never imports the model package.
"""

from __future__ import annotations

from dataclasses import dataclass

from .row_format import FP4_E4M3_B16, FP4_E8M0_B32, FP8_E8M0_B32, RowFormat


@dataclass(frozen=True)
class DSV41Geometry:
    n_layers: int
    head_dim: int  # the shared latent width (K == V)
    index_head_dim: int
    window: int  # sliding window == window page P
    compress_ratios: tuple[int, ...]  # per layer; 0 = window-only layer
    kv_source_layer_ids: tuple[int, ...]  # layers that own a compressed KV + index-key pool
    win_fmt: RowFormat = FP8_E8M0_B32
    main_fmt: RowFormat = FP4_E4M3_B16
    idx_fmt: RowFormat = FP4_E8M0_B32
    # Layers whose window KV is REQUEST-PRIVATE: kept in a per-page-table-row ring of ``window``
    # slots (slot = row * window + pos % window) instead of the shared, page-bound window pool, so
    # it never lands in radix-shared pages. The decoder layers under Decoder SWA Bounded Replay:
    # their KV is recomputed per request over the prompt's last window (positions before it are
    # never computed), so a shared page could hold rows another request never wrote. Exact mode
    # computes every position and keeps the decoder on the shared pool.
    private_window_layer_ids: tuple[int, ...] = ()

    @property
    def shared_window_layer_ids(self) -> tuple[int, ...]:
        return tuple(l for l in range(self.n_layers) if l not in self.private_window_layer_ids)

    def is_private_window(self, layer_id: int) -> bool:
        return layer_id in self.private_window_layer_ids

    def __post_init__(self) -> None:
        for layer in self.private_window_layer_ids:
            if not (0 <= layer < self.n_layers):
                raise ValueError(f"private window layer {layer} out of range")
        if len(self.compress_ratios) != self.n_layers:
            raise ValueError(f"compress_ratios has {len(self.compress_ratios)} entries for {self.n_layers} layers")
        for fmt, dim in ((self.win_fmt, self.head_dim), (self.main_fmt, self.head_dim), (self.idx_fmt, self.index_head_dim)):
            fmt.validate_dim(dim)
        for ratio in set(self.compress_ratios):
            if ratio and self.window % ratio:
                raise ValueError(f"window {self.window} must be a multiple of compress ratio {ratio}")
        for src in self.kv_source_layer_ids:
            if not (0 <= src < self.n_layers) or self.compress_ratios[src] == 0:
                raise ValueError(f"kv source layer {src} is not a compressing layer")
        for layer in range(self.n_layers):
            if self.compress_ratios[layer] and self.kv_source_of(layer) is None:
                raise ValueError(f"layer {layer} compresses but no kv source precedes it")
            src = self.kv_source_of(layer)
            if src is not None and self.compress_ratios[src] != self.compress_ratios[layer]:
                raise ValueError(f"layer {layer} (ratio {self.compress_ratios[layer]}) reads source {src} (ratio {self.compress_ratios[src]})")

    def kv_source_of(self, layer: int) -> int | None:
        """The most recent kv source at or before ``layer``; None for window-only layers."""
        if self.compress_ratios[layer] == 0:
            return None
        sources = [s for s in self.kv_source_layer_ids if s <= layer]
        return max(sources) if sources else None

    def ratio_of(self, layer: int) -> int:
        return self.compress_ratios[layer]

    @property
    def ring_sources(self) -> tuple[int, ...]:
        """Sources whose compressor carries a partial group across steps (ratio > 1)."""
        return tuple(s for s in self.kv_source_layer_ids if self.compress_ratios[s] > 1)

    def ring_size(self, source: int) -> int:
        """Compress-state ring slots per window page for a source: the ratio (one slot per
        partial-group position); ``ring_size | window`` so pages map to disjoint blocks."""
        return self.compress_ratios[source]

    @property
    def win_row_bytes(self) -> int:
        return self.win_fmt.row_bytes(self.head_dim)

    @property
    def main_row_bytes(self) -> int:
        return self.main_fmt.row_bytes(self.head_dim)

    @property
    def idx_row_bytes(self) -> int:
        return self.idx_fmt.row_bytes(self.index_head_dim)

    @property
    def state_bytes(self) -> int:
        """One compress-state ring slot: fp32 ``kv | score`` of the latent width."""
        return 2 * self.head_dim * 4


def dsv41_geometry(args) -> DSV41Geometry:
    """The geometry of a DeepSeek-V4.1 ``dsv41_args``. Decoder SWA Bounded Replay computes the decoder
    only over each request's last window, so that KV is request-private: per-request rings, never
    radix-shared pages."""
    n = args.n_layers
    return DSV41Geometry(
        n_layers=n,
        head_dim=args.head_dim,
        index_head_dim=args.index_head_dim,
        window=args.window_size,
        compress_ratios=tuple(args.compress_ratios[:n]),
        kv_source_layer_ids=tuple(args.backbone_kv_sources),
        private_window_layer_ids=() if args.swa_decoder_replay == "exact" else tuple(range(args.decoder_start_layer, n)),
    )


__all__ = ["DSV41Geometry", "dsv41_geometry"]
