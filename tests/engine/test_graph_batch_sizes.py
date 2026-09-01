from freetoken.engine.graph import _determine_cuda_graph_bs


def test_offload_graph_shapes_do_not_exceed_flashlib_copy_plan():
    sizes = _determine_cuda_graph_bs(
        cuda_graph_bs=None,
        cuda_graph_max_bs=32,
        free_memory=24 << 30,
        moe_cache_size=56,
        moe_top_k=2,
    )

    assert sizes == [1, 2, 4, 8, 16, 24]


def test_explicit_graph_shapes_are_filtered_too():
    sizes = _determine_cuda_graph_bs(
        cuda_graph_bs=[1, 8, 32],
        cuda_graph_max_bs=None,
        free_memory=24 << 30,
        moe_cache_size=56,
        moe_top_k=2,
    )

    assert sizes == [1, 8]
