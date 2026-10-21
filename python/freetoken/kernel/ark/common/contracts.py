import torch


def gfx1201(tensor):
    return (
        bool(torch.version.hip)
        and tensor.is_cuda
        and getattr(torch.cuda.get_device_properties(tensor.device), "gcnArchName", "").startswith("gfx1201")
    )


def plain_silu(activation, apply_router_weight_on_input, act_alpha, act_limit):
    return activation == "silu" and not apply_router_weight_on_input and act_alpha == 1.0 and act_limit == float("inf")


def moe_supported(x, w1, w2, tw, ids):
    dtype = torch.float8_e4m3fn
    return (
        gfx1201(x)
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and w1.ndim == 3
        and w2.ndim == 3
        and w1.dtype == dtype
        and w2.dtype == dtype
        and w1.is_contiguous()
        and w2.is_contiguous()
        and w1.shape[0] == w2.shape[0]
        and w1.shape[1] == 2 * w2.shape[2]
        and w1.shape[2] == x.shape[1]
        and w2.shape[1] == x.shape[1]
        and ids.ndim == 2
        and ids.shape == tw.shape
        and ids.shape[0] == x.shape[0]
        and ids.dtype == torch.int32
        and tw.dtype == torch.float32
        and ids.is_contiguous()
        and tw.is_contiguous()
        and x.shape[0] == 1
        and w1.shape[0] > 0
        and 0 < ids.shape[1]
        and w2.shape[2] > 0
        and x.shape[1] % 128 == 0
        and w2.shape[2] % 128 == 0
        and ids.shape[1] <= 16
        and all(t.device == x.device for t in (w1, w2, tw, ids))
    )


def scales_supported(s1, s2, w1, w2):
    return (
        s1.dtype == s2.dtype == torch.bfloat16
        and s1.is_contiguous()
        and s2.is_contiguous()
        and s1.ndim == s2.ndim == 3
        and s1.shape[:2] == (w1.shape[0], w1.shape[1] // 128)
        and s2.shape[:2] == (w2.shape[0], w2.shape[1] // 128)
        and s1.shape[2] >= w1.shape[2] // 128
        and s2.shape[2] >= w2.shape[2] // 128
        and s1.device == w1.device
        and s2.device == w2.device
    )
