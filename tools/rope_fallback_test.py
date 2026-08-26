"""Direct unit test of the engine fallback rope vs HF ground truth.
Build q_norm output (HF), apply fallback rope with engine cache, compare to HF rope.
This isolates the fallback function from all engine plumbing.
"""
import sys, numpy as np, torch
sys.path.insert(0, "/Users/petersheppard/FreeToken/python")
from freetoken.kernel.torch_fallback import apply_rope_with_cos_sin_cache_inplace
from freetoken.layers.rotary import get_rope
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "_test_models/qwen3-0.6b"
NQ, D = 16, 128
tok = AutoTokenizer.from_pretrained(MODEL)
input_ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()

# HF q_proj -> q_norm
q0 = {}
def q0_hook(m, a, o): q0["raw"] = o.detach().float().cpu().numpy()[0]
model.model.layers[0].self_attn.q_proj.register_forward_hook(q0_hook)
with torch.no_grad():
    model(input_ids=input_ids.unsqueeze(0))
seq = len(input_ids)
hf_q = torch.tensor(q0["raw"]).reshape(seq, NQ, D).double()
ln = model.model.layers[0].self_attn.q_norm
hf_qn = ln(hf_q)  # (seq, NQ, D)

# HF ground-truth rope
dim = 128; base = 10000.0
inv = 1.0/(base**(torch.arange(0,dim,2,dtype=torch.float64)/dim))
freqs = torch.outer(torch.arange(seq,dtype=torch.float64), inv)
cos = torch.cos(freqs); sin = torch.sin(freqs)  # (seq, dim/2)
half = D//2
x1 = hf_qn[..., :half]; x2 = hf_qn[..., half:]
rot = torch.cat([-x2, x1], -1)
c = cos.unsqueeze(1).expand(seq, NQ, half)
s = sin.unsqueeze(1).expand(seq, NQ, half)
hf_rope = hf_qn * torch.cat([c,c],-1) + rot * torch.cat([s,s],-1)

# ENGINE fallback rope (in-place) on SAME hf_qn
eng_q = hf_qn.clone().reshape(seq, NQ*D)  # fallback takes (tokens, head_size*?) - check shape
# fallback _rope handles 2D (tokens, head_size) by inserting dummy head dim
# but our q is (seq, NQ, D); flatten to (seq*NQ, D) then it treats each as separate token
eng_in = hf_qn.reshape(seq*NQ, D).clone()  # (seq*NQ, D)
# positions: each (seq, NQ) block gets position p
pos_flat = torch.arange(seq).repeat_interleave(NQ)
cache = get_rope(head_dim=128, rotary_dim=128, max_position=40960, base=10000.0)._cos_sin_cache.double()
apply_rope_with_cos_sin_cache_inplace(pos_flat, eng_in, eng_in.clone(), D, cache, is_neox=True)
eng_rope = eng_in.reshape(seq, NQ, D)

cs = torch.nn.functional.cosine_similarity(hf_rope.flatten().unsqueeze(0), eng_rope.flatten().unsqueeze(0)).item()
diff = (hf_rope - eng_rope).abs().max().item()
print(f"FALLBACK rope vs HF ground-truth: cos={cs:.6f} maxdiff={diff:.4f}")
print("  (1.0 => fallback correct; else fallback is the bug)")
