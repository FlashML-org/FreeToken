"""Real GLM prefix activations and selected-expert outputs, NOT full generation."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer, Glm4MoeAttention, Glm4MoeRMSNorm,
    Glm4MoeRotaryEmbedding, Glm4MoeTopkRouter, Glm4MoeMLP,
)
from checkpoint_slices import SliceReader, MODEL, REVISION

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('FREETOKEN_VALIDATION_OUTPUT', 'sigmoid-prefix-results'))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(ROOT/'python'))
from freetoken.layers.linear import Fp32RouterLinear
from freetoken.models.glm4_moe.moe import Glm4MoeSparseBlock

PROMPTS = [
    '一个水箱有 A、B 两根进水管和 C 一根排水管。单独使用时，A 管 6 小时注满，B 管 8 小时注满，C 管 12 小时排空。水箱最初为空，先同时打开 A、B 管 2 小时，再打开 C 管，三根管一起运行。还需要多少小时才能注满？请给出计算过程。',
    'Write a Python function that returns the first non-repeating character in a string. Explain the time complexity.',
    '解释一下为什么天空通常是蓝色的，而日落时常呈现红色。',
]


def load_module(module, prefix, reader):
    state = {name:reader.tensor(prefix+'.'+name).cuda() for name in module.state_dict()}
    module.load_state_dict(state, strict=True, assign=True)
    assert all(not p.is_meta for p in module.parameters())
    module.eval()
    return module


def compare(w, ids, wr, ir):
    same = (ids.sort(-1).values == ir.sort(-1).values).all(-1)
    aw = w.gather(1,ids.argsort(-1))
    rw = wr.gather(1,ir.argsort(-1))
    return dict(rows=len(ids), expert_set_mismatch_rows=int((~same).sum()),
        order_mismatch_rows=int((ids != ir).any(-1).sum()),
        max_abs_weight_error_matching_sets=float((aw[same]-rw[same]).abs().max()) if same.any() else None)


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    reader = SliceReader()
    cfg = Glm4MoeConfig.from_dict(reader.config)
    cfg._attn_implementation = 'eager'
    assert cfg.first_k_dense_replace == 1
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    encoded = [tokenizer.apply_chat_template([dict(role='user',content=p)],
        tokenize=True,add_generation_prompt=True,return_dict=True) for p in PROMPTS]
    token_lists = [item['input_ids'] for item in encoded]
    assert all(isinstance(token,int) for ids in token_lists for token in ids)
    print('Token counts:',[len(x) for x in token_lists],flush=True)
    embeddings = [reader.embedding_rows(ids).cuda().unsqueeze(0) for ids in token_lists]
    with torch.device('meta'):
        layer0 = Glm4MoeDecoderLayer(cfg,0)
        attn1 = Glm4MoeAttention(cfg,1)
        norm1 = Glm4MoeRMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        post1 = Glm4MoeRMSNorm(cfg.hidden_size,cfg.rms_norm_eps)
        hf_router = Glm4MoeTopkRouter(cfg)
        shared = Glm4MoeMLP(cfg, cfg.moe_intermediate_size*cfg.n_shared_experts)
    load_module(layer0,'model.layers.0',reader)
    load_module(attn1,'model.layers.1.self_attn',reader)
    load_module(norm1,'model.layers.1.input_layernorm',reader)
    load_module(post1,'model.layers.1.post_attention_layernorm',reader)
    load_module(hf_router,'model.layers.1.mlp.gate',reader)
    load_module(shared,'model.layers.1.mlp.shared_experts',reader)
    rotary = Glm4MoeRotaryEmbedding(cfg).cuda()
    with torch.device('cuda'):
        gate = Fp32RouterLinear(cfg.hidden_size,cfg.num_local_experts,has_bias=False)
    gate.weight = torch.empty_like(hf_router.weight)
    gate.load_state_dict({'weight':hf_router.weight.detach().clone()})
    block = SimpleNamespace(gate=gate,e_score_correction_bias=hf_router.e_score_correction_bias,
        top_k=cfg.num_experts_per_tok,norm_topk_prob=cfg.norm_topk_prob,
        n_group=cfg.n_group,topk_group=cfg.topk_group,routed_scaling_factor=cfg.routed_scaling_factor)
    outputs = []
    for prompt, token_ids, embeds in zip(PROMPTS,token_lists,embeddings):
        n = len(token_ids)
        positions = torch.arange(n,device='cuda')[None,:]
        mask = torch.full((n,n),torch.finfo(embeds.dtype).min,device='cuda',dtype=embeds.dtype).triu(1)[None,None]
        rope = rotary(embeds,positions)
        hidden = layer0(embeds,attention_mask=mask,position_ids=positions,position_embeddings=rope)
        residual1 = hidden+attn1(norm1(hidden),position_embeddings=rope,attention_mask=mask)[0]
        route_input = post1(residual1).reshape(n,-1)
        logits = F.linear(route_input.float(),hf_router.weight.float())
        s = logits.sigmoid()
        old_ids = (s+hf_router.e_score_correction_bias).topk(cfg.num_experts_per_tok,dim=-1).indices
        old_w = s.gather(1,old_ids)
        old_w = old_w/(old_w.sum(-1,keepdim=True)+1e-20)*cfg.routed_scaling_factor
        w, ids = Glm4MoeSparseBlock._route(block,route_input)
        _, hw, hi = hf_router(route_input)
        result = dict(prompt=prompt,token_ids=token_ids,vs_original=compare(w,ids,old_w,old_ids),
            vs_hf=compare(w,ids,hw,hi))
        print('Routing:',json.dumps(result['vs_original']),flush=True)
        # Only the final prompt token's expert outputs are evaluated to bound downloads.
        x = route_input[-1:]
        expert_ids = sorted(set(ids[-1].tolist()+old_ids[-1].tolist()))
        expert_outputs = {}
        for expert in expert_ids:
            prefix = f'model.layers.1.mlp.experts.{expert}'
            gate_w = reader.tensor(prefix+'.gate_proj.weight').cuda()
            up_w = reader.tensor(prefix+'.up_proj.weight').cuda()
            down_w = reader.tensor(prefix+'.down_proj.weight').cuda()
            expert_outputs[expert] = F.linear(F.silu(F.linear(x,gate_w))*F.linear(x,up_w),down_w)
            del gate_w,up_w,down_w
        def combine(index, weight):
            terms = [expert_outputs[int(i)].float()*a.float() for i,a in zip(index,weight)]
            return torch.stack(terms).sum(0).to(x.dtype)
        shared_out = shared(x)
        old_out = combine(old_ids[-1],old_w[-1])+shared_out+residual1[:,-1,:]
        new_out = combine(ids[-1],w[-1])+shared_out+residual1[:,-1,:]
        diff=(new_out.float()-old_out.float()).abs()
        result['last_token_layer1_output'] = dict(experts=expert_ids,
            bitwise_equal=bool(torch.equal(old_out,new_out)),max_abs_error=float(diff.max()),
            rms_error=float(diff.square().mean().sqrt()),
            reference_max_abs=float(old_out.abs().max()))
        outputs.append(result)
        reader.save_manifest()
        torch.save(dict(hidden_states=route_input.cpu(),logits=logits.cpu(),old_ids=old_ids.cpu(),
            old_weights=old_w.cpu(),new_ids=ids.cpu(),new_weights=w.cpu()),reader.root/f'prompt-{len(outputs)}-activations.pt')
        report=dict(model=MODEL,revision=REVISION,scope='HF BF16 prefix through layer1 route; actual selected expert weights for last input token; not full generation and not FreeToken expert backend',
            gpu=torch.cuda.get_device_name(),torch=torch.__version__,results=outputs)
        (OUT/'real-prompt-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Expert output:',result['last_token_layer1_output'],flush=True)


if __name__ == '__main__':
    main()
