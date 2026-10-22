import json,platform
import torch,triton
x=torch.arange(1024,device='cuda',dtype=torch.float32)
y=(x*x).sum()
torch.cuda.synchronize()
expected=sum(i*i for i in range(1024))
assert abs(y.item()-expected) <= 32
pinned=torch.empty(1024,dtype=torch.float32,pin_memory=True)
pinned.copy_(x.cpu())
assert torch.equal(pinned.cuda(),x)
print(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),gpu_sum=y.item(),expected=expected,pinned_memory_test='passed'),indent=2))
