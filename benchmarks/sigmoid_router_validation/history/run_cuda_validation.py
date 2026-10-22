"""Record actual CUDA checks separately from the earlier CPU-only run."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'FreeToken'
KERNEL = REPO / 'python/freetoken/kernel/triton/sigmoid_router.py'

if '--isolated-kernel' in sys.argv:
    # Load the real kernel unchanged, bypassing unrelated engine package imports.
    name = 'freetoken.kernel.triton.sigmoid_router'
    spec = importlib.util.spec_from_file_location(name, KERNEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    import pytest
    raise SystemExit(pytest.main(['tests/kernels/test_sigmoid_router.py', '-q', '-ra',
                                  '--confcutdir=tests/kernels', '--tb=short']))

stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = ROOT / 'validation' / ('cuda-' + stamp)
out.mkdir()
env = dict(os.environ, PYTHONPATH=str(REPO/'python'), PYTHONUNBUFFERED='1')
commands = [
    ('hardware', ['nvidia-smi']),
    ('environment', [sys.executable, '-c', 'import torch,triton; print("torch",torch.__version__,"triton",triton.__version__); print("CUDA",torch.version.cuda,"available",torch.cuda.is_available()); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())']),
    ('gpu_kernel_tests', [sys.executable, str(Path(__file__).resolve()), '--isolated-kernel']),
    ('gpu_model_tests', [sys.executable, '-m', 'pytest', 'tests/models/test_sigmoid_routing.py', '-q', '-ra', '--confcutdir=tests/models', '--tb=short', '--maxfail=1']),
    ('benchmark', [sys.executable, 'benchmarks/bench_sigmoid_router.py']),
    ('packages', [str(ROOT/'validation/uv.exe'), 'pip', 'freeze', '--python', sys.executable]),
]
results = []
for name, command in commands:
    print('Running ' + name, flush=True)
    p = subprocess.run(command, cwd=REPO, env=env, text=True, encoding='utf-8', errors='replace', capture_output=True)
    output = p.stdout + p.stderr
    (out/(name+'.log')).write_text(output, encoding='utf-8')
    results.append(dict(name=name, command=command, exit_code=p.returncode, output=output))
    (out/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'{name}: exit={p.returncode}\n{output[:10000]}', flush=True)
print('Results: ' + str(out), flush=True)
