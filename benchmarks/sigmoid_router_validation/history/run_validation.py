"""Record local checks without treating skipped GPU tests as passes."""
import datetime
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'FreeToken'
OUT = ROOT / 'validation'
FILES = [
    'python/freetoken/kernel/triton/sigmoid_router.py',
    'python/freetoken/layers/linear.py',
    *[f'python/freetoken/models/{f}/moe.py' for f in ('glm4_moe','glm_moe_dsa','glm5_next','minimax_m2','minimax_m3')],
    'tests/kernels/test_sigmoid_router.py', 'tests/models/test_sigmoid_routing.py',
    'benchmarks/bench_sigmoid_router.py',
]
commands = [
    ('environment', [sys.executable, '-c', 'import torch,platform; print(platform.platform()); print("torch",torch.__version__); print("CUDA available",torch.cuda.is_available()); print("CUDA runtime",torch.version.cuda)']),
    ('syntax', [sys.executable, '-m', 'compileall', '-q', *FILES]),
    ('lint', [sys.executable, '-m', 'ruff', 'check', '--select', 'E9,F', *FILES]),
    ('diff_check', ['git', 'diff', '--check']),
    ('cpu_reference_only', [sys.executable, '-m', 'pytest', str(OUT/'test_cpu_reference.py'), '-q', '--confcutdir', str(OUT)]),
    ('gpu_kernel_tests', [sys.executable, '-m', 'pytest', 'tests/kernels/test_sigmoid_router.py', '-q', '-ra', '--confcutdir=tests/kernels']),
    ('gpu_model_tests', [sys.executable, '-m', 'pytest', 'tests/models/test_sigmoid_routing.py', '-q', '-ra', '--confcutdir=tests/models']),
    ('benchmark_attempt', [sys.executable, 'benchmarks/bench_sigmoid_router.py']),
    ('packages', [str(OUT/'uv.exe'), 'pip', 'freeze', '--python', sys.executable]),
]
results = []
for name, command in commands:
    p = subprocess.run(command, cwd=REPO, text=True, encoding='utf-8', errors='replace', capture_output=True)
    output = p.stdout + p.stderr
    (OUT/f'{name}.log').write_text(output, encoding='utf-8')
    results.append(dict(name=name, command=command, exit_code=p.returncode, output=output))
    print(f'{name}: exit={p.returncode}\n{output[:4000]}', flush=True)
report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              python=sys.version, platform=platform.platform(), results=results)
(OUT/'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
