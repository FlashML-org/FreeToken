"""Collect reviewable experiment evidence as ordinary files, without an archive."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'FreeToken/benchmarks/sigmoid_router_validation'
DEST.mkdir(parents=True, exist_ok=True)
entries = []


def copy(source, relative):
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if relative.startswith('reports/'):
        content = target.read_text(encoding='utf-8-sig')
        content = content.replace('(validation/', '(../history/')
        content = content.replace('(FreeToken/benchmarks/sigmoid_router_validation/REVIEW.md)', '(../REVIEW.md)')
        content = content.replace('(FreeToken/benchmarks/sigmoid_router_validation/linux/README.md)', '(../linux/README.md)')
        target.write_text(content, encoding='utf-8')
    entries.append(dict(path=relative, source=str(source), bytes=target.stat().st_size,
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


for source in sorted((ROOT / 'validation').iterdir()):
    if source.is_file() and source.suffix in {'.py', '.json', '.log', '.md'}:
        copy(source, 'history/' + source.name)
for source in sorted((ROOT / 'validation/cuda-20261021T011532Z').iterdir()):
    if source.is_file():
        copy(source, 'history/cuda-20261021T011532Z/' + source.name)
prefix = Path('D:/AI/FreeToken-validation/GLM-4.5-Air-prefix')
for name in ['config.json', 'tensor-manifest.json', 'prompt-1-activations.pt',
             'prompt-2-activations.pt', 'prompt-3-activations.pt']:
    copy(prefix / name, 'real-prefix/' + name)
for name in ['FreeToken-实验报告.md', 'FreeToken-PR-验收报告.md']:
    copy(ROOT / name, 'reports/' + name)
(DEST / 'inventory.json').write_text(json.dumps(dict(
    scope='All local small experiment records and saved prefix activations; no archive or upload',
    files=entries,
    excluded=['model weight binaries and embedding rows (see tensor-manifest.json)',
              'virtual environments, downloaded runtimes and uv.exe',
              'baseline checkout and baseline.zip (recreate using git archive of the recorded base)',
              'Python/Triton/Hugging Face caches',
              'PR/application drafts, which are not experimental results']),
    ensure_ascii=False, indent=2), encoding='utf-8')
(DEST / '.gitignore').write_text('!*.log\n__pycache__/\n', encoding='utf-8')
print(json.dumps(dict(files=len(entries), bytes=sum(x['bytes'] for x in entries),
                      directory=str(DEST)), ensure_ascii=False))
