"""Render the measured timing distributions without rerunning GPU benchmarks."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'history/extended-benchmark.json').read_text(encoding='utf-8'))
rows = []
for result in data['results']:
    row = {k:result[k] for k in ('tokens','experts','topk','groups','keep','eps','scale')}
    for name in ('eager','fused'):
        values = result[name+'_samples_us']
        row.update({name+'_median_us':statistics.median(values), name+'_min_us':min(values),
                    name+'_max_us':max(values),name+'_sample_stdev_us':statistics.stdev(values)})
    row['speedup'] = row['eager_median_us']/row['fused_median_us']
    rows.append(row)
with (ROOT/'deepening/microbenchmark-distribution.csv').open('w',newline='',encoding='utf-8') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ['# Timing distributions', '',
    'Five measurements per path. Range = min–max; SD = sample standard deviation. All latencies in microseconds.',
    'These descriptive summaries are not confidence intervals. Measurements reuse the existing 20-case run.', '',
    '| Tokens | E / k / groups / keep | eps / scale | Eager median [range], SD | Fused median [range], SD | Ratio |',
    '|---:|---|---|---|---|---:|']
for row in rows:
    def cell(name, row=row):
        return f"{row[name+'_median_us']:.3f} [{row[name+'_min_us']:.3f}–{row[name+'_max_us']:.3f}], {row[name+'_sample_stdev_us']:.3f}"
    lines.append(f"| {row['tokens']} | {row['experts']} / {row['topk']} / {row['groups']} / {row['keep']} | {row['eps']} / {row['scale']} | {cell('eager')} | {cell('fused')} | {row['speedup']:.2f}x |")
lines += ['', '## Full-router ablation', '',
    'Includes input cast, router matrix multiplication, and selection. Real first-prompt activations, layer-1 GLM gate. No expert computation.',
    'Timing: CUDA Graph, five rotating-order measurements. Event counts: a separate single eager invocation, not Graph launch counts.', '',
    '| Tokens | Variant | Median [min–max], SD (us) | Captured GPU events |', '|---:|---|---|---:|']
ablation = json.loads((ROOT/'deepening/ablation.json').read_text())
for r in ablation['results']:
    lines.append(f"| {r['tokens']} | {r['variant']} | {r['median_us']:.3f} [{r['min_us']:.3f}–{r['max_us']:.3f}], {r['sample_stdev_us']:.3f} | {r['eager_profile']['cuda_event_count']} |")
lines += ['', 'The profiler records GPU event names in ablation.json, including cast kernels. These counts do not measure host-side CUDA Graph launches.', '']
(ROOT/'deepening/TIMINGS.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote TIMINGS.md and microbenchmark-distribution.csv')
