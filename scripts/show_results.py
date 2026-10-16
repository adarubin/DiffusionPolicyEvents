from pathlib import Path
import csv
path=Path(__file__).resolve().parents[1]/'results/tables/aggregate.csv'
with path.open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
assert sum(int(r['trials']) for r in rows)==600
for r in rows:
    n=int(r['trials']);k=int(r['sustained_success'])
    print(f"{r['condition_id']:16} {r['modality']:6} {k}/{n} ({100*k/n:.2f}%)")
print('Summary-table inspection only; full step-level verification requires the pinned core evidence archive.')
