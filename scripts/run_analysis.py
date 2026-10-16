from pathlib import Path
import argparse,hashlib,sys
p=argparse.ArgumentParser(description='CPU-only reproduction of locked final600 analysis; requires external pinned evidence.')
p.add_argument('--evidence',required=True,type=Path)
p.add_argument('--output',default='analysis_outputs',type=Path)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
engine=root/'analysis/analysis_engine.py'
expected='d01baea11c6dbb7b455a94183852c305d3a5a8e55f9867c35b0c2717deb1674a'
if hashlib.sha256(engine.read_bytes()).hexdigest()!=expected:
    raise SystemExit('Pinned analysis engine hash mismatch')
sys.path.insert(0,str(engine.parent))
import analysis_engine
r=analysis_engine.build_local_analysis(a.evidence,a.output)
print('Local analysis bundle:',r['bundle'])
print('This command does not publish to Drive; use Notebook09 for that workflow.')
