"""Administrative D50 adapter; acquisition/emulation functions remain verified source bytes."""
import hashlib, json, sys
from pathlib import Path

def adapt_membership(source):
    old = "demo=request['demo'];assert demo in train, 'Source pilot is restricted to one training demo.'"
    new = """demo=request['demo']
    role=request['dataset_role']
    assert role in ['train','validation'], 'Explicit D50 dataset role required.'
    assert demo in (train if role=='train' else val), 'Demo does not belong to the declared split role.'"""
    assert source.count(old) == 1, 'Verified membership adapter point changed.'
    source=source.replace(old,new)
    old_metadata="'scene_contract':'each demonstration uses its own recorded XML; this pilot validates one train demo only',"
    new_metadata="'dataset_role':role,'scene_contract':'each demonstration uses its own recorded XML; explicit locked D50 train/validation role',"
    assert source.count(old_metadata)==1, 'Verified provenance adapter point changed.'
    return source.replace(old_metadata,new_metadata)

def main(request):
    path=Path(request['verified_source_worker'])
    source=path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == request['verified_source_sha256']
    ctx={'__name__':'d50_source_administrative_adapter'}
    exec(compile(adapt_membership(source),str(path),'exec'),ctx)
    return ctx['main'](request)

if __name__=='__main__': main(json.loads(Path(sys.argv[1]).read_text()))
