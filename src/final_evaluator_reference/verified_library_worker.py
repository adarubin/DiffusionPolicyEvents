"""Load an existing local EGL vendor library in a disposable process. No graphics import."""
import ctypes,hashlib,json,os,sys,time,traceback
from pathlib import Path

def inspect_library(path):
    p=Path(path);assert p.is_file(), 'Selected local library is missing.'
    assert p.name.startswith('libEGL_nvidia.so'), 'Only an existing NVIDIA EGL vendor library may be loaded.'
    library=ctypes.CDLL(str(p),mode=getattr(os,'RTLD_NOW',2)|getattr(os,'RTLD_LOCAL',0))
    entry=getattr(library,'__egl_Main',None)
    assert entry is not None, 'Loaded library lacks the GLVND EGL vendor entry point.'
    maps=Path('/proc/self/maps')
    loaded=sorted({line.split()[-1] for line in maps.read_text().splitlines() if '/' in line and any(k in line.lower() for k in ['nvidia','libegl','glvnd'])}) if maps.exists() else []
    return {'status':'LOADABLE','library':str(p),'resolved_path':str(p.resolve()),
      'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'egl_vendor_entry_point':True,'loaded_graphics_paths':loaded,
      'scope':'dynamic library load only; not proof of EGL device availability or NVIDIA rendering'}

def main(request):
    folder=Path(request['output']);folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    result={'id':request['id'],'task':request['task'],'status':'RUNNING'}
    try:result.update(inspect_library(request['library']))
    except Exception as exc:result.update(status='LOAD_FAILED',error=str(exc),traceback=traceback.format_exc())
    result['elapsed_seconds']=time.perf_counter()-started
    (folder/'worker_result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

if __name__=='__main__':main(json.loads(Path(sys.argv[1]).read_text()))
