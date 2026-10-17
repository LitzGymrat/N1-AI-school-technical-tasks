"""Execute the remaining registered frozen analyses locally. No network or central-state writes."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MECH=ROOT/'night_20261016/mechanism'
sys.path.insert(0,str(MECH))
import core
import probes
import run as runner

START=time.time();STATE={};LOADED=[]
ORIGINAL_GENERATE=core.generate
ORIGINAL_LOAD=core.load_model


def emit(phase,**kwargs):
    STATE.update(phase=phase,pid=os.getpid(),**kwargs)
    STATE["elapsed_seconds"]=round(time.time()-START,2)
    core.write_json_atomic(HERE/'status.json',STATE)
    print(json.dumps(STATE,ensure_ascii=False),flush=True)


def parameter_hash(raw):
    digest=hashlib.sha256()
    for name,p in raw.state_dict().items():
        digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate(tk,seed,n,device,batch=32,only_long=False):
    values=ORIGINAL_GENERATE(tk,seed,n,torch.device('cpu'),batch,only_long)
    return {k:v.to(device) for k,v in values.items()}


def load_model(source,device):
    raw,tk,precision,meta=ORIGINAL_LOAD(source,device)
    LOADED.append((raw,parameter_hash(raw),meta['sha256']))
    meta.update(checkpoint_config_precision=precision,precision='fp32',
                local_analysis_source_sha256=core.sha256_file(Path(__file__)),
                local_execution='Mac Apple GPU; CPU-generated analysis samples; FP32; no external upload')
    return raw,tk,'fp32',meta


def evaluate(raw,data,device,precision,batch,path=None,condition='native',control_seed=902000):
    rows=[];t=time.time();total=(len(data['x'])+batch-1)//batch
    for start,piece in core.chunks(data,batch):
        lg=core.logits(raw,piece,device,precision,path,condition,control_seed=control_seed)
        rows.append({k:v.detach().cpu().numpy() for k,v in core.query_arrays(lg,piece).items()})
        if (start//batch+1)%32==0 or start+len(piece['x'])==len(data['x']):
            emit('evaluating',condition=condition,batch=start//batch+1,total_batches=total,
                 condition_elapsed_seconds=round(time.time()-t,2))
    return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}


def main():
    lock=(HERE/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.backends.mps.is_available();torch.set_num_threads(4)
    core.write_json_atomic(HERE/'environment.json',{'torch':torch.__version__,'python':platform.python_version(),
        'platform':platform.platform(),'device':'mps','precision':'fp32','data_generation':'cpu',
        'external_upload':False,'transformer_updated':False,'source':str(MECH),
        'analysis_sources':{str(p):core.sha256_file(p) for p in [Path(__file__),MECH/'run.py',MECH/'core.py',MECH/'probes.py']}})
    first=ROOT/'local_two_layer_mechanisms_20261017'
    while not (first/'complete.json').exists():
        state=json.loads((first/'status.json').read_text()) if (first/'status.json').exists() else {}
        if state.get('phase')=='failed':raise RuntimeError('Two-layer predecessor failed: '+str(state.get('error')))
        emit('waiting_for_two_layer_gpu',predecessor_phase=state.get('phase'),predecessor_seed=state.get('seed'))
        time.sleep(10)
    core.generate=generate;probes.generate=generate;runner.generate=generate
    runner.evaluate=evaluate;runner.load_model=load_model
    manifest=HERE/'manifest.json';spec=json.loads(manifest.read_text());jobs=spec['jobs']
    completed=[];failed=[]
    for ix,job in enumerate(jobs):
        out=Path(job['output_dir']);out.mkdir(parents=True,exist_ok=True)
        if (out/'package_complete.json').exists():completed.append(job['job_id']);continue
        LOADED.clear();emit('starting_package',job=job['job_id'],index=ix+1,total=len(jobs),completed=len(completed))
        try:
            sys.argv=['run.py','--manifest',str(manifest),'--job',job['job_id'],'--device','mps','--output',str(out)]
            runner.main()
            verification=[]
            for raw,before,sha in LOADED:
                after=parameter_hash(raw)
                assert before==after and all(not p.requires_grad and p.grad is None for p in raw.parameters())
                verification.append({'source_sha256':sha,'parameter_sha256_before':before,'parameter_sha256_after':after})
            core.write_json_atomic(out/'local_frozen_verification.json',{'status':'passed','checks':verification,'external_upload':False})
            completed.append(job['job_id']);emit('package_completed',job=job['job_id'],completed=len(completed))
        except Exception as exc:
            error={'job':job['job_id'],'error':type(exc).__name__+': '+str(exc)};failed.append(error)
            (out/'local_failure.txt').write_text(traceback.format_exc())
            emit('package_failed',**error)
        finally:
            LOADED.clear();torch.mps.empty_cache()
            core.write_json_atomic(HERE/'progress.json',{'completed':completed,'failed':failed,'total':len(jobs)})
    final={'status':'completed' if not failed else 'incomplete','completed':completed,'failed':failed,
           'total':len(jobs),'elapsed_seconds':time.time()-START,'transformer_updated':False}
    core.write_json_atomic(HERE/'finish.json',final)
    emit('finished',**final)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        emit('failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc();raise
