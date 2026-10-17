"""Run the registered M8 fixed-tap affine probes on Mac; never update Transformer weights."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import platform
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
MECH=ROOT/'night_20261016/mechanism'
sys.path.insert(0,str(MECH))
import numpy as np
import torch
import torch.nn.functional as F
import core
import probes

START=time.time()
STATE={}
FIT_INDEX=0
RUN=None
WEIGHTS={}


def emit(phase,**values):
    STATE.update(phase=phase,elapsed_seconds=round(time.time()-START,2),**values)
    core.write_json_atomic(HERE/'status.json',STATE)
    print(json.dumps(STATE,ensure_ascii=False),flush=True)


def parameter_hash(raw):
    h=hashlib.sha256()
    for name,p in raw.state_dict().items():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


original_generate=core.generate
original_features=probes.features


def cpu_generated(tk,seed,sequences,device,batch=32,only_long=False):
    # torch.Generator does not support MPS. Same task, sizes, range, and seeds;
    # CPU generation is explicitly recorded rather than called CUDA-identical.
    data=original_generate(tk,seed,sequences,torch.device('cpu'),batch,only_long)
    return {k:v.to(device) for k,v in data.items()}


def capture_features(raw,data,path,device,precision,batch,fixed=False):
    emit('extracting_frozen_features',split_sequences=len(data['x']))
    values=original_features(raw,data,path,device,precision,batch,fixed)
    banks,labels,_=values
    with torch.no_grad():
        h=banks['query_residual']
        lg=raw.head(raw.nf(h)).float()
        target=labels['value'];correct=lg.argmax(-1)==target
        long=labels['distance']>=raw_task.L//2
        readout={'queries':len(target),'accuracy':float(correct.float().mean()),
                 'long_queries':int(long.sum()),'long_accuracy':float(correct[long].float().mean())}
    NATIVE[len(data['x'])]=readout
    emit('features_ready',native_readout=readout)
    return values


def fitted(X,y,classes,steps=400):
    global FIT_INDEX
    ix=FIT_INDEX;FIT_INDEX+=1
    taps=['source_embedding','relay_residual','query_embedding','query_residual']
    tap=taps[ix//4];label='key' if ix%4<2 else 'value'
    control='true_labels' if ix%2==0 else 'permuted_labels'
    name=f'{tap}/{label}/{control}'
    t=time.time()
    with torch.enable_grad():
        W=torch.zeros(X.shape[1],classes,device=X.device,requires_grad=True)
        b=torch.zeros(classes,device=X.device,requires_grad=True)
        opt=torch.optim.Adam([W,b],lr=.01,weight_decay=1e-4)
        for step in range(1,steps+1):
            opt.zero_grad();loss=F.cross_entropy(X@W+b,y);loss.backward();opt.step()
            if step==1 or step%100==0:
                value=float(loss.detach())
                emit('fitting_affine_probe',probe_index=ix+1,probe=name,probe_step=step,loss=value,
                     fit_elapsed_seconds=round(time.time()-t,2))
    WEIGHTS[name]={'W':W.detach().cpu(),'b':b.detach().cpu()}
    torch.save(WEIGHTS,CURRENT/'probe_weights.pt')
    return W.detach(),b.detach()


def main():
    global RUN,FIT_INDEX,WEIGHTS,CURRENT,NATIVE,raw_task
    lock=(HERE/'run.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    device=torch.device('mps')
    assert torch.backends.mps.is_available()
    env={'recording_mode':'local_only','device':str(device),'torch':torch.__version__,'python':platform.python_version(),
         'platform':platform.platform(),'model_precision':'fp32','data_generation':'cpu',
         'feature_batch':32,'probe_full_batch':16384,'probe_steps':400,'transformer_updated':False,
         'split_seeds':{'train':920000,'valid':930000,'test':940000},
         'split_sequences':{'train':512,'valid':128,'test':128},
         'note':'Same affine formulation as four-layer probes. CPU samples and FP32 MPS forwards differ from CUDA/BF16.'}
    core.write_json_atomic(HERE/'environment.json',env)
    probes.generate=cpu_generated;probes.features=capture_features;probes.fit_probe=fitted
    emit('starting',pid=os.getpid(),device=str(device))
    for seed in [0,1,2]:
        CURRENT=HERE/f'n2_M02_s{seed}_original64000';CURRENT.mkdir(exist_ok=True)
        if (CURRENT/'complete.json').exists():
            emit('reusing_complete_seed',seed=seed);continue
        FIT_INDEX=0;WEIGHTS={};NATIVE={}
        if seed in [0,1]:
            p=ROOT/f'continuation_m02_100k_20261016/source_seeds/n2_M02_s{seed}/final_resume.pt'
            expected=['304b54a1f8523b48285669e5d22b10b74b7aa980b309a7079209ce2d506dd69c',
                      'eb3cd79bbd79f4702e64abca5958684863156a2c814418449d181360e5a04ec1'][seed]
        else:
            receipt=json.loads((HERE/'seed2_source.json').read_text())
            p=Path(receipt['path']);expected=receipt['sha256']
        sha=core.sha256_file(p)
        assert sha==expected
        source={'path':str(p),'sha256':sha,'step':64000,
                'checkpoint_role':'original_training_final_before_any_continuation',
                'wandb_run_id':['f4da27bbced6fd16','b0393e498ee66355','9a8560af90aae292'][seed]}
        raw,tk,_,metadata=core.load_model(source,device);raw_task=tk
        assert raw.cfg.n_layer==2 and raw.cfg.n_head==1
        before=parameter_hash(raw)
        metadata.update(execution=env,parameter_sha256_before=before,
                        analysis_sources={str(f):core.sha256_file(f) for f in [Path(__file__),MECH/'probes.py',MECH/'core.py']})
        core.write_json_atomic(CURRENT/'source_manifest.json',metadata)
        emit('loaded_exact_original_checkpoint',seed=seed,step=64000,weight_sha256=sha)
        # One small FP32 CPU/MPS parity check on the same input, before any classifier fit.
        fixture=original_generate(tk,919999,2,torch.device('cpu'),2)
        cpu_model,_,_,_=core.load_model(source,torch.device('cpu'))
        cpu_logits=core.logits(cpu_model,fixture,torch.device('cpu'),'fp32')
        mps_data={k:v.to(device) for k,v in fixture.items()}
        mps_logits=core.logits(raw,mps_data,device,'fp32').cpu()
        parity={'max_abs_logit_error':float((cpu_logits-mps_logits).abs().max()),
                'argmax_agreement':float((cpu_logits.argmax(-1)==mps_logits.argmax(-1)).float().mean())}
        assert parity['argmax_agreement']==1.0,parity
        core.write_json_atomic(CURRENT/'cpu_mps_parity.json',parity)
        del cpu_model,cpu_logits,mps_logits
        core.write_json_atomic(CURRENT/'recording.json',{'mode':'local_only','external_upload':False})
        result=probes.run_probes(raw,tk,{},CURRENT/'probe',device,'fp32',32,seed,fixed=True)
        after=parameter_hash(raw)
        assert before==after
        result.update(parameter_sha256_before=before,parameter_sha256_after=after,
                      transformer_parameters_unchanged=True,execution=env)
        core.write_json_atomic(CURRENT/'probe/probe_metrics.json',result)
        metrics={}
        for name,row in result['probes'].items():
            for split in ['valid','test']:
                metrics[f'{name}/{split}_accuracy']=row['splits'][split]['accuracy']
        core.write_json_atomic(CURRENT/'probe_summary.json',metrics)
        core.write_json_atomic(CURRENT/'native_readout.json',NATIVE)
        core.write_json_atomic(CURRENT/'complete.json',{'status':'completed','seed':seed,
          'transformer_parameters_unchanged':True,'classifiers':FIT_INDEX,'steps_per_classifier':400,
          'elapsed_seconds':time.time()-START,'source_sha256':sha})
        emit('seed_completed',seed=seed,classifiers=FIT_INDEX)
        del raw;torch.mps.empty_cache()
    emit('completed',seeds=[0,1,2],transformer_updated=False)
    core.write_json_atomic(HERE/'complete.json',STATE)


if __name__=='__main__':
    try:main()
    except Exception as e:
        emit('failed',error=type(e).__name__+': '+str(e))
        if RUN:RUN.finish(exit_code=1)
        traceback.print_exc();raise
