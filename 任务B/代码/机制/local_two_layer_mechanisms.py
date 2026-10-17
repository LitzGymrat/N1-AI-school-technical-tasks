"""Local-only, frozen 2L1H mechanism tests. Architecture-fixed heads; no network/upload."""
from pathlib import Path
import fcntl
import hashlib
import json
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
from run import bootstrap_ratio, data_manifest

START=time.time()
STATE={}
DEVICE=torch.device('mps')
BATCH=32


def emit(phase,**fields):
    STATE.update(phase=phase,elapsed_seconds=round(time.time()-START,2),**fields)
    core.write_json_atomic(HERE/'status.json',STATE)
    print(json.dumps(STATE,ensure_ascii=False),flush=True)


def parameter_hash(raw):
    h=hashlib.sha256()
    for k,v in raw.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_arrays(path,arrays):
    np.savez_compressed(path,**arrays)


def generate(tk,seed,n,batch=16):
    return {k:v.to(DEVICE) for k,v in core.generate(tk,seed,n,torch.device('cpu'),batch).items()}


def evaluate(raw,data,path,condition,seed):
    pieces=[];t=time.time();total=(len(data['x'])+BATCH-1)//BATCH
    for start,piece in core.chunks(data,BATCH):
        lg=core.logits(raw,piece,DEVICE,'fp32',path,condition,control_seed=902000+seed)
        pieces.append({k:v.detach().cpu().numpy() for k,v in core.query_arrays(lg,piece).items()})
        if (start//BATCH+1)%32==0 or start+len(piece['x'])==len(data['x']):
            emit('evaluating',condition=condition,batch=start//BATCH+1,total_batches=total,
                 condition_elapsed_seconds=round(time.time()-t,2))
    return {k:np.concatenate([r[k] for r in pieces]) for k in pieces[0]}


def restriction_tests(raw,tk,data,path,seed,out):
    out.mkdir(exist_ok=True)
    conditions=['native','no_op_recompute','block_binding_edge','block_retrieval_edge','block_both',
                'binding_control','retrieval_control','zero_retrieval_head_output','unrelated_head_or_layer_control']
    result={'status':'running','path':path,'conditions':{},'data':data_manifest(data,777,BATCH),
            'precision':'fp32','head_choice':'architecture_fixed_before_evaluation; no effect-based selection'}
    base=None;arrays={}
    for condition in conditions:
        target=out/(condition+'.npz')
        if target.exists():
            a=dict(np.load(target));emit('reusing_saved_condition',condition=condition)
        else:
            a=evaluate(raw,data,path,condition,seed);save_arrays(target,a)
        if condition=='native':
            base=a;result['baseline']=core.summarize(a,tk.L)
        else:
            arrays[condition]=a
            result['conditions'][condition]={'scores':core.summarize(a,tk.L),
                'effect':core.paired_effect(base,a,tk.L,970000+seed)}
        if condition=='no_op_recompute':
            diff=abs(float(base['correct'].mean())-float(a['correct'].mean()))*100
            result['no_op_difference_pp']=diff
            result['no_op_acceptance']=diff<=.2
            result['no_op_prediction_disagreements']=int((base['pred']!=a['pred']).sum())
            if not result['no_op_acceptance']:
                result['status']='technical_failure_no_op';core.write_json_atomic(out/'results.json',result)
                raise RuntimeError('No-op changed accuracy by more than 0.2 percentage points')
        core.write_json_atomic(out/'results.json',result)
        emit('condition_completed',condition=condition,long_accuracy=core.summarize(a,tk.L)['long']['accuracy'])
    result['target_minus_matched_control']={
        'binding':core.paired_effect(arrays['binding_control'],arrays['block_binding_edge'],tk.L,970000+seed),
        'retrieval':core.paired_effect(arrays['retrieval_control'],arrays['block_retrieval_edge'],tk.L,970000+seed)}
    coordinates={'query':data['q'].cpu().numpy(),'source':data['s'].cpu().numpy(),'relay':(data['s']+1).cpu().numpy()}
    for name,row in [('binding',data['s']+1),('retrieval',data['q'])]:
        col,valid=core.unrelated_column(data,row,1,902000+seed)
        coordinates[name+'_control_column']=col.cpu().numpy();coordinates[name+'_control_valid']=valid.cpu().numpy()
    save_arrays(out/'intervention_coordinates.npz',coordinates)
    result['status']='completed';core.write_json_atomic(out/'results.json',result)
    return result


def patch_tests(raw,tk,path,seed,out):
    out.mkdir(exist_ok=True)
    clean=generate(tk,903000,512,16)
    long=clean['d']>=tk.L//2
    assert bool(long.any(1).all())
    g=torch.Generator().manual_seed(903001)
    choice=torch.multinomial(long.cpu().float(),1,generator=g).to(DEVICE)
    clean={k:(v.gather(1,choice) if k!='x' else v) for k,v in clean.items()}
    corrupt={k:v.clone() for k,v in clean.items()};bi=torch.arange(512,device=DEVICE)[:,None]
    value=clean['x'][bi,clean['s']+1]
    shift=torch.randint(1,tk.Vv,value.shape,generator=g).to(DEVICE)
    alternative=(value-(tk.Vb+tk.K)+shift)%tk.Vv+tk.Vb+tk.K
    corrupt['x'][bi,clean['s']+1]=alternative;corrupt['x'][bi,clean['q']+1]=alternative
    conditions=['clean_native','corrupt_native','binding_restore','retrieval_restore','unrelated_restore','identity']
    records={k:[] for k in conditions};identity_error=0.
    for start,cl in core.chunks(clean,16):
        co={k:v[start:start+len(cl['x'])] for k,v in corrupt.items()}
        v=value[start:start+len(cl['x'])];alt=alternative[start:start+len(cl['x'])]
        cc={};oc={}
        lc=core.logits(raw,cl,DEVICE,'fp32',path,capture=cc,control_seed=902000+seed)
        lo=core.logits(raw,co,DEVICE,'fp32',path,capture=oc,control_seed=902000+seed)
        patches={
            'binding_restore':{'kind':'residual','layer':0,'location':'relay','values':cc['binding']},
            'retrieval_restore':{'kind':'head','layer':1,'head':0,'location':'query','values':cc['head']},
            'unrelated_restore':{'kind':'residual','layer':0,'location':'unrelated','values':cc['unrelated']},
            'identity':{'kind':'residual','layer':0,'location':'relay','values':oc['binding']}}
        outputs={'clean_native':lc,'corrupt_native':lo}
        for name,patch in patches.items():
            outputs[name]=core.logits(raw,co,DEVICE,'fp32',path,patch=patch,control_seed=902000+seed)
        identity_error=max(identity_error,float((outputs['identity']-lo).abs().max()))
        for name,lg in outputs.items():
            margin=(lg.gather(-1,v[...,None])-lg.gather(-1,alt[...,None])).squeeze(-1);pred=lg.argmax(-1)
            records[name].append({'margin':margin.cpu().numpy(),'pred':pred.cpu().numpy(),
                 'correct_original':(pred==v).cpu().numpy(),'correct_alternative':(pred==alt).cpu().numpy()})
        if (start//16+1)%8==0:emit('activation_restore',batch=start//16+1,total_batches=32)
    arrays={name:{k:np.concatenate([row[k] for row in seq]).reshape(-1) for k in seq[0]} for name,seq in records.items()}
    flat={f'{name}__{k}':v for name,row in arrays.items() for k,v in row.items()}
    flat.update({k:clean[k].cpu().numpy() for k in ['s','q','d']})
    flat['original_value']=value.cpu().numpy();flat['alternative_value']=alternative.cpu().numpy()
    save_arrays(out/'pairs.npz',flat)
    result={'status':'completed','pairs':512,'queries_per_sequence':1,'path':path,
            'seed':903000,'data_generation':'cpu','precision':'fp32',
            'identity_max_logit_error':identity_error,'identity_acceptance':identity_error<=1e-5,
            'clean_sha256':core.tensor_hash(clean),'corrupt_sha256':core.tensor_hash(corrupt),
            'conditions':{},'recovery':{}}
    for name,a in arrays.items():
        result['conditions'][name]={'original_value_accuracy':float(a['correct_original'].mean()),
             'alternative_value_accuracy':float(a['correct_alternative'].mean()),'mean_margin':float(a['margin'].mean())}
    subsets={'all':None,'clean_correct':arrays['clean_native']['correct_original'],
             'clean_correct_and_corrupt_responsive':arrays['clean_native']['correct_original'] & arrays['corrupt_native']['correct_alternative']}
    result['subset_definition']={'primary':'all pairs; no selection on patch recovery',
                                'corrupt_responsive':'corrupt native predicts alternative; secondary only'}
    for name in ['binding_restore','retrieval_restore','unrelated_restore','identity']:
        result['recovery'][name]={sub:bootstrap_ratio(arrays['clean_native']['margin'],arrays['corrupt_native']['margin'],a['margin'],mask,970000+seed)
                                 for sub,mask in subsets.items() for a in [arrays[name]]}
    result['normalized_patch_minus_unrelated_control']={name:{sub:bootstrap_ratio(
        arrays['clean_native']['margin'],arrays['corrupt_native']['margin'],
        arrays['corrupt_native']['margin']+arrays[name]['margin']-arrays['unrelated_restore']['margin'],mask,970000+seed)
        for sub,mask in subsets.items()} for name in ['binding_restore','retrieval_restore']}
    core.write_json_atomic(out/'results.json',result)
    assert result['identity_acceptance'],'Identity restoration changed logits'
    return result


def main():
    lock=(HERE/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.backends.mps.is_available();torch.set_num_threads(4)
    protocol=json.loads((MECH/'tasks.json').read_text())
    benchmark=protocol['common_data']['full_fixed_benchmark']
    assert core.sha256_file(benchmark['path'])==benchmark['sha256']
    data={k:v.long().to(DEVICE) for k,v in torch.load(benchmark['path'],map_location='cpu',weights_only=False).items() if k in ('x','q','s','d')}
    assert len(data['x'])==4096 and data['q'].numel()==131072
    env={'device':'mps','precision':'fp32','torch':torch.__version__,'platform':platform.platform(),
         'transformer_updated':False,'external_upload':False,'batch':32,'patch_batch':16,
         'full_fixed_benchmark':benchmark,'patch_sequences':512,'patch_seed':903000,
         'patch_data_generation':'CPU; same task and budget, not CUDA-identical random stream'}
    core.write_json_atomic(HERE/'environment.json',env)
    emit('started',device='mps',pid=__import__('os').getpid())
    for seed in [0,1,2]:
        out=HERE/f'n2_M02_s{seed}_original64000';out.mkdir(exist_ok=True)
        if (out/'complete.json').exists():emit('reusing_complete_seed',seed=seed);continue
        source_meta=json.loads((ROOT/f'local_two_layer_probes_20261017/n2_M02_s{seed}_original64000/source_manifest.json').read_text())
        source={'path':source_meta['path'],'sha256':source_meta['sha256'],'step':64000,
                'checkpoint_role':'original_training_final_before_any_continuation',
                'wandb_run_id':['f4da27bbced6fd16','b0393e498ee66355','9a8560af90aae292'][seed]}
        raw,tk,_,meta=core.load_model(source,DEVICE);before=parameter_hash(raw)
        assert raw.cfg.n_layer==2 and raw.cfg.n_head==1
        path={'status':'fixed_architecture','delta':1,'binding':{'layer':0,'head':0},
              'retrieval':{'layer':1,'head':0},'other_head':{'layer':0,'head':0},
              'basis':'H1: source value reads previous key in layer1; query reads source value in layer2. Fixed for all seeds before evaluation.',
              'not_empirically_localized':True}
        meta.update(environment=env,parameter_sha256_before=before,
            analysis_sources={str(f.resolve()):core.sha256_file(f) for f in [Path(__file__),MECH/'core.py',MECH/'run.py']})
        core.write_json_atomic(out/'source_manifest.json',meta);core.write_json_atomic(out/'path.json',path)
        emit('model_loaded',seed=seed,step=64000,source_sha256=meta['sha256'])
        ko=restriction_tests(raw,tk,data,path,seed,out/'restrictions')
        patch=patch_tests(raw,tk,path,seed,out/'restoration')
        after=parameter_hash(raw);assert before==after
        assert all(not p.requires_grad and p.grad is None for p in raw.parameters())
        core.write_json_atomic(out/'complete.json',{'status':'completed','seed':seed,
            'restriction_conditions_including_native':9,'restoration_conditions':6,
            'parameter_sha256_before':before,'parameter_sha256_after':after,
            'transformer_parameters_unchanged':True,'elapsed_seconds':time.time()-START,
            'no_op_acceptance':ko['no_op_acceptance'],'identity_acceptance':patch['identity_acceptance']})
        emit('seed_completed',seed=seed)
        del raw;torch.mps.empty_cache()
    emit('completed',seeds=[0,1,2],transformer_updated=False)
    core.write_json_atomic(HERE/'complete.json',STATE)


if __name__=='__main__':
    try:main()
    except Exception as e:
        emit('failed',error=type(e).__name__+': '+str(e));traceback.print_exc();raise
