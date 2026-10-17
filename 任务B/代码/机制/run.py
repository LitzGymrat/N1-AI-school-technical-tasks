"""Run one registered frozen-model package. No automatic training or GPU launch on import."""
import argparse
import copy
import csv
import json
import math
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from core import (load_model,generate,select_path,attention_profile,evaluate,summarize,paired_effect,
                  logits,query_arrays,chunks,tensor_hash,sha256_file,write_json_atomic,unrelated_column,amp)
from probes import run_probes

HERE=Path(__file__).resolve().parent


def save_arrays(out,name,arrays):
    np.savez_compressed(Path(out)/(name+'.npz'),**arrays)


def data_manifest(data,seed,batch):
    return {'seed':seed,'sha256':tensor_hash(data),'n_sequences':len(data['x']),
      'n_queries':int(data['q'].numel()),'batch_size':batch,'batches':math.ceil(len(data['x'])/batch),
      'distance_histogram':torch.bincount(data['d'].flatten(),minlength=data['x'].shape[1]).cpu().tolist()}


def load_eval(manifest,tk,device,n=None):
    source=manifest['common_data']['full_fixed_benchmark'];path=Path(source['path'])
    if sha256_file(path)!=source['sha256']:raise ValueError('Fixed evaluation benchmark hash mismatch')
    raw=torch.load(path,map_location='cpu',weights_only=False)
    data={k:v[:n].long().to(device) if n else v.long().to(device) for k,v in raw.items() if k in ('x','q','s','d')}
    if data['x'].shape[1]!=tk.L or data['q'].shape[1]!=tk.n_q:raise ValueError('Evaluation geometry mismatch')
    return data


def stage_m1(raw,tk,precision,device,batch,out,manifest,selection_n=64,diag_n=128,evaluation=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    selection_data=generate(tk,900000,selection_n,device,batch)
    check={k:v[:batch] for k,v in selection_data.items()}
    with torch.no_grad(),amp(device,precision):
        full=raw(check['x'],raw.block_mask(tk.L,device))
        full=full.gather(1,check['q'][...,None].expand(-1,-1,full.shape[-1])).float()
    wrapped=logits(raw,check,device,precision)
    wrapper_error=float((full-wrapped).abs().max())
    if not torch.allclose(full,wrapped,rtol=1e-4,atol=1e-5):raise RuntimeError('Query wrapper differs from full native forward')
    del full,wrapped,check
    path=select_path(raw,selection_data,device,precision,batch)
    write_json_atomic(out/'selection.json',path) # fixed before evaluation is ever loaded/scored
    diag_data=generate(tk,901000,diag_n,device,batch)
    diagnostic=attention_profile(raw,diag_data,device,precision,batch,binding_delta=path['delta'])
    write_json_atomic(out/'attention.json',diagnostic)
    data=evaluation if evaluation is not None else load_eval(manifest,tk,device)
    native=evaluate(raw,data,device,precision,batch)
    noop=evaluate(raw,data,device,precision,batch,path,'no_op_recompute')
    save_arrays(out,'native',native);save_arrays(out,'no_op_recompute',noop)
    no_op=summarize(noop,tk.L);base=summarize(native,tk.L)
    difference=abs(base['all']['accuracy']-no_op['all']['accuracy'])
    result={'status':'completed','wrapper_max_logit_error':wrapper_error,'baseline':base,'no_op':no_op,'no_op_difference_pp':100*difference,
      'no_op_acceptance':difference<=.002,'prediction_disagreement':int((native['pred']!=noop['pred']).sum()),
      'maximum_margin_difference':float(np.max(np.abs(native['margin']-noop['margin']))),
      'path_status':path['status'],'data':{'selection':data_manifest(selection_data,900000,batch),
      'diagnostic':data_manifest(diag_data,901000,batch),'evaluation':data_manifest(data,777 if evaluation is None else 904000,batch)}}
    write_json_atomic(out/'results.json',result)
    if not result['no_op_acceptance']:raise RuntimeError('No-op tolerance failed; mechanism interpretation stopped')
    return path,result


def stage_ko(raw,tk,precision,device,batch,out,data,path,model_seed,kind='full'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    native=evaluate(raw,data,device,precision,batch);save_arrays(out,'native',native)
    result={'status':'completed','baseline':summarize(native,tk.L),'path':path,'conditions':{},
            'data':data_manifest(data,777 if tk.L==1024 else 904000,batch),
            'control_note':'binding_control and retrieval_control are separate matched subconditions of unrelated_position_control'}
    if path.get('status')!='localized':
        result['status']='NA_unlocalized';write_json_atomic(out/'results.json',result);return result
    conditions=['no_op_recompute','block_binding_edge','block_retrieval_edge','binding_control','retrieval_control']
    if kind in ('full','long'):conditions.insert(3,'block_both')
    if kind=='full':conditions+=['zero_retrieval_head_output','unrelated_head_or_layer_control']
    arrays={}
    for condition in conditions:
        if condition=='unrelated_head_or_layer_control' and not path.get('other_head'):
            result['conditions'][condition]={'status':'NA_no_other_head'};continue
        a=evaluate(raw,data,device,precision,batch,path,condition,control_seed=902000+model_seed);arrays[condition]=a;save_arrays(out,condition,a)
        result['conditions'][condition]={'scores':summarize(a,tk.L),'effect':paired_effect(native,a,tk.L,970000+model_seed)}
        if condition=='no_op_recompute':
            diff=abs(result['baseline']['all']['accuracy']-result['conditions'][condition]['scores']['all']['accuracy'])
            result['no_op_difference_pp']=100*diff;result['no_op_acceptance']=diff<=.002
            if diff>.002:
                result['status']='technical_failure_no_op';write_json_atomic(out/'results.json',result)
                raise RuntimeError('No-op recomputation exceeds 0.2 percentage points')
        write_json_atomic(out/'results.json',result)
    result['target_minus_matched_control']={
      'binding':paired_effect(arrays['binding_control'],arrays['block_binding_edge'],tk.L,970000+model_seed),
      'retrieval':paired_effect(arrays['retrieval_control'],arrays['block_retrieval_edge'],tk.L,970000+model_seed)}
    coordinates={'query':data['q'].cpu().numpy(),'source':data['s'].cpu().numpy(),'relay':(data['s']+path['delta']).cpu().numpy()}
    for kind,row in [('binding',data['s']+path['delta']),('retrieval',data['q'])]:
        col,valid=unrelated_column(data,row,path['delta'],902000+model_seed)
        coordinates[kind+'_control_column']=col.cpu().numpy();coordinates[kind+'_control_valid']=valid.cpu().numpy()
    save_arrays(out,'intervention_coordinates',coordinates)
    write_json_atomic(out/'results.json',result);return result


def bootstrap_ratio(clean,corrupt,patched,mask=None,seed=970000):
    if mask is not None:clean,corrupt,patched=clean[mask],corrupt[mask],patched[mask]
    if not len(clean):return {'R':None,'ci95':None,'n':0}
    denominator=float((clean-corrupt).mean())
    if abs(denominator)<1e-6:return {'R':None,'ci95':None,'n':len(clean),'denominator':denominator}
    ix=np.random.default_rng(seed).integers(len(clean),size=(1000,len(clean)))
    den=(clean-corrupt)[ix].mean(1);num=(patched-corrupt)[ix].mean(1)
    den_ci=np.quantile(den,[.025,.975]).tolist()
    if den_ci[0]<=0<=den_ci[1]:
        return {'R':None,'ci95':None,'n':len(clean),'denominator':denominator,'denominator_ci95':den_ci,
                'NA_reason':'clean_minus_corrupt_margin_not_separated_from_zero'}
    ratios=np.divide(num,den,out=np.full(1000,np.nan),where=np.abs(den)>=1e-6)
    return {'R':float((patched-corrupt).mean()/denominator),'ci95':np.nanquantile(ratios,[.025,.975]).tolist(),
            'n':len(clean),'denominator':denominator,'denominator_ci95':den_ci,'not_clipped':True}


def stage_patch(raw,tk,precision,device,out,path,model_seed):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if path.get('status')!='localized':
        result={'status':'NA_unlocalized'};write_json_atomic(out/'results.json',result);return result
    clean=generate(tk,903000,512,device,16)
    long=clean['d']>=tk.L//2
    if not bool(long.any(1).all()):raise RuntimeError('Patch set has a sequence without long query; report before changing dataset')
    g=torch.Generator().manual_seed(903001)
    choice=torch.multinomial(long.cpu().float(),1,generator=g).to(device)
    clean={k:(v.gather(1,choice) if k!='x' else v) for k,v in clean.items()}
    corrupt={k:v.clone() for k,v in clean.items()};bi=torch.arange(512,device=device)[:,None]
    value=clean['x'][bi,clean['s']+1]
    shift=torch.randint(1,tk.Vv,value.shape,generator=g).to(device)
    alternative=((value-(tk.Vb+tk.K)+shift)%tk.Vv)+(tk.Vb+tk.K)
    corrupt['x'][bi,clean['s']+1]=alternative;corrupt['x'][bi,clean['q']+1]=alternative
    conditions=['clean_native','corrupt_native','binding_restore','retrieval_restore','unrelated_restore','identity']
    records={k:[] for k in conditions};max_identity_error=0.0
    for start,cl in chunks(clean,16):
        co={k:v[start:start+len(cl['x'])] for k,v in corrupt.items()};v=value[start:start+len(cl['x'])];alt=alternative[start:start+len(cl['x'])]
        clean_cache={};corrupt_cache={}
        lc=logits(raw,cl,device,precision,path,capture=clean_cache,control_seed=902000+model_seed)
        lo=logits(raw,co,device,precision,path,capture=corrupt_cache,control_seed=902000+model_seed)
        patches={
          'binding_restore':{'kind':'residual','layer':path['binding']['layer'],'location':'relay','values':clean_cache['binding']},
          'retrieval_restore':{'kind':'head','layer':path['retrieval']['layer'],'head':path['retrieval']['head'],'location':'query','values':clean_cache['head']},
          'unrelated_restore':{'kind':'residual','layer':path['binding']['layer'],'location':'unrelated','values':clean_cache['unrelated']},
          'identity':{'kind':'residual','layer':path['binding']['layer'],'location':'relay','values':corrupt_cache['binding']}}
        outputs={'clean_native':lc,'corrupt_native':lo}
        for name,patch in patches.items():outputs[name]=logits(raw,co,device,precision,path,patch=patch,control_seed=902000+model_seed)
        max_identity_error=max(max_identity_error,float((outputs['identity']-lo).abs().max()))
        for name,lg in outputs.items():
            margin=(lg.gather(-1,v[...,None])-lg.gather(-1,alt[...,None])).squeeze(-1)
            pred=lg.argmax(-1)
            records[name].append({'margin':margin.cpu().numpy(),'pred':pred.cpu().numpy(),
              'correct_original':(pred==v).cpu().numpy(),'correct_alternative':(pred==alt).cpu().numpy()})
    arrays={name:{k:np.concatenate([row[k] for row in seq]).reshape(-1) for k in seq[0]} for name,seq in records.items()}
    flat={f'{name}__{k}':v for name,row in arrays.items() for k,v in row.items()}
    flat.update({k:clean[k].cpu().numpy() for k in ['s','q','d']});flat['original_value']=value.cpu().numpy();flat['alternative_value']=alternative.cpu().numpy()
    save_arrays(out,'pairs',flat)
    result={'status':'completed','pairs':512,'selected_queries_per_sequence':1,'identity_max_logit_error':max_identity_error,
      'identity_acceptance':max_identity_error<=1e-5,'clean_sha256':tensor_hash(clean),'corrupt_sha256':tensor_hash(corrupt),
      'seed':903000,'path':path,'conditions':{},'recovery':{}}
    for name,a in arrays.items():result['conditions'][name]={'original_value_accuracy':float(a['correct_original'].mean()),
       'alternative_value_accuracy':float(a['correct_alternative'].mean()),'mean_margin':float(a['margin'].mean())}
    subsets={'all':None,'clean_correct':arrays['clean_native']['correct_original'],
      'clean_correct_and_corrupt_responsive':arrays['clean_native']['correct_original'] & arrays['corrupt_native']['correct_alternative']}
    result['subset_definition']={'corrupt_responsive':'native corrupt predicts the alternative value; secondary subset only',
      'primary':'all pairs; no selection on patch recovery'}
    for name in ['binding_restore','retrieval_restore','unrelated_restore','identity']:
        result['recovery'][name]={subset:bootstrap_ratio(arrays['clean_native']['margin'],arrays['corrupt_native']['margin'],
          arrays[name]['margin'],mask,970000+model_seed) for subset,mask in subsets.items()}
    result['normalized_patch_minus_unrelated_control']={name:{subset:bootstrap_ratio(
      arrays['clean_native']['margin'],arrays['corrupt_native']['margin'],
      arrays['corrupt_native']['margin']+arrays[name]['margin']-arrays['unrelated_restore']['margin'],mask,970000+model_seed)
      for subset,mask in subsets.items()} for name in ['binding_restore','retrieval_restore']}
    write_json_atomic(out/'results.json',result)
    if not result['identity_acceptance']:raise RuntimeError('Identity patch changed logits')
    return result


def stage_m5(raw,tk,precision,device,batch,out,manifest,path,seed,job):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    diag=generate(tk,901000,128,device,batch)
    profile=attention_profile(raw,diag,device,precision,batch,binding_delta=path['delta'])
    write_json_atomic(out/'fixed_path_diagnostics.json',profile)
    result=stage_ko(raw,tk,precision,device,batch,out,load_eval(manifest,tk,device,512),path,seed,'sparse')
    fixed={'binding_mass':None,'retrieval_mass':None}
    if path.get('binding'):
        fixed['binding_mass']=profile['binding_mass'][path['binding']['layer']][path['binding']['head']]
    if path.get('retrieval'):
        fixed['retrieval_mass']=profile['relay_mass'][path['retrieval']['layer']][path['retrieval']['head']][path['delta']-1]
    timeline={'checkpoint_step':job.get('source',{}).get('step'),
      'fixed_path_attention':fixed,'observed_training_metrics_at_step':job.get('source',{}).get('observed_training_metrics_at_step'),
      'first_long_90_step':job.get('first_long_90_step'),'relative_step':job.get('relative_step'),
      'checkpoint_roles':job.get('checkpoint_roles'),'missing_midpoint_note':job.get('missing_midpoint_note'),
      'training_curve':job.get('source',{}).get('source_metrics_path'),'path_reselected':False}
    curve=job.get('source',{}).get('source_metrics_path')
    if curve and Path(curve).is_file():
        rows=[json.loads(line) for line in Path(curve).read_text().splitlines() if line.strip()]
        first90=job.get('first_long_90_step')
        rows=[{**row,'relative_step':row['step']-first90 if first90 is not None else None}
              for row in rows if 'step' in row and ('valid_loss' in row or 'bench_long' in row)]
        write_json_atomic(out/'training_curve_aligned.json',{'path':curve,'sha256':sha256_file(curve),'rows':rows})
    else:timeline['training_curve_status']='not_available_at_registered_path'
    write_json_atomic(out/'timeline_point.json',timeline)
    return result


def reuse_m6(job,out):
    src=job['source'];p=Path(src['path'])
    if sha256_file(p)!=src['sha256']:raise ValueError('M6 original result changed')
    provenance=[]
    for source in src['checkpoint_sources']:
        cp=Path(source['path'])
        if not cp.is_file() or sha256_file(cp)!=source['sha256']:raise ValueError('M6 checkpoint provenance mismatch: '+str(cp))
        provenance.append({'path':str(cp),'sha256':source['sha256']})
    original=json.loads(p.read_text());rows=[];curve_index={}
    for name,model in original.items():
        for condition,result in model['knockout_512seq'].items():
            rows.append({'model':name,'condition':condition,**{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in result.items()}})
        curve_index[name]=model.get('bench_delta_curve_131072q')
    if len(rows)!=48:raise ValueError('Expected exactly 48 archived KO records')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'reused_48_KO_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for row in rows for k in row}));w.writeheader();w.writerows(rows)
    write_json_atomic(out/'per_distance_curve_index.json',curve_index)
    report={'status':'reused_existing_results','source':src,'model_inference_performed':False,'conditions':48,
      'scope':'Archived KO masks all heads in its recorded last layers at query rows; it is not single-head localization.',
      'denominator':'512 sequences per archived KO; do not pool with new 4096-sequence evaluations',
      'original_results':original,'verified_checkpoint_provenance':provenance}
    write_json_atomic(out/'results.json',report);return report


def selection_for(job,out):
    for stage in job.get('stages',[]):
        candidate=stage.get('selection_artifact')
        if candidate:
            p=Path(candidate)
            if not p.is_absolute():p=Path(out)/p
            if p.is_file():return json.loads(p.read_text())
    raise FileNotFoundError('Run the registered M1 dependency first; do not reselect a path on evaluation data')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=HERE/'tasks.json');ap.add_argument('--job',required=True)
    ap.add_argument('--device',default='cuda');ap.add_argument('--output',type=Path);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    manifest=json.loads(args.manifest.read_text());job=next(j for j in manifest['jobs'] if j['job_id']==args.job)
    if args.validate_only:
        print(json.dumps({'job':job['job_id'],'input_status':job['input_status'],'source':job['source'],'dependencies':job.get('depends_on',[])},ensure_ascii=False));return
    if job['input_status']!='ready_input':raise FileNotFoundError(json.dumps({'job':job['job_id'],'status':job['input_status'],'source':job['source']},ensure_ascii=False))
    out=args.output or Path(job['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if (out/'package_complete.json').exists():raise FileExistsError('Package already completed; existing result is reusable')
    write_json_atomic(out/'registered_job.json',job)
    if job['task_family']=='M6':
        result=reuse_m6(job,out);write_json_atomic(out/'package_complete.json',{'status':result['status']});return
    device=torch.device(args.device);seed=job['training_seed'];started=time.time()
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
    source=job['sources']['final'] if job['task_family']=='M9' else job['source']
    raw,tk,precision,metadata=load_model(source,device)
    if device.type=='cuda' and raw.cfg.pos=='alibi':
        # Compile only the reused FlexAttention forward kernel; avoid dense LxL score storage.
        from lrexp import model as runtime_model
        runtime_model.flex_attention=torch.compile(runtime_model.flex_attention,dynamic=False)
        metadata['alibi_execution']='compiled native FlexAttention forward; Transformer frozen'
    metadata['analysis_source_hashes']={f.name:sha256_file(f) for f in HERE.glob('*.py')}
    metadata['analysis_manifest_sha256']=sha256_file(args.manifest)
    batch=max(1,min(32,32768//tk.L))
    write_json_atomic(out/'resolved_source_manifest.json',metadata)
    write_json_atomic(out/'config.json',{'model':metadata['model'],'task':metadata['task'],
      'precision':precision,'job_id':job['job_id'],'transformer_frozen':True})
    write_json_atomic(out/'environment.json',{'torch':torch.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'python':platform.python_version(),'batch_size':batch,'transformer_frozen':True})
    family=job['task_family'];results={};path=None
    if family in ('M1','M7','M8','M9','LONG'):
        m1out=out if family in ('M1','LONG') else out/job['stages'][0]['stage_id']
        if family=='LONG':
            ev=generate(tk,904000,65536//tk.n_q,device,batch)
            path,results['M1']=stage_m1(raw,tk,precision,device,batch,m1out,manifest,4096//tk.n_q,16384//tk.n_q,ev)
            results['KO']=stage_ko(raw,tk,precision,device,batch,out/'KO',ev,path,seed,'long')
        else:path,results['M1']=stage_m1(raw,tk,precision,device,batch,m1out,manifest)
        write_json_atomic(out/'selection.json',path)
    else:path=selection_for(job,out)
    if family=='M2':results['M2']=stage_ko(raw,tk,precision,device,batch,out,load_eval(manifest,tk,device),path,seed)
    if family=='M3':results['M3']=stage_patch(raw,tk,precision,device,out,path,seed)
    if family in ('M4','M7','M8'):results['probe']=run_probes(raw,tk,path,out/'probe',device,precision,batch,seed,fixed=family=='M8')
    if family=='M5':results['M5']=stage_m5(raw,tk,precision,device,batch,out,manifest,path,seed,job)
    if family=='M8' and any(s['stage_id']=='M2' for s in job['stages']):
        results['M2']=stage_ko(raw,tk,precision,device,batch,out/'M2',load_eval(manifest,tk,device),path,seed)
        results['M3']=stage_patch(raw,tk,precision,device,out/'M3',path,seed)
    if family=='M9':
        del raw
        for point,src in job['sources'].items():
            raw,tk,precision,meta=load_model(src,device);target=out/('step_'+point);target.mkdir(exist_ok=True)
            write_json_atomic(target/'resolved_source_manifest.json',meta)
            pointjob={**job,'source':src,'missing_midpoint_note':'Predeclared sparse course points; not a measured transition midpoint'}
            results[point]=stage_m5(raw,tk,precision,device,batch,target,manifest,path,seed,pointjob)
            del raw
    else:assert all(not p.requires_grad and p.grad is None for p in raw.parameters())
    write_json_atomic(out/'package_complete.json',{'status':'completed','job':job['job_id'],'elapsed_seconds':time.time()-started,
      'transformer_training_performed':False,'linear_probe_fits_performed':results.get('probe',{}).get('status')=='completed',
      'stage_statuses':{k:v.get('status') for k,v in results.items()}})
    print('DONE',job['job_id'],flush=True)

if __name__=='__main__':main()
