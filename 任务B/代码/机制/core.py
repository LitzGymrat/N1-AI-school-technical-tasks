"""Frozen MQAR mechanics. Reuse the exact runtime model/task; intervene on named rows only."""
import contextlib
import hashlib
import json
import math
import sys
import types
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RUNTIME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME))
from lrexp.model import GPT, ModelCfg
from lrexp.task import Task, TaskCfg
from lrexp.train2 import QFwd
from lrexp.checkpoint_io import sha256_file, write_json_atomic


def tensor_hash(data):
    digest = hashlib.sha256()
    for key in sorted(data):
        a = data[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode()); digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def filtered_cfg(cls, value):
    return cls(**{k:v for k,v in value.items() if k in {f.name for f in fields(cls)}})


def load_model(source, device):
    from asset_cache import materialized
    with materialized(source) as path:
        if not path.is_file():
            raise FileNotFoundError(f'Missing registered weight: {path}')
        digest = sha256_file(path)
        if not source.get('sha256') or digest != source['sha256']:
            raise ValueError('Weight hash must match the registered immutable source')
        saved = torch.load(path, map_location='cpu', weights_only=False)
    cfg = saved['cfg']
    actual_step = saved.get('step_completed', saved.get('step'))
    expected_step = source.get('step', source.get('step_completed'))
    if expected_step is not None and actual_step is not None and int(expected_step) != int(actual_step):
        raise ValueError('Checkpoint step mismatch; no latest/continuation substitution')
    for section in ('model','task'):
        expected = source.get(section+'_config', source.get('expected_'+section+'_config'))
        if expected is not None and cfg[section] != expected:
            raise ValueError('Checkpoint '+section+' configuration differs from registered source')
    if source.get('checkpoint_role') == 'original_training_final_before_any_continuation':
        if actual_step != 64000 or cfg['train']['steps'] != 64000:
            raise ValueError('M8 requires original 64000-step state')
        if saved.get('wandb_run_id') != source['wandb_run_id']:
            raise ValueError('M8 run identity mismatch')
    mc = filtered_cfg(ModelCfg, cfg['model']); tk = filtered_cfg(TaskCfg, cfg['task'])
    raw = GPT(mc).to(device).eval(); raw.load_state_dict(saved['model'], strict=True)
    for param in raw.parameters():param.requires_grad_(False)
    precision = cfg.get('train',{}).get('precision','bf16')
    metadata = {'path':str(path.resolve()), 'sha256':digest, 'step':actual_step or expected_step,
                'step_evidence':'checkpoint' if actual_step is not None else source.get('source_summary_path'),
                'model':cfg['model'], 'task':cfg['task'], 'precision':precision,
                'runtime_model_sha256':sha256_file(RUNTIME/'lrexp/model.py'),
                'runtime_task_sha256':sha256_file(RUNTIME/'lrexp/task.py')}
    del saved
    return raw, tk, precision, metadata


def amp(device, precision):
    return torch.autocast('cuda',dtype=torch.bfloat16) if str(device).startswith('cuda') and precision=='bf16' else contextlib.nullcontext()


def chunks(data, size):
    for start in range(0, len(data['x']), size):
        yield start, {k:v[start:start+size] for k,v in data.items()}


def generate(tk, seed, sequences, device, batch=32, only_long=False):
    cfg = replace(tk, dist='uniform', d_min=tk.L//2 if only_long else 4, d_max=tk.L-2)
    task = Task(cfg, torch.device(device), seed=seed)
    parts = [task.sample(min(batch,sequences-start)) for start in range(0,sequences,batch)]
    return {k:torch.cat([a[k] for a in parts]) for k in ('x','q','s','d')}


def rows(data, location, delta=1):
    return {'query':data['q'], 'source':data['s'], 'value':data['s']+1,
            'relay':data['s']+delta}[location]


def valid_events(data, delta):
    return (data['s']+delta <= data['q']) & (data['s']+delta < data['x'].shape[1])


def unrelated_column(data, row, delta, seed):
    """Fixed hash-of-coordinates rule, independent of logits or test effects."""
    result = torch.remainder(data['s']*104729 + data['q']*13007 + seed, row.clamp_min(1))
    forbidden = (data['s'], data['s']+1, data['s']+delta)
    for _ in range(4):
        bad = torch.stack([result==v for v in forbidden]).any(0)
        result = torch.where(bad, torch.remainder(result+1,row.clamp_min(1)), result)
    valid = row>0
    for value in forbidden:valid &= result != value
    return result, valid


def probabilities(attn, q, k, row, head, blocked_column=None):
    # .float() alone does not prevent the outer BF16 autocast from downcasting
    # einsum. Keep diagnostic/recomputed attention in FP32 like SDPA accumulation.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _probabilities_fp32(attn,q,k,row,head,blocked_column)


def _probabilities_fp32(attn, q, k, row, head, blocked_column=None):
    B,H,L,hd = q.shape
    bi = torch.arange(B,device=q.device)[:,None]
    scores = torch.einsum('bnd,bld->bnl',q[bi,head,row].float(),k[:,head].float()) / math.sqrt(hd)
    idx = torch.arange(L,device=q.device)[None,None,:]
    if attn.cfg.pos=='alibi':scores += attn.slopes[head].float() * (idx-row[...,None])
    scores.masked_fill_(idx>row[...,None],float('-inf'))
    if blocked_column is not None:
        scores.scatter_(-1,blocked_column[...,None],float('-inf'))
    return scores.softmax(-1)


def _native_heads(attn,q,k,v,block_mask):
    if attn.cfg.pos!='alibi':return F.scaled_dot_product_attention(q,k,v,is_causal=True)
    from lrexp.model import flex_attention
    slopes=attn.slopes
    def score_mod(score,b,h,qi,ki):return score+slopes[h]*(ki-qi).to(score.dtype)
    return flex_attention(q,k,v,score_mod=score_mod,block_mask=block_mask)


@contextlib.contextmanager
def intervene(raw, data, path, condition, control_seed=902000, patch=None, capture=None):
    """Apply selected-head row replacements before proj, or residual patches after a block.

    patch entries contain layer/location/values. No final logits are patched.
    Native heads and every untouched row/head remain the native backend output.
    """
    originals=[]; hooks=[]
    delta=path.get('delta',1); eligible=valid_events(data,delta)
    binding=path.get('binding'); retrieval=path.get('retrieval')
    operations=[]
    if condition in ('no_op_recompute','block_binding_edge','block_both','binding_control') and binding:
        operations.append((binding,'relay','source','mask' if condition in ('block_binding_edge','block_both') else ('control' if condition=='binding_control' else 'noop')))
    if condition in ('no_op_recompute','block_retrieval_edge','block_both','retrieval_control','zero_retrieval_head_output') and retrieval:
        operations.append((retrieval,'query','relay','mask' if condition in ('block_retrieval_edge','block_both') else ('control' if condition=='retrieval_control' else ('zero' if condition=='zero_retrieval_head_output' else 'noop'))))
    if condition=='unrelated_head_or_layer_control' and path.get('other_head'):
        operations.append((path['other_head'],'query','relay','zero'))
    for layer,block in enumerate(raw.blocks):
        relevant=[op for op in operations if op[0]['layer']==layer]
        need_capture=capture is not None and retrieval and retrieval['layer']==layer
        head_patch=patch if patch and patch['kind']=='head' and patch['layer']==layer else None
        if relevant or need_capture or head_patch:
            attn=block.attn; original=attn.forward; originals.append((attn,original))
            def replaced(self,x,cos,sin,block_mask=None,ops=relevant,do_capture=need_capture,hpatch=head_patch):
                q,k,v=self._qkv(x,cos,sin); y=_native_heads(self,q,k,v,block_mask)
                B,H,L,hd=y.shape; bi=torch.arange(B,device=x.device)[:,None]
                if ops or hpatch:y=y.clone()
                for target,rowname,colname,action in ops:
                    head=target['head']; row=rows(data,rowname,delta).clamp_max(L-1)
                    keep=eligible & (row>=0)
                    control_col,control_ok=unrelated_column(data,row,delta,control_seed)
                    # Apply target and matched-control edges on exactly the same events.
                    if rowname=='relay':keep &= control_ok
                    if action=='zero':new=torch.zeros((*row.shape,hd),device=x.device,dtype=y.dtype)
                    else:
                        col=None
                        if action=='mask':col=rows(data,colname,delta).clamp(0,L-1)
                        if action=='control':col=control_col;keep &= control_ok
                        p=probabilities(self,q,k,row,head,col)
                        with torch.autocast(device_type=x.device.type, enabled=False):
                            new=(p@v[:,head].float()).to(y.dtype)
                    prior=y[bi,head,row];y[bi,head,row]=torch.where(keep[...,None],new,prior)
                if do_capture:
                    head=retrieval['head'];capture['head']=y[bi,head,data['q']].detach().clone()
                if hpatch:
                    row=rows(data,hpatch['location'],delta).clamp_max(L-1);head=hpatch['head']
                    prior=y[bi,head,row];y[bi,head,row]=torch.where(eligible[...,None],hpatch['values'].to(y.dtype),prior)
                return self.proj(y.transpose(1,2).reshape(B,L,self.inner))
            attn.forward=types.MethodType(replaced,attn)
        need_residual=capture is not None and binding and binding['layer']==layer
        residual_patch=patch if patch and patch['kind']=='residual' and patch['layer']==layer else None
        if need_residual or residual_patch:
            def hook(module,inputs,output,cap=need_residual,rpatch=residual_patch):
                bi=torch.arange(output.shape[0],device=output.device)[:,None]
                if cap:
                    relay=rows(data,'relay',delta).clamp_max(output.shape[1]-1)
                    capture['binding']=output[bi,relay].detach().clone()
                    control,_=unrelated_column(data,relay,delta,control_seed)
                    capture['unrelated']=output[bi,control].detach().clone()
                if rpatch:
                    if rpatch['location']=='unrelated':row,ok=unrelated_column(data,rows(data,'relay',delta),delta,control_seed)
                    else:row=rows(data,rpatch['location'],delta);ok=eligible
                    row=row.clamp_max(output.shape[1]-1);updated=output.clone()
                    updated[bi,row]=torch.where(ok[...,None],rpatch['values'].to(output.dtype),output[bi,row]);return updated
            hooks.append(block.register_forward_hook(hook))
    try:yield
    finally:
        for attn,original in originals:attn.forward=original
        for hook in hooks:hook.remove()


def logits(raw,data,device,precision,path=None,condition='native',patch=None,capture=None,control_seed=902000):
    with torch.no_grad(),amp(device,precision),intervene(raw,data,path or {},condition,control_seed=control_seed,patch=patch,capture=capture):
        return QFwd(raw)(data['x'],data['q'],raw.block_mask(data['x'].shape[1],device)).float()


def query_arrays(lg,data):
    target=data['x'].gather(1,data['q']+1)
    correct_logit=lg.gather(-1,target[...,None]).squeeze(-1)
    wrong=lg.clone();wrong.scatter_(-1,target[...,None],float('-inf'))
    return {'pred':lg.argmax(-1),'target':target,'correct':lg.argmax(-1)==target,
            'ce':F.cross_entropy(lg.flatten(0,1),target.flatten(),reduction='none').reshape_as(target),
            'margin':correct_logit-wrong.max(-1).values, 's':data['s'],'q':data['q'],'d':data['d']}


def summarize(arrays,L):
    result={}
    for label,mask in [('all',np.ones_like(arrays['d'],dtype=bool)),('long',arrays['d']>=L//2),('short',arrays['d']<L//2)]:
        n=int(mask.sum());result[label]={'n_queries':n,'accuracy':float(arrays['correct'][mask].mean()) if n else None,
          'ce':float(arrays['ce'][mask].mean()) if n else None,'margin':float(arrays['margin'][mask].mean()) if n else None}
    return result


def evaluate(raw,data,device,precision,batch,path=None,condition='native',control_seed=902000):
    parts=[]
    for _,piece in chunks(data,batch):
        parts.append({k:v.detach().cpu().numpy() for k,v in query_arrays(logits(raw,piece,device,precision,path,condition,control_seed=control_seed),piece).items()})
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def paired_effect(base,other,L,seed=970000):
    result={};rng=np.random.default_rng(seed);N=len(base['correct']);ix=rng.integers(N,size=(1000,N))
    masks=[('all',np.ones_like(base['d'],bool)),('long',base['d']>=L//2),('short',base['d']<L//2)]
    for lower,upper in zip(np.linspace(0,L,9,dtype=int)[:-1],np.linspace(0,L,9,dtype=int)[1:]):
        masks.append((f'distance_{lower}_{upper-1}',(base['d']>=lower)&(base['d']<upper)))
    for label,mask in masks:
        count=mask.sum(1);den=count.sum()
        delta=((base['correct'].astype(float)-other['correct'])*mask).sum(1)*100
        if not den:result[label]=None;continue
        den_boot=count[ix].sum(1);vals=np.divide(delta[ix].sum(1),den_boot,out=np.full(1000,np.nan),where=den_boot>0)
        result[label]={'accuracy_drop_pp':float(delta.sum()/den),'ci95':np.nanquantile(vals,[.025,.975]).tolist(),
          'ce_increase':float(((other['ce']-base['ce'])*mask).sum()/den),
          'margin_drop':float(((base['margin']-other['margin'])*mask).sum()/den),'n_queries':int(den),'n_sequences':N}
    return result


@torch.no_grad()
def attention_profile(raw,data,device,precision,batch,max_delta=64,binding_delta=None):
    H=raw.cfg.n_head;D=min(max_delta,data['x'].shape[1]-1)
    sums=np.zeros((raw.cfg.n_layer,H,D));counts=np.zeros(D);binding_sum=np.zeros((raw.cfg.n_layer,H));binding_count=0
    key_sum=np.zeros((raw.cfg.n_layer,H));prev_sum=np.zeros_like(key_sum);long_count=0
    for _,piece in chunks(data,batch):
        raw.set_diag(True)
        try:
            logits(raw,piece,device,precision)
            long=piece['d']>=piece['x'].shape[1]//2
            long_count+=int(long.sum())
            offsets=torch.arange(1,D+1,device=device)
            col=piece['s'][...,None]+offsets;mask=long[...,None] & (col<=piece['q'][...,None])
            counts+=mask.sum((0,1)).cpu().numpy()
            for layer,block in enumerate(raw.blocks):
                q,k=block.attn.saved
                for head in range(H):
                    p=probabilities(block.attn,q,k,piece['q'],head)
                    key_sum[layer,head]+=float((p.gather(-1,piece['s'][...,None]).squeeze(-1)*long).sum())
                    pp_prev=probabilities(block.attn,q,k,piece['s']+1,head)
                    prev_sum[layer,head]+=float((pp_prev.gather(-1,piece['s'][...,None]).squeeze(-1)*long).sum())
                    values=p.gather(-1,col.clamp_max(p.shape[-1]-1));sums[layer,head]+=(values*mask).sum((0,1)).cpu().numpy()
                    if binding_delta is not None:
                        row=piece['s']+binding_delta;ok=long & (row<=piece['q'])
                        pp=probabilities(block.attn,q,k,row.clamp_max(p.shape[-1]-1),head)
                        val=pp.gather(-1,piece['s'][...,None]).squeeze(-1)
                        binding_sum[layer,head]+=float((val*ok).sum())
            if binding_delta is not None:binding_count+=int((long & (piece['s']+binding_delta<=piece['q'])).sum())
        finally:raw.set_diag(False)
    return {'relay_mass':np.divide(sums,counts[None,None,:],out=np.zeros_like(sums),where=counts[None,None,:]>0).tolist(),
            'query_to_source_key_mass':(key_sum/max(1,long_count)).tolist(),
            'source_value_to_source_key_mass':(prev_sum/max(1,long_count)).tolist(),
            'long_query_count':long_count,'offset_counts':counts.astype(int).tolist(), 'binding_mass':(binding_sum/max(1,binding_count)).tolist(),
            'binding_count':binding_count, 'n_sequences':len(data['x']), 'data_sha256':tensor_hash(data)}


def select_path(raw,data,device,precision,batch):
    profile=attention_profile(raw,data,device,precision,batch)
    mass=np.asarray(profile['relay_mass']);layer,head,offset=np.unravel_index(np.argmax(mass),mass.shape)
    delta=int(offset+1);retrieval={'layer':int(layer),'head':int(head),'mass':float(mass[layer,head,offset]),
      'module':f'blocks.{layer}.attn','output_projection':f'blocks.{layer}.attn.proj'}
    local=attention_profile(raw,data,device,precision,batch,binding_delta=delta)
    binding=None
    if layer>0:
        scores=np.asarray(local['binding_mass'])[:layer];bl,bh=np.unravel_index(np.argmax(scores),scores.shape)
        binding={'layer':int(bl),'head':int(bh),'mass':float(scores[bl,bh]),
          'module':f'blocks.{bl}.attn','residual_tap':f'blocks.{bl}:output'}
    baseline=evaluate(raw,data,device,precision,batch)
    acc=summarize(baseline,data['x'].shape[1])['long']['accuracy']
    located=bool(binding and retrieval['mass']>=.3 and binding['mass']>=.1 and acc is not None and acc>=.5)
    others=[{'layer':l,'head':h} for l in range(raw.cfg.n_layer) for h in range(raw.cfg.n_head) if (l,h)!=(layer,head)]
    return {'status':'localized' if located else 'unlocalized','delta':delta,'retrieval':retrieval,'binding':binding,
      'other_head':others[0] if others else None,'selection_long_accuracy':acc,'selection_data_sha256':tensor_hash(data),
      'selection_seed':900000,'thresholds':{'retrieval_mass':.3,'binding_mass':.1,'selection_long_accuracy':.5},
      'threshold_role':'fixed operational gate before evaluation, not evidence of causality',
      'profile':profile,'binding_profile':local,'control_rule':'fixed coordinate hash, exclude s/s+1/relay; no effect-based selection'}
