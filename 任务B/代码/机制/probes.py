"""Sequence-separated frozen features and fixed-budget affine readouts (no Transformer updates)."""
import hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from core import amp, chunks, generate, tensor_hash, write_json_atomic


@torch.no_grad()
def features(raw,data,path,device,precision,batch,fixed=False):
    delta=1 if fixed else path['delta']
    binding_layer=0 if fixed else path['binding']['layer']
    read_layer=len(raw.blocks)-1 if fixed else path['retrieval']['layer']
    banks={k:[] for k in ['source_embedding','relay_residual','query_embedding','query_residual']}
    labels={k:[] for k in ['key','value','distance','sequence_id','source','query']}
    sequence_hashes=[]
    for offset,piece in chunks(data,batch):
        x,q,s=piece['x'],piece['q'],piece['s'];r=s+delta
        valid=r<=q;bi=torch.arange(len(x),device=device)[:,None]
        sequence_hashes += [hashlib.sha256(row.cpu().contiguous().numpy().tobytes()).hexdigest() for row in x]
        with amp(device,precision):
            h=raw.emb(x)
            if raw.cfg.smear:h=h+raw.emb_prev(F.pad(x[:,:-1],(1,0),value=0))
            banks['source_embedding'].append(h[bi,s+1][valid].float().detach())
            banks['query_embedding'].append(h[bi,q][valid].float().detach())
            for layer,block in enumerate(raw.blocks):
                h=block(h,raw.cos,raw.sin,raw.block_mask(x.shape[1],device))
                if layer==binding_layer:banks['relay_residual'].append(h[bi,r.clamp_max(x.shape[1]-1)][valid].float().detach())
                if layer==read_layer:banks['query_residual'].append(h[bi,q][valid].float().detach())
        labels['key'].append(x[bi,s][valid])
        labels['value'].append(x[bi,s+1][valid]);labels['distance'].append(piece['d'][valid])
        labels['sequence_id'].append((bi.expand_as(q)+offset)[valid])
        labels['source'].append(s[valid]);labels['query'].append(q[valid])
    return {k:torch.cat(v) for k,v in banks.items()},{k:torch.cat(v) for k,v in labels.items()},sequence_hashes


def fit_probe(X, y, classes, steps=400):
    """Same full-batch zero-initialized affine/Adam formulation as lrexp.train.linear_probe."""
    with torch.enable_grad():
        W=torch.zeros(X.shape[1],classes,device=X.device,requires_grad=True)
        b=torch.zeros(classes,device=X.device,requires_grad=True)
        opt=torch.optim.Adam([W,b],lr=.01,weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad();F.cross_entropy(X@W+b,y).backward();opt.step()
    return W.detach(),b.detach()


def run_probes(raw,tk,path,out,device,precision,batch,model_seed,fixed=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not fixed and path.get('status')!='localized':
        result={'status':'NA_unlocalized','transformer_updated':False};write_json_atomic(out/'probe_metrics.json',result);return result
    banks={};labels={};hashes={};dataset_hash={}
    for split,seed,n in [('train',920000,512),('valid',930000,128),('test',940000,128)]:
        data=generate(tk,seed,n,device,batch)
        banks[split],labels[split],hashes[split]=features(raw,data,path,device,precision,batch,fixed)
        labels[split]['key']-=tk.Vb;labels[split]['value']-=tk.Vb+tk.K
        dataset_hash[split]={'seed':seed,'sequences':n,'sha256':tensor_hash(data),'queries':len(labels[split]['key'])}
        del data
    assert not(set(hashes['train'])&set(hashes['valid']) or set(hashes['train'])&set(hashes['test']) or set(hashes['valid'])&set(hashes['test']))
    torch.manual_seed(950000+model_seed)
    result={'status':'completed','transformer_updated':False,'steps':400,'optimizer':'Adam lr=.01 wd=1e-4 full_batch',
      'fixed_architecture_taps':fixed,'data':dataset_hash,'probes':{},'tap_boundary':'source/query embeddings and block outputs; no q+1 features'}
    result['taps']={'source_embedding':{'module':'emb plus emb_prev if smear','position':'s+1'},
      'query_embedding':{'module':'emb plus emb_prev if smear','position':'q'},
      'relay_residual':{'module':f"blocks.{0 if fixed else path['binding']['layer']}:output",'position':'s+1' if fixed else 's+'+str(path['delta'])},
      'query_residual':{'module':f"blocks.{len(raw.blocks)-1 if fixed else path['retrieval']['layer']}:output",'position':'q'}}
    pred_arrays={};standardization={}
    for tap in banks['train']:
        mu=banks['train'][tap].mean(0);sd=banks['train'][tap].std(0)+1e-5
        X={k:(v[tap]-mu)/sd for k,v in banks.items()}
        standardization[tap+'_mean']=mu.cpu().numpy();standardization[tap+'_std']=sd.cpu().numpy()
        for label,classes in [('key',tk.K),('value',tk.Vv)]:
            for condition in ['true_labels','permuted_labels']:
                ys={}
                for split,seed in [('train',951000),('valid',952000),('test',953000)]:
                    y=labels[split][label]
                    if condition=='permuted_labels':
                        idx=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed)).to(y.device);y=y[idx]
                    ys[split]=y
                W,b=fit_probe(X['train'],ys['train'],classes,400)
                name=f'{tap}/{label}/{condition}';row={'chance':1/classes,'splits':{}}
                with torch.no_grad():
                    for split in ['train','valid','test']:
                        lg=X[split]@W+b;pred=lg.argmax(-1);y=ys[split];d=labels[split]['distance']
                        correct=pred==y
                        row['splits'][split]={'n':len(y),'accuracy':float(correct.float().mean()),'ce':float(F.cross_entropy(lg,y)),
                          'class_histogram':torch.bincount(y,minlength=classes).cpu().tolist(),
                          'long_accuracy':float(correct[d>=tk.L//2].float().mean()) if bool((d>=tk.L//2).any()) else None,
                          'short_accuracy':float(correct[d<tk.L//2].float().mean()) if bool((d<tk.L//2).any()) else None}
                        if split!='train':
                            prefix=name.replace('/','__')+'__'+split
                            pred_arrays[prefix+'__pred']=pred.cpu().numpy();pred_arrays[prefix+'__target']=y.cpu().numpy()
                result['probes'][name]=row
                write_json_atomic(out/'probe_metrics.json',result)
    for split in labels:
        for field in ['distance','sequence_id','source','query']:pred_arrays[split+'__'+field]=labels[split][field].cpu().numpy()
    np.savez_compressed(out/'probe_predictions.npz',**pred_arrays)
    np.savez_compressed(out/'train_standardization.npz',**standardization)
    write_json_atomic(out/'split_hashes.json',hashes)
    assert all(not p.requires_grad and p.grad is None for p in raw.parameters())
    return result
