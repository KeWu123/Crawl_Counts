"""Validate FP32 CUDA acceleration of the declared Adam training control.

No TF32/autocast. NumPy supplies identical shuffle orders. CPU archives remain
the reference; every numerical difference is measured rather than assumed away.
"""
from pathlib import Path
import json,time,gc
import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/www/audit/optimization_20261115'

def load():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.38)
    c=np.load(OUT/'cohort.npz'); X=sparse.load_npz(OUT/'features.npz'); tr=~c['held']
    Xtr=torch.from_numpy(X[tr].toarray()).cuda(); Xte=torch.from_numpy(X[c['held']].toarray()).cuda()
    y=torch.from_numpy(c['y'][tr]).cuda()
    return c,Xtr,Xte,y

def fit(c,Xtr,Xte,y,arm,seed,epochs=24,permutation=None):
    start=time.perf_counter(); tr=~c['held']; w=c[arm][tr].astype(np.float32)
    if permutation is not None:w=w[np.random.default_rng(2026111501+permutation).permutation(len(w))]
    w=torch.from_numpy(w.copy()).cuda(); n,d=Xtr.shape; rng=np.random.default_rng(seed)
    W=torch.zeros((d,9),device='cuda'); b=torch.zeros(9,device='cuda')
    mW=torch.zeros_like(W);vW=torch.zeros_like(W);mb=torch.zeros_like(b);vb=torch.zeros_like(b)
    t=0; predictions={}; trace=[]
    for epoch in range(1,epochs+1):
        order=rng.permutation(n)
        for s in range(0,n,256):
            ix=torch.from_numpy(order[s:s+256]).cuda(); Xb=Xtr[ix]
            logits=Xb@W+b; logits-=logits.max(1,keepdim=True).values
            p=torch.exp(logits);p/=p.sum(1,keepdim=True)
            p[torch.arange(len(ix),device='cuda'),y[ix]]-=1.
            wb=w[ix,None]/w[ix].sum(); G=Xb.T@(p*wb)+1e-5*W;gb=(p*wb).sum(0);t+=1
            for P,G_,M,V in ((W,G,mW,vW),(b,gb,mb,vb)):
                M.mul_(.9);M.add_(.1*G_);V.mul_(.999);V.add_(.001*G_*G_)
                P.sub_(.05*(M/(1-.9**t))/(torch.sqrt(V/(1-.999**t))+1e-8))
        if epoch in (1,8,16,24,32,48,64,96):
            with torch.no_grad():
                loss_sum=0.
                for s in range(0,n,256):
                    logits=Xtr[s:s+256]@W+b
                    ce=torch.logsumexp(logits,dim=1)-logits[torch.arange(len(logits),device='cuda'),y[s:s+256]]
                    loss_sum+=(w[s:s+256]*ce).sum().item()
                loss=loss_sum/w.sum().item()+.5e-5*(W*W).sum().item()
                trace.append({'epoch':epoch,'full_weighted_train_objective':loss,'seconds':time.perf_counter()-start})
        if epoch in (24,48,96):
            predictions[f'epoch_{epoch}']=(Xte@W+b).argmax(1).cpu().numpy().astype(np.uint8)
    torch.cuda.synchronize()
    return predictions,trace,time.perf_counter()-start

def main():
    with threadpool_limits(limits=1):
        c,Xtr,Xte,y=load();p,tr,seconds=fit(c,Xtr,Xte,y,'documents',11)
    archived=np.load(ROOT/'paper/www/audit/predictions.npz')['documents'][0]
    result={'seconds':seconds,'archive_mismatches':int((p['epoch_24']!=archived).sum()),'trace':tr,
            'backend':torch.__version__,'GPU':torch.cuda.get_device_name(0),'TF32':False}
    (OUT/'gpu_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(OUT/'gpu_validation.npz',**p)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
