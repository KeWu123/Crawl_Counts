"""Frozen FP32 Adam engines for normalization and second-learner validation."""
import hashlib,time
import numpy as np
import torch

CHECKPOINTS=(24,48,96)

def init_params(d,nc,seed,model):
    rng=np.random.default_rng(seed)
    if model=='linear':
        pars=[np.zeros((d,nc),np.float32),np.zeros(nc,np.float32)]
    else:
        # Identical shape, initialization and RNG consumption to tilt_sweep.mlp_train.
        pars=[(rng.standard_normal((d,64))/np.sqrt(d)).astype(np.float32),np.zeros(64,np.float32),
              (rng.standard_normal((64,nc))/8.).astype(np.float32),np.zeros(nc,np.float32)]
    return [torch.from_numpy(x).cuda() for x in pars],rng

def forward(X,p):
    if len(p)==2:return X@p[0]+p[1],None
    H=X@p[0]+p[1]
    return torch.relu(H)@p[2]+p[3],H

def gradients(X,y,w,p,normalizer,reg):
    logits,H=forward(X,p)
    logits-=logits.max(1,keepdim=True).values
    prob=torch.exp(logits);prob/=prob.sum(1,keepdim=True)
    prob[torch.arange(len(y),device='cuda'),y]-=1.
    residual=prob*(w[:,None]/normalizer)
    if H is None:return [X.T@residual+reg*p[0],residual.sum(0)]
    A=torch.relu(H)
    G2=A.T@residual+reg*p[2];g2=residual.sum(0)
    D=(residual@p[2].T)*(H>0)
    return [X.T@D+reg*p[0],D.sum(0),G2,g2]

def fit(Xtr,Xte,y,w,seed,model='linear',normalization='batch',epochs=96):
    started=time.perf_counter();n,d=Xtr.shape;nc=int(y.max().item())+1
    p,rng=init_params(d,nc,seed,model);lr=.05 if model=='linear' else .003
    reg=1e-5 if model=='linear' else 1e-4
    m=[torch.zeros_like(x) for x in p];v=[torch.zeros_like(x) for x in p]
    wt=torch.from_numpy(w.astype(np.float32)).cuda();total=wt.sum()
    t=0;saved={};trace=[];order_digest=hashlib.sha256();sums=[];esses=[];scales=[]
    for epoch in range(1,epochs+1):
        order=rng.permutation(n);order_digest.update(order.tobytes())
        for s in range(0,n,256):
            ids=order[s:s+256];ix=torch.from_numpy(ids).cuda();a=wt[ix]
            normalizer=a.sum() if normalization=='batch' else len(ids)*total/n
            g=gradients(Xtr[ix],y[ix],a,p,normalizer,reg)
            t+=1
            for P,G,M,V in zip(p,g,m,v):
                M.mul_(.9);M.add_(.1*G);V.mul_(.999);V.add_(.001*G*G)
                P.sub_(lr*(M/(1-.9**t))/(torch.sqrt(V/(1-.999**t))+1e-8))
            ww=w[ids].astype(float);sm=ww.sum()
            sums.append(sm);esses.append(sm*sm/(ww@ww));scales.append(sm/(len(ids)*w.mean(dtype=float)))
        if epoch in CHECKPOINTS:
            with torch.no_grad():
                logits,_=forward(Xte,p);logits=logits.cpu().numpy().copy()
                saved[f'epoch_{epoch}']=logits.argmax(1).astype(np.uint8)
                saved[f'logits_{epoch}']=logits
                if model=='linear':
                    saved[f'W_{epoch}']=p[0].cpu().numpy().copy();saved[f'b_{epoch}']=p[1].cpu().numpy().copy()
                ce=0.
                for s in range(0,n,256):
                    Z,_=forward(Xtr[s:s+256],p)
                    l=torch.logsumexp(Z,1)-Z[torch.arange(len(Z),device='cuda'),y[s:s+256]]
                    ce+=(wt[s:s+256]*l).sum().item()
                penalty=.5*reg*sum((P*P).sum().item() for k,P in enumerate(p) if k%2==0)
                trace.append({'epoch':epoch,'global_CE':ce/total.item(),'L2_penalty':penalty,
                    'global_objective':ce/total.item()+penalty,'seconds':time.perf_counter()-started})
    with torch.no_grad():
        # Global objective gradient at the final checkpoint, using mini-batch sums.
        G=[torch.zeros_like(P) for P in p]
        for s in range(0,n,256):
            gg=gradients(Xtr[s:s+256],y[s:s+256],wt[s:s+256],p,total,0.)
            for a,b in zip(G,gg):a.add_(b)
        for k in range(0,len(p),2):G[k].add_(reg*p[k])
        grad_linf=max(a.abs().max().item() for a in G)
    for k,P in enumerate(p):saved[f'param_{k}']=P.cpu().numpy().copy()
    torch.cuda.synchronize()
    assert all(np.isfinite(x).all() for x in saved.values())
    sums=np.asarray(sums);esses=np.asarray(esses);scales=np.asarray(scales)
    diag={'global_gradient_linf_final':grad_linf,'batch_order_sha256':order_digest.hexdigest(),
          'batch_weight_sum_CV':float(sums.std(ddof=1)/sums.mean()),
          'batch_weight_sum_quantiles':np.quantile(sums,[0,.25,.5,.75,1]).tolist(),
          'batch_ESS_quantiles':np.quantile(esses,[0,.25,.5,.75,1]).tolist(),
          'batch_to_global_gradient_scale_quantiles':np.quantile(scales,[0,.25,.5,.75,1]).tolist()}
    return saved,trace,diag,time.perf_counter()-started
