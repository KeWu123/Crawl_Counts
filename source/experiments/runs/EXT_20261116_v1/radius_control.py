"""Execute planned R01: full-support four-candidate radius LP with certificate."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];DEST=OUT/'R01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'protocol.json').exists()
    input=ROOT/'paper/www/audit/partitions.npz'
    protocol={'id':'R01_v1','frozen_before_solver':True,'parent_plan':'experiments/experiment_plan_20261115/configs/prospective_protocols.json',
        'input_sha256':sha(input),'entry_sha256':sha(Path(__file__)),'candidates':['url','title_date','title','lsh'],
        'objective':'min_pi max_k 0.5 E_Q |sum_j pi_j v_j-v_k|',
        'compression':'exact identical density-vector groups with full frequency weights; no subsampling',
        'solver':'HiGHS continuous LP, time limit300s, primal/dual tolerance1e-9',
        'required':'primal feasibility<=1e-7; dual gap<=1e-7; compressed/full-support objectives agree<=1e-10',
        'comparison':['diameter midpoint','bound-minimizing mixture'],'new_training':0}
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    start=time.perf_counter();g=np.load(input);keys=protocol['candidates'];n=len(g['url'])
    V=np.column_stack([n/(len(np.bincount(g[k]))*np.bincount(g[k])[g[k]]) for k in keys])
    U,counts=np.unique(V,axis=0,return_counts=True);m=len(U);freq=counts/n
    nv=5+4*m;rows=[];cols=[];vals=[];rhs=[]
    for k in range(4):
        for i in range(m):
            for sign in [1.,-1.]:
                row=len(rhs);rows.extend([row]*5);cols.extend([0,1,2,3,5+k*m+i]);vals.extend([*(sign*U[i]),-1.]);rhs.append(sign*U[i,k])
        row=len(rhs);rows.append(row);cols.append(4);vals.append(-1.)
        rows.extend([row]*m);cols.extend(range(5+k*m,5+(k+1)*m));vals.extend(.5*freq);rhs.append(0.)
    A=sparse.coo_matrix((vals,(rows,cols)),shape=(len(rhs),nv)).tocsr();b=np.array(rhs)
    eq=sparse.csr_matrix(([1.]*4,([0]*4,range(4))),shape=(1,nv));obj=np.zeros(nv);obj[4]=1.
    r=linprog(obj,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.],bounds=(0,None),method='highs',
        options={'time_limit':300,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'threads':1})
    assert r.success,r.message
    pi=r.x[:4];density=V@pi;dist=np.mean(np.abs(density[:,None]-V),axis=0)*.5;R=float(dist.max())
    D=np.array([[np.abs(V[:,i]-V[:,j]).mean()*.5 for j in range(4)] for i in range(4)])
    B=float((pi@D).max());dual=float(b@r.ineqlin.marginals+r.eqlin.marginals[0]);primal=float(r.fun)
    violation=max(0.,float((A@r.x-b).max()),abs(pi.sum()-1),float(-r.x.min()))
    assert abs(R-primal)<1e-10 and abs(dual-primal)<1e-7 and violation<1e-7
    compressed=float(max(.5*np.sum(freq*np.abs(U@pi-U[:,k])) for k in range(4)))
    assert abs(compressed-R)<1e-10
    info={'status':'numerically_certified_optimum','n_documents':n,'n_exact_density_groups':m,'n_variables':nv,
        'n_inequalities':len(rhs),'pi':dict(zip(keys,pi.tolist())),'R':R,'B':B,'per_candidate_TV':dict(zip(keys,dist.tolist())),
        'candidate_D':D.tolist(),'diameter_half':float(D.max()/2),'primal_objective':primal,'dual_objective':dual,
        'duality_gap':float(abs(primal-dual)),'primal_feasibility_violation':violation,'compressed_R':compressed,
        'seconds':time.perf_counter()-start,'solver_message':r.message,
        'scope':'floating-point LP certificate at reported tolerances; not a symbolic exact proof or an external-target guarantee'}
    (DEST/'results.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(DEST/'solver_certificate.npz',x=r.x,inequality_duals=r.ineqlin.marginals,equality_duals=r.eqlin.marginals,
                        lower_duals=r.lower.marginals,compressed_density_vectors=U,frequencies=freq)
    print(json.dumps(info,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
