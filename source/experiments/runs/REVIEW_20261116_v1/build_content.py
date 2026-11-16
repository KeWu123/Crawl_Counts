"""Construct candidates and centers using feature matrices only, before reference access."""
import io,time,zipfile
import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
from common import *

def radius_center(laws):
    # Densities relative to uniform records; identical density rows are exactly compressed.
    names=list(laws);V=np.array([laws[k] for k in names]);k,n=V.shape
    U,counts=np.unique((n*V).T,axis=0,return_counts=True);freq=counts/n;m=len(U);nv=k+1+k*m
    rr=[];cc=[];vals=[];rhs=[]
    for j in range(k):
        for i in range(m):
            for sign in [1.,-1.]:
                row=len(rhs);rr.extend([row]*(k+1));cc.extend([*range(k),k+1+j*m+i]);vals.extend([*(sign*U[i]),-1.]);rhs.append(sign*U[i,j])
        row=len(rhs);rr.extend([row]*(m+1));cc.extend([k,*range(k+1+j*m,k+1+(j+1)*m)]);vals.extend([-1.,*(.5*freq)]);rhs.append(0.)
    A=sparse.coo_matrix((vals,(rr,cc)),shape=(len(rhs),nv)).tocsr();rhs=np.array(rhs)
    c=np.zeros(nv);c[k]=1;eq=np.r_[np.ones(k),np.zeros(nv-k)]
    fit=linprog(c,A_ub=A,b_ub=rhs,A_eq=eq[None,:],b_eq=[1.],bounds=(0.,None),method='highs')
    assert fit.success,fit.message
    pi=fit.x[:k];center=pi@V;D=.5*np.abs(V[:,None,:]-V[None,:,:]).sum(2)
    R=float((.5*np.abs(V-center).sum(1)).max());B=float((D.T@pi).max())
    y=fit.ineqlin.marginals;lam=fit.eqlin.marginals[0];ld=fit.lower.marginals
    stationarity=float(abs(c-A.T@y-eq*lam-ld).max());dual=float(rhs@y+lam)
    assert stationarity<1e-8 and abs(R-dual)<1e-8 and (A@fit.x-rhs).max()<1e-8
    assert B+1e-12>=R and abs(pi.sum()-1)<1e-10
    uniform=V.mean(0)
    result={'candidate_names':names,'pi':dict(zip(names,pi.tolist())),'R':R,'B':B,'D':D.tolist(),
        'uniform_pi':{name:1/k for name in names},'uniform_R':float((.5*abs(V-uniform).sum(1)).max()),
        'uniform_B':float((D.mean(0)).max()),'feasible_set':'convex mixture of declared candidate laws',
        'compressed_density_rows':m,'primal_dual_gap':abs(R-dual),'dual_stationarity_linf':stationarity}
    return result,center,uniform,{'x':fit.x,'duals':y,'eq_dual':np.array([lam]),'lower_duals':ld,'U':U,'counts':counts}

def main():
    started=time.perf_counter();p=read(HERE/'protocol.json')
    assert not (HERE/'content_frozen.json').exists()
    for name,h in p['implementation_sha256'].items():assert sha(HERE/name)==h,name
    assert sha(ZIP)==p['source_zip_sha256']
    views=[];terms_by=[];files_read=[];sizes=[]
    with zipfile.ZipFile(ZIP) as z:
        for pub in ['bbc','guardian','reuters']:
            names=[f'3sources/3sources_{pub}.terms',f'3sources/3sources_{pub}.mtx']
            # This constructor has no code path that opens .docs or .clist.
            files_read.extend(names)
            terms=[s.strip() for s in z.read(names[0]).decode('utf-8').splitlines()]
            x=mmread(io.BytesIO(z.read(names[1]))).tocoo()
            assert len(set(terms))==len(terms)==x.shape[0] and all(terms)
            assert np.all(x.data>0)
            terms_by.append(terms);views.append(x);sizes.append(x.shape[1])
    vocab=sorted(set().union(*(set(v) for v in terms_by)));index={t:i for i,t in enumerate(vocab)}
    documents=[]
    for terms,x in zip(terms_by,views):
        mapping=np.array([index[t] for t in terms])
        documents.append(sparse.coo_matrix((x.data,(x.col,mapping[x.row])),shape=(x.shape[1],len(vocab))).tocsr())
    X=sparse.vstack(documents,format='csr').astype(np.float64);n=X.shape[0];assert n==948
    df=np.bincount(X.indices,minlength=X.shape[1]);idf=1+np.log((1+n)/(1+df))
    X.data=1+np.log(X.data);X=X.multiply(idf).tocsr()
    norms=np.sqrt(X.multiply(X).sum(1)).A1
    X=sparse.diags(np.divide(1.,norms,out=np.zeros(n),where=norms>0))@X
    similarity=np.clip((X@X.T).toarray(),0.,1.);np.fill_diagonal(similarity,0.)
    pub=np.repeat(np.arange(3),sizes);laws={'documents':np.ones(n)/n,'publisher':1/(3*np.bincount(pub)[pub])}
    partitions={'publisher':pub};details={}
    for threshold in [.50,.70]:
        name=f'content{round(100*threshold):03d}'
        edges=np.triu(similarity>=threshold,1)
        K,g=connected_components(sparse.csr_matrix(edges|edges.T),directed=False)
        sizes_g=np.bincount(g);partitions[name]=g;laws[name]=1/(K*sizes_g[g])
        details[name]={'threshold':threshold,'clusters':int(K),'largest_group':int(sizes_g.max()),
                       'edge_count':int(edges.sum()),'non_singleton_groups':int((sizes_g>1).sum())}
    assert all(abs(v.sum()-1)<1e-12 for v in laws.values())
    arrays={**{f'law_{k}':v for k,v in laws.items()},**{f'group_{k}':v for k,v in partitions.items()}}
    centers={}
    for tag,names in p['content']['candidate_sets'].items():
        result,center,uniform,certificate=radius_center({k:laws[k] for k in names})
        arrays[f'center_{tag}']=center;arrays[f'uniform_{tag}']=uniform;centers[tag]=result
        np.savez_compressed(HERE/f'lp_certificate_{tag}.npz',**certificate)
    np.savez_compressed(HERE/'content_laws.npz',**arrays)
    sparse.save_npz(HERE/'pooled_tfidf.npz',X)
    (HERE/'term_vocabulary.txt').write_text('\n'.join(vocab)+'\n',encoding='utf-8')
    a,b=np.triu_indices(n,1);np.savez_compressed(HERE/'pairwise_cosine.npz',row=a,col=b,cosine=similarity[a,b])
    write(HERE/'content_frozen.json',{'status':'constructed_without_curator_reference_access','files_read':files_read,
        'terms':len(vocab),'n_records':n,'publisher_sizes':sizes,'zero_vectors':int((norms==0).sum()),
        'partitions':details,'centers':centers,'wall_seconds':time.perf_counter()-started,
        'outputs_sha256':{name:sha(HERE/name) for name in ['content_laws.npz','pooled_tfidf.npz','term_vocabulary.txt','pairwise_cosine.npz']},
        'labels_used_for_candidate_or_center_selection':False})
    print(json.dumps({'terms':len(vocab),'partitions':details,'centers':centers},indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
