"""BBC author-release corpus extension; no invented metadata or labels."""
from pathlib import Path
import argparse,hashlib,io,json,re,sys,time,zipfile
import numpy as np,requests,torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from engine import fit
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];BBC=OUT/'BBC01_v2'
sys.path.insert(0,str(ROOT/'code'))
from real_ccnews import token_ids,shingle_hashes,minhash,lsh_pairs,exact_jaccard,partition
from real_provenance import norm_title,group_of
from real_task import features
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def prepare():
    assert not (BBC/'cohort.npz').exists()
    BBC.mkdir(exist_ok=True)
    url='https://raw.githubusercontent.com/derekgreene/bbc-datasets/main/raw/bbc-fulltext.zip'
    r=requests.get(url,timeout=60);r.raise_for_status();data=r.content
    git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert git=='2e2ee8d266d6e215ae4b481c27e19f726e147bcb'
    (BBC/'bbc-fulltext.zip').write_bytes(data)
    rd=requests.get('https://raw.githubusercontent.com/derekgreene/bbc-datasets/main/README.md',timeout=30);rd.raise_for_status()
    (BBC/'SOURCE_README.md').write_text(rd.text,encoding='utf-8')
    classes=['business','entertainment','politics','sport','tech'];records=[]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.testzip() is None
        for name in sorted(z.namelist()):
            path=Path(name)
            if len(path.parts)!=3 or path.parts[1] not in classes or path.suffix!='.txt':continue
            raw=z.read(name)
            try:text=raw.decode('utf-8');enc='utf-8'
            except UnicodeDecodeError:text=raw.decode('cp1252');enc='cp1252'
            lines=text.strip().splitlines();title=lines[0].strip();body='\n'.join(lines[1:]).strip()
            if not title or not body:continue
            records.append({'id':name,'class':classes.index(path.parts[1]),'title':title,'body':body,
                'encoding':enc,'raw_sha256':hashlib.sha256(raw).hexdigest()})
    assert len(records)==2225
    with (BBC/'records.jsonl').open('w',encoding='utf-8') as f:
        for x in records:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    # The corpus remains the native benchmark. Shingle-ineligible articles are singletons.
    vocab={};raw=[];eligible=[]
    for i,row in enumerate(records):
        hs=shingle_hashes(token_ids(row['body'],vocab),n=200)
        if len(hs)>=40:raw.append(hs);eligible.append(i)
    from collections import Counter
    counts=Counter(int(v) for hs in raw for v in hs)
    drop=np.array([v for v,c in counts.items() if c>.5*len(raw)],dtype=np.uint64)
    shingles=[];ids=[]
    for i,hs in zip(eligible,raw):
        h=hs[~np.isin(hs,drop)] if len(drop) else hs
        if len(h)>=30:ids.append(i);shingles.append(h[:120])
    flat=np.concatenate(shingles);offs=np.r_[0,np.cumsum([len(s) for s in shingles])]
    candidates=lsh_pairs(minhash(flat,offs,len(ids)));edges=exact_jaccard(flat,offs,candidates)
    if len(edges):edges[:,:2]=np.asarray(ids)[edges[:,:2].astype(int)]
    n=len(records);content=partition(n,edges,.90);title=group_of([norm_title(r['title']) for r in records])
    union_edges=np.concatenate([edges[edges[:,2]>=.90],np.array([(int(ix[0]),int(i),1.) for c in np.unique(title)
        if len(ix:=np.flatnonzero(title==c))>1 for i in ix[1:]]).reshape(-1,3)],axis=0)
    groups=partition(n,union_edges,.90);ng=groups.max()+1
    perm=np.random.default_rng(7).permutation(ng);held=np.isin(groups,perm[:int(.2*ng)])
    y=np.array([r['class'] for r in records]);assert set(y[held])==set(y[~held])==set(range(5))
    weights={'documents':np.ones(n),'title':1./np.bincount(title)[title],'content':1./np.bincount(content)[content]}
    ev={k:v[held]/v[held].sum() for k,v in weights.items()}
    X=features([r['body'] for r in records]);sparse.save_npz(BBC/'features.npz',X)
    np.savez_compressed(BBC/'cohort.npz',y=y,held=held,groups=groups,title_groups=title,content_groups=content,
                        ids=np.array([r['id'] for r in records]),**weights)
    np.savez_compressed(BBC/'evaluation.npz',y=y[held],groups=groups[held],held_ids=np.array([r['id'] for r in records])[held],**ev)
    laws={k:v/v.sum() for k,v in weights.items()}
    D={a:{b:float(.5*np.abs(laws[a]-laws[b]).sum()) for b in laws} for a in laws}
    repetitions={}
    for name,z in [('title',title),('content',content)]:
        sizes=np.bincount(z);K=len(sizes)
        repetitions[name]={'K':K,'kappa':n/K,'member':float(sizes[sizes>1].sum()/n),
            'cluster':float((sizes>1).mean()),'extra':1-K/n,'TV_vs_documents':D['documents'][name]}
    midpoint=(laws['title']+laws['content'])/2;R=max(.5*np.abs(midpoint-laws[k]).sum() for k in ['title','content'])
    assert abs(R-D['title']['content']/2)<1e-12
    for law,w in laws.items():
        for cls in range(5):assert abs((w-laws['documents'])@(y==cls))<=D['documents'][law]+1e-12
    dump(BBC/'data_manifest.json',{'dataset':'BBC fulltext author release','url':url,'git_blob_sha1':git,
        'zip_sha256':hashlib.sha256(data).hexdigest(),'source_readme_sha256':sha(BBC/'SOURCE_README.md'),
        'rights':'BBC content copyright; author README permits non-commercial research; repository code license does not relicense article text',
        'n':n,'native_classes':classes,'native_class_counts':np.bincount(y).tolist(),'n_train':int((~held).sum()),
        'n_test':int(held.sum()),'test_union_groups':int(len(np.unique(groups[held]))),
        'train_class_counts':np.bincount(y[~held],minlength=5).tolist(),'test_class_counts':np.bincount(y[held],minlength=5).tolist(),
        'shingle_eligible':len(ids),'period_rule_not_applied_to_native_curated_benchmark':True,
        'metadata_unavailable':['URL','calendar date','multiple publishers'],'title_excluded_from_classifier_body':True,
        'split':'union of exact normalized title and content .90 components; seed7, shared across conditions',
        'repetitions':repetitions,'TV':D,'midpoint_R':float(R),'bound_B':D['title']['content']/2,
        'feature_shape':list(X.shape),'feature_nnz':int(X.nnz),'implementation_sha256':sha(Path(__file__)),
        'corpus_protocol_sha256':sha(OUT/'protocol.json')})
    dump(BBC/'frozen_inputs.json',{name:sha(BBC/name) for name in ['features.npz','cohort.npz','evaluation.npz','records.jsonl','data_manifest.json']})
    print(json.dumps({'prepared_BBC':True,'n':n,'n_train':int((~held).sum()),'n_test':int(held.sum()),'TV':D}))

def train():
    assert not (BBC/'training_complete.json').exists()
    for name,h in json.loads((BBC/'frozen_inputs.json').read_text()).items():assert sha(BBC/name)==h
    p=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'BBC01_v2_protocol.json').read_text())['implementation_sha256']==sha(Path(__file__))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.38);torch.set_num_threads(1)
    c=np.load(BBC/'cohort.npz');ev=np.load(BBC/'evaluation.npz');X=sparse.load_npz(BBC/'features.npz');tr=~c['held']
    Xtr=torch.from_numpy(X[tr].toarray()).cuda();Xte=torch.from_numpy(X[~tr].toarray()).cuda();y=torch.from_numpy(c['y'][tr]).cuda()
    runs=BBC/'runs';runs.mkdir(exist_ok=True);count=0;seconds=0.
    for model in ['linear','MLP64']:
        for arm in ['documents','title','content']:
            for seed in range(11,16):
                consumed=sum(json.loads(q.read_text())['fit_seconds'] for q in OUT.glob('*/progress.json'))
                assert consumed<p['compute_policy']['maximum_total_new_fit_wall_seconds']
                name=f'{model}_{arm}_{seed}';dest=runs/f'{name}.npz';assert not dest.exists()
                saved,trace,diag,elapsed=fit(Xtr,Xte,y,c[arm][tr],seed,model=model,normalization='batch')
                if model=='linear':Z=X[~tr]@saved['param_0']+saved['param_1']
                else:Z=np.maximum(X[~tr]@saved['param_0']+saved['param_1'],0)@saved['param_2']+saved['param_3']
                assert np.array_equal(Z.argmax(1),saved['epoch_96'])
                saved['held_ids']=ev['held_ids'];np.savez_compressed(dest,**saved)
                dump(runs/f'{name}.json',{'model':model,'arm':arm,'seed':seed,'trace':trace,'diagnostics':diag,
                    'fit_seconds':elapsed,'CPU_saved_model_predictions_match':True,
                    'max_CPU_CUDA_logit_difference':float(np.max(np.abs(Z-saved['logits_96']))),
                    'errors':{str(e):{law:float(ev[law]@(saved[f'epoch_{e}']!=ev['y'])) for law in ['documents','title','content']} for e in [24,48,96]},
                    'input_manifest_sha256':sha(BBC/'frozen_inputs.json'),'prediction_sha256':sha(dest)})
                count+=1;seconds+=elapsed;progress={'done':count,'total':30,'last':name,'fit_seconds':seconds}
                dump(BBC/'progress.json',progress);print(json.dumps(progress),flush=True)
    dump(BBC/'training_complete.json',{'done':count,'fit_seconds':seconds,'peak_VRAM_GiB':torch.cuda.max_memory_allocated()/2**30})

def analyse():
    from analyse_cc import ci
    assert json.loads((BBC/'training_complete.json').read_text())['done']==30
    e=np.load(BBC/'evaluation.npz');out={'primary_epoch':96,'native_labels':True,'laws':['documents','title','content'],'models':{}}
    for model in ['linear','MLP64']:
        er={a:(np.array([np.load(BBC/'runs'/f'{model}_{a}_{s}.npz')['epoch_96'] for s in range(11,16)])!=e['y']).mean(0) for a in ['documents','title','content']}
        errors={a:{law:float(e[law]@v) for law in out['laws']} for a,v in er.items()}
        # Direct resampling recomputes both ratios for fixed-prediction rescoring.
        inv=np.unique(e['groups'],return_inverse=True)[1];ng=inv.max()+1;rng=np.random.default_rng(20261116);draws=[]
        n1=np.bincount(inv,weights=e['content']*er['documents']);d1=np.bincount(inv,weights=e['content'])
        n0=np.bincount(inv,weights=e['documents']*er['documents']);d0=np.bincount(inv,weights=e['documents'])
        for _ in range(20):
            counts=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(250)])
            draws.extend((counts@n1/(counts@d1)-counts@n0/(counts@d0)).tolist())
        out['models'][model]={'errors':errors,'fixed_document_model_scoring_span':float(np.ptp(list(errors['documents'].values()))),
            'three_arm_refit_span_under_content':float(np.ptp([errors[a]['content'] for a in er])),
            'content_minus_documents_refit':ci(er['content']-er['documents'],e['content'],e['groups']),
            'title_minus_documents_refit':ci(er['title']-er['documents'],e['content'],e['groups']),
            'fixed_document_model_content_minus_documents':{'point':errors['documents']['content']-errors['documents']['documents'],
                'SE':float(np.std(draws,ddof=1)),'percentile95':np.quantile(draws,[.025,.975]).tolist()}}
    dump(BBC/'results.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','train','analyse']);args=ap.parse_args()
    with threadpool_limits(limits=1):globals()[args.action]()
