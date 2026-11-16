"""Standalone, offline reconstruction of the anonymous numeric audit bundle."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits

def main(root):
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    manifest=read('MANIFEST.json')
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    expected=read('expected_results.json');ev=np.load(root/'cc_evaluation.npz');pred=np.load(root/'cc_predictions.npz')
    g=np.load(root/'cc_partitions.npz');laws=['documents','url','title_date','title','lsh','mixture','outlet']
    repetitions={}
    for key in ['url','title_date','title','lsh']:
        counts=np.unique(g[key],return_counts=True)[1]
        repetitions[key]={'K':len(counts),'member':float(counts[counts>1].sum()/counts.sum()),
                         'cluster':float((counts>1).mean()),'extra':1-len(counts)/counts.sum()}
    crossing={}
    for arm in pred.files:
        error=(pred[arm]!=ev['y']).mean(0)
        crossing[arm]={law:float(ev[law]@error) for law in laws}
        for law in laws:assert abs(crossing[arm][law]-expected['crossing'][arm][law])<1e-12
    control=np.load(root/'cc_control_predictions.npz');table3=[]
    for arm in ['documents','content','outlet']:
        table3.append([float(ev['title_date']@(control[f'budget_{arm}_{epoch}']!=ev['y']).mean(0)) for epoch in [24,48,96]])
    for name,n in [('unrestricted',10),('unrestricted',5),('classmatched',5)]:
        error=(control[name][:n]!=ev['y']).mean((0,1));table3.append([float(ev['title_date']@error)])
    assert np.allclose(np.concatenate(table3),expected['table3'],rtol=0,atol=1e-12)
    ext=np.load(root/'extension_predictions.npz');bbc=np.load(root/'bbc_evaluation.npz');cells=0
    for job in read('extension_jobs.json'):
        e=bbc if job['block']=='BBC01_v2' else ev
        these=['documents','title','content'] if job['block']=='BBC01_v2' else laws
        for epoch in [24,48,96]:
            error=ext[f"{job['id']}_{epoch}"]!=e['y']
            for law in these:
                value=float(e[law]@error);assert abs(value-job['errors'][str(epoch)][law])<1e-12;cells+=1
    assert cells==900
    matched=np.load(root/'matched_inputs.npz');Hrows=[]
    for i,old in enumerate(expected['H']):
        y=matched[f'y_{i}'];q=matched[f'q_{i}'];w=matched[f'w_{i}'];groups=matched[f'g_{i}']
        d=(matched[f'doc_{i}']!=y).mean(0);c=(matched[f'content_{i}']!=y).mean(0)
        de=float((w-q)@d);df=float(w@(c-d));inv=np.unique(groups,return_inverse=True)[1];ng=inv.max()+1
        agg=lambda v:np.bincount(inv,weights=v,minlength=ng)
        n0,d0,n1,d1,n2=[agg(v) for v in [q*d,q,w*d,w,w*c]]
        rng=np.random.default_rng(20261116);samples=[]
        for _ in range(20):
            count=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(250)])
            a=count@n0/(count@d0);b=count@n1/(count@d1);c1=count@n2/(count@d1)
            samples.extend(abs(b-a)-abs(c1-b))
        interval=100*np.quantile(samples,[.025,.975]);point=100*(abs(de)-abs(df))
        assert abs(point-old['H_pp'])<1e-12 and np.max(abs(interval-old['H_percentile95_pp']))<1e-12
        Hrows.append({'setting':old['setting'],'H_pp':point,'conditional_percentile95_pp':interval.tolist()})
    ref=np.load(root/'reference_evaluation.npz');z=np.load(root/'content_laws.npz');target=ref['reference'];tv=lambda a,b:float(.5*abs(a-b).sum())
    coverage=[];cov=expected['coverage']
    for rec in cov['sets']:
        names=rec['candidate_names'];P=np.array([z['law_'+name] for name in names]);center=z['center_'+rec['set']]
        R=max(tv(center,p) for p in P);distance=tv(center,target);epsilon=min(tv(p,target) for p in P)
        for got,key in [(R,'R'),(distance,'TV_to_reference')]:assert abs(got-rec['policies']['radius_optimal'][key])<1e-12
        assert abs(epsilon-rec['epsilon'])<1e-12 and distance<=R+epsilon+1e-12
        for topic in cov['topics']:
            f=ref['topic']==topic;bounds=[min(float(p@f) for p in P),max(float(p@f) for p in P)]
            assert np.max(abs(np.array(bounds)-rec['candidate_ranges'][topic]))<1e-12
        coverage.append({'set':rec['set'],'R':R,'reference_TV':distance,'epsilon':epsilon})
    r01=read('cc_radius_optimal.json');v=np.array([len(g[k])/(len(np.bincount(g[k]))*np.bincount(g[k])[g[k]]) for k in ['url','title_date','title','lsh']])
    pi=np.array([r01['pi'][k] for k in ['url','title_date','title','lsh']]);radius=max(.5*abs(pi@v-row).mean() for row in v)
    assert abs(radius-r01['R'])<1e-12
    out={'status':'passed','crossing_cells':77,'original_control_values':12,'new_fit_error_cells':cells,
         'H':Hrows,'coverage':coverage,'CC_radius_optimal_R':float(radius),'training_started':False,
         'scope':'conditional fixed-model diagnostics and finite supplied reference; not a population guarantee'}
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=parser.parse_args()
    with threadpool_limits(limits=1):main(a.root)
