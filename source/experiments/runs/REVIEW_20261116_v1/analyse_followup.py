"""Evaluate frozen feature candidates and jointly bootstrap the two matched paths."""
import csv,time,zipfile
from threadpoolctl import threadpool_limits
from common import *

def evaluate_content(protocol):
    frozen=read(HERE/'content_frozen.json')
    for name,h in frozen['outputs_sha256'].items():assert sha(HERE/name)==h
    assert all(name.endswith(('.terms','.mtx')) for name in frozen['files_read'])
    z=np.load(HERE/'content_laws.npz');ids=[]
    # Reference files are first opened only after feature partitions/centers are frozen.
    with zipfile.ZipFile(ZIP) as archive:
        for pub in ['bbc','guardian','reuters']:
            ids.extend(s.strip() for s in archive.read(f'3sources/3sources_{pub}.docs').decode().splitlines() if s.strip())
        classes={}
        for line in archive.read('3sources/3sources.disjoint.clist').decode().splitlines():
            label,values=line.split(':',1)
            for sid in values.split(','):
                if sid.strip():classes[sid.strip()]=label
    unique,story=np.unique(ids,return_inverse=True);assert (len(ids),len(unique))==(948,416)
    topics=sorted(set(classes.values()));labels=np.array([classes[s] for s in ids])
    reference=1/(len(unique)*np.bincount(story)[story])
    old=np.load(EXT/'S01_reference/laws.npz');assert abs(reference-old['linked_story']).max()<1e-12
    tv=lambda a,b:float(.5*abs(a-b).sum())
    law_names=['documents','publisher','content050','content070']
    laws={k:z['law_'+k] for k in law_names}
    proportions={k:{t:float(w@(labels==t)) for t in topics} for k,w in {**laws,'reference':reference}.items()}
    results=[]
    for tag,names in protocol['content']['candidate_sets'].items():
        record={'set':tag,'candidate_names':names,'candidate_proportions':{k:proportions[k] for k in names},
                'candidate_ranges':{t:[min(proportions[k][t] for k in names),max(proportions[k][t] for k in names)] for t in topics},
                'epsilon':min(tv(laws[k],reference) for k in names),'policies':{}}
        for policy,prefix in [('radius_optimal','center_'),('uniform','uniform_')]:
            center=z[prefix+tag];R=max(tv(center,laws[k]) for k in names)
            prop={t:float(center@(labels==t)) for t in topics};dist=tv(center,reference)
            h=float(prop['health']);hstar=proportions['reference']['health']
            record['policies'][policy]={'R':R,'B':frozen['centers'][tag]['B' if policy=='radius_optimal' else 'uniform_B'],
                'TV_to_reference':dist,'proportions':prop,
                'topic_error_pp':{t:100*(prop[t]-proportions['reference'][t]) for t in topics},
                'max_abs_topic_error_pp':max(100*abs(prop[t]-proportions['reference'][t]) for t in topics),
                'health_candidate_radius_band':[max(0.,h-R),min(1.,h+R)],
                'health_reference_outside_candidate_radius_band':not h-R-1e-12<=hstar<=h+R+1e-12,
                'reference_outside_candidate_radius':dist>R+1e-12,
                'R_plus_epsilon':R+record['epsilon']}
            assert dist<=R+record['epsilon']+1e-12
        results.append(record)
    pairs={};ref_pairs=int(np.sum(np.bincount(story)*(np.bincount(story)-1)//2))
    assert ref_pairs==701
    for name in ['content050','content070']:
        g=z['group_'+name];gc=np.bincount(g)
        _,cnt=np.unique(np.column_stack([g,story]),axis=0,return_counts=True)
        tp=int(np.sum(cnt*(cnt-1)//2));pred=int(np.sum(gc*(gc-1)//2))
        pairs[name]={**frozen['partitions'][name],'true_positive_pairs':tp,'predicted_pairs':pred,'reference_pairs':ref_pairs,
                     'pair_precision':tp/pred if pred else None,'pair_recall':tp/ref_pairs}
    np.savez_compressed(HERE/'reference_evaluation.npz',story=story,topic=labels,reference=reference)
    out={'candidate_construction_blind_to_reference':True,'n_records':948,'n_story_ids':416,
         'topics':topics,'all_law_proportions':proportions,'content_linkage_diagnostics':pairs,'sets':results,
         'scope':'finite released collection and supplied correspondence; no population sampling or CC-News detector validation'}
    write(HERE/'coverage_results.json',out)
    with (HERE/'topic_errors.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['candidate_set','center_policy','topic','center_proportion','reference_proportion','error_pp','candidate_min','candidate_max']);writer.writeheader()
        for rec in results:
            for pol,row in rec['policies'].items():
                for t in topics:writer.writerow({'candidate_set':rec['set'],'center_policy':pol,'topic':t,
                    'center_proportion':row['proportions'][t],'reference_proportion':proportions['reference'][t],
                    'error_pp':row['topic_error_pp'][t],'candidate_min':rec['candidate_ranges'][t][0],'candidate_max':rec['candidate_ranges'][t][1]})
    return out

def paired_H(protocol):
    old=read(EXT/'matched_pair_results.json')['rows'];rows=[];saved={}
    for index,setting in enumerate(settings()):
        name,_,_,_,law,gkey=setting;ev,pred=matched_inputs(setting);expected=protocol['used_matched_array_sha256'][name]
        assert {k:ahash(v) for k,v in ev.items()}==expected['evaluation_arrays']
        assert {k:ahash(v) for k,v in pred.items()}==expected['prediction_arrays']
        e={k:(v!=ev['y']).mean(0) for k,v in pred.items()}
        q=ev['documents'];w=ev[law];inv=np.unique(ev[gkey],return_inverse=True)[1];ng=inv.max()+1
        group=lambda v:np.bincount(inv,weights=v,minlength=ng)
        n0=group(q*e['documents']);d0=group(q);n1=group(w*e['documents']);d1=group(w);n2=group(w*e['content'])
        de=float(w@e['documents']-q@e['documents']);df=float(w@(e['content']-e['documents']))
        rng=np.random.default_rng(20261116);draws=[]
        for _ in range(20):
            counts=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(250)])
            e0=counts@n0/(counts@d0);e1=counts@n1/(counts@d1);e2=counts@n2/(counts@d1)
            draws.extend(np.column_stack([e1-e0,e2-e1]))
        draws=np.array(draws);H=abs(de)-abs(df);hd=abs(draws[:,0])-abs(draws[:,1]);lo,hi=np.quantile(hd,[.025,.975])
        assert abs(de-old[index]['delta_eval']['point'])<1e-12 and abs(df-old[index]['delta_fit']['point'])<1e-12
        assert np.max(abs(draws.std(0,ddof=1)-np.array([old[index]['delta_eval']['SE'],old[index]['delta_fit']['SE']])))<1e-12
        row={'setting':name,'n_test':len(e['documents']),'test_clusters':int(ng),'delta_eval_pp':100*de,'delta_fit_pp':100*df,
             'H_pp':100*H,'H_percentile95_pp':[100*float(lo),100*float(hi)],'H_SE_pp':100*float(hd.std(ddof=1)),
             'joint_covariance_pp2':(10000*np.cov(draws,rowvar=False)).tolist(),'interval_contains_zero':bool(lo<=0<=hi),
             'component_percentile95_pp':(100*np.quantile(draws,[.025,.975],axis=0)).tolist(),
             'conditional_scope':'fixed fitted models, partitions and split; post-hoc descriptive percentile interval'}
        rows.append(row);saved[f'components_{index}']=draws;saved[f'H_{index}']=hd
    np.savez_compressed(HERE/'paired_bootstrap_draws.npz',**saved)
    write(HERE/'H_results.json',{'definition':'abs(delta_eval)-abs(delta_fit)','resamples':5000,'rows':rows,
          'no_cross_corpus_significant_reversal_test':True,'nonsmooth_absolute_contrast':True})
    return rows

def main():
    started=time.perf_counter();assert not (HERE/'coverage_results.json').exists()
    p=read(HERE/'protocol.json')
    for name,h in p['implementation_sha256'].items():assert sha(HERE/name)==h
    coverage=evaluate_content(p);H=paired_H(p)
    ev=np.load(ROOT/'paper/www/audit/evaluation.npz');laws=['documents','url','title_date','title','lsh','mixture','outlet']
    assert all(abs(ev[k].sum()-1)<1e-12 for k in laws)
    write(HERE/'restricted_test_TV.json',{'support':'same 2156 CC-News held-out documents; full-corpus laws restricted then renormalized',
          'laws':laws,'TV':[[float(.5*abs(ev[a]-ev[b]).sum()) for b in laws] for a in laws]})
    write(HERE/'analysis_completion.json',{'status':'completed','wall_seconds':time.perf_counter()-started,
          'new_training_fits':0,'all_frozen_content_thresholds_reported':True,'paired_H_settings':4})
    print(json.dumps({'coverage':[{'set':r['set'],'epsilon':r['epsilon'],**r['policies']['radius_optimal']} for r in coverage['sets']],
                      'linkage':coverage['content_linkage_diagnostics'],'paired_H':H},indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
