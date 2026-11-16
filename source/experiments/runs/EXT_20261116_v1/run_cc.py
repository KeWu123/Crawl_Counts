"""Execute 15 N01 and 15 L01 trajectories, retaining all checkpoints."""
from pathlib import Path
import argparse,hashlib,json,time,traceback
import numpy as np
import psutil,torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from engine import fit

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];OLD=ROOT/'paper/www/audit/optimization_20261115'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    q=p.with_suffix('.pending');q.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');q.replace(p)
def main(block):
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for name,h in p['implementation_sha256'].items():assert sha(OUT/name)==h,name
    for name,h in p['sources_sha256'].items():assert sha(ROOT/name)==h,name
    assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.38);torch.set_num_threads(1)
    free,total=torch.cuda.mem_get_info();assert free>10*2**30 and psutil.virtual_memory().available>12*2**30
    start=time.perf_counter();c=np.load(OLD/'cohort.npz');ev=np.load(ROOT/'paper/www/audit/evaluation.npz')
    X=sparse.load_npz(OLD/'features.npz')
    if block=='L01':X=sparse.csr_matrix((X.data,c['active_hash_columns'][X.indices],X.indptr),shape=(len(c['y']),131072))
    tr=~c['held'];assert np.array_equal(c['ids'][~tr],ev['held_ids'])
    Xtr=torch.from_numpy(X[tr].toarray()).cuda();Xte=torch.from_numpy(X[~tr].toarray()).cuda();y=torch.from_numpy(c['y'][tr]).cuda()
    folder=OUT/block;folder.mkdir(exist_ok=True);runs=folder/'runs';runs.mkdir(exist_ok=True)
    write(folder/'preflight.json',{'free_VRAM_GiB':free/2**30,'GPU':torch.cuda.get_device_name(0),
        'model_input_shape':list(X.shape),'model':'linear' if block=='N01' else '64-hidden ReLU',
        'initialization':'zero' if block=='N01' else 'full 131072-dimensional historical NumPy RNG initializer',
        'seeds_change':'shuffle' if block=='N01' else 'initialization and shuffle, shared across arms'})
    count=0;fit_seconds=0.
    with threadpool_limits(limits=1):
        for arm in ['documents','content','outlet']:
            for seed in [11,12,13,14,15]:
                consumed=sum(json.loads(q.read_text())['fit_seconds'] for q in OUT.glob('*/progress.json'))
                assert consumed<p['compute_policy']['maximum_total_new_fit_wall_seconds'],'Frozen local compute limit reached.'
                name=f'{arm}_{seed}';dest=runs/f'{name}.npz';log=runs/f'{name}.json'
                assert not dest.exists() and not log.exists(),'No overwriting completed conditions.'
                result,trace,diag,seconds=fit(Xtr,Xte,y,c[arm][tr],seed,
                    model='linear' if block=='N01' else 'MLP64',normalization='global' if block=='N01' else 'batch')
                assert seconds<p['compute_policy']['max_single_fit_seconds']
                if block=='N01' and arm=='documents':
                    anchor=np.load(ROOT/'paper/www/audit/predictions.npz')['documents'][seed-11]
                    assert np.array_equal(result['epoch_24'],anchor),'Document normalization identity failed.'
                # Independently recompute final logits from saved parameters with SciPy CSR on CPU.
                if block=='N01':Z=X[~tr]@result['param_0']+result['param_1']
                else:Z=np.maximum(X[~tr]@result['param_0']+result['param_1'],0)@result['param_2']+result['param_3']
                assert np.array_equal(Z.argmax(1),result['epoch_96']),'CPU saved-model prediction mismatch.'
                err=float(np.max(np.abs(Z-result['logits_96'])))
                result['held_ids']=ev['held_ids'];np.savez_compressed(dest,**result)
                errors={str(e):{law:float(ev[law]@(result[f'epoch_{e}']!=ev['y'])) for law in
                    ['documents','url','title_date','title','lsh','mixture','outlet']} for e in [24,48,96]}
                write(log,{'block':block,'arm':arm,'seed':seed,'fit_seconds':seconds,'errors':errors,'trace':trace,
                    'diagnostics':diag,'cpu_saved_model_classifications_match':True,'CPU_CUDA_logit_max_difference':err,
                    'protocol_sha256':sha(OUT/'protocol.json'),'predictions_sha256':sha(dest)})
                count+=1;fit_seconds+=seconds
                write(folder/'progress.json',{'status':'running','done':count,'total':15,'last':name,'fit_seconds':fit_seconds,
                    'elapsed_seconds':time.perf_counter()-start,'last_title_day_error96':errors['96']['title_date']})
                print(json.dumps(json.loads((folder/'progress.json').read_text())),flush=True)
    write(folder/'progress.json',{'status':'training_completed','done':count,'total':15,'fit_seconds':fit_seconds,
        'elapsed_seconds':time.perf_counter()-start,'peak_VRAM_GiB':torch.cuda.max_memory_allocated()/2**30})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('block',choices=['N01','L01']);a=ap.parse_args()
    try:main(a.block)
    except Exception as ex:
        (OUT/(a.block+'_failure.txt')).write_text(traceback.format_exc(),encoding='utf-8');raise
