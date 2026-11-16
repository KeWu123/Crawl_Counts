from pathlib import Path
import hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
EXT=ROOT/'experiments/runs/EXT_20261116_v1'
ZIP=EXT/'semantic_reference_probe/3sources.zip'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()
def ahash(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256((str(x.dtype)+str(x.shape)).encode()+x.tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def settings():
    return [('CC / Linear',ROOT/'paper/www/audit/optimization_20261115/runs','budget_',ROOT/'paper/www/audit/evaluation.npz','lsh','title_day'),
      ('CC / MLP',EXT/'L01/runs','',ROOT/'paper/www/audit/evaluation.npz','lsh','title_day'),
      ('BBC / Linear',EXT/'BBC01_v2/runs','linear_',EXT/'BBC01_v2/evaluation.npz','content','groups'),
      ('BBC / MLP',EXT/'BBC01_v2/runs','MLP64_',EXT/'BBC01_v2/evaluation.npz','content','groups')]
def matched_inputs(setting):
    name,folder,prefix,ep,law,gkey=setting
    with np.load(ep) as z:ev={k:z[k].copy() for k in ['y','documents',law,gkey,'held_ids']}
    pred={}
    for arm in ['documents','content']:
        values=[]
        for seed in range(11,16):
            with np.load(folder/f'{prefix}{arm}_{seed}.npz') as z:values.append(z['epoch_96'].copy())
        pred[arm]=np.array(values)
    return ev,pred
