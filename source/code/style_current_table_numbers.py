"""Rank native table means from verified unrounded results; preserve all numeric strings."""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/runs/REVIEW_20261116_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
def unmark(text):return re.sub(r'\\(?:textbf|PaperBestMean)\{(0\.\d{4})\}',r'\1',text)
def contrast(a,b):
    def lum(h):
        rgb=np.array([int(h[i:i+2],16)/255 for i in (0,2,4)]);rgb=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
        return float(rgb@np.array([.2126,.7152,.0722]))
    x,y=sorted([lum(a),lum(b)]);return (y+.05)/(x+.05)

def main(paper):
    files=['audit_repetition_table_styled.tex','audit_task_table_styled.tex','audit_controls_table.tex']
    before={name:(paper/name).read_text(encoding='utf-8') for name in files};normalized={k:digest(unmark(v)) for k,v in before.items()}
    task=read(ROOT/'paper/www/audit/training_evaluation.json')['results']
    arms=['documents',*[f'tau={t:.2f}' for t in [.50,.60,.70,.80,.85,.90,.95]],'url','title','outlet']
    laws=['documents','url','title_date','title','lsh','mixture','outlet']
    values=np.array([[task[a][k]['error'] for k in laws] for a in arms]);best=values==values.min(0)
    lines=unmark(before[files[1]]).splitlines(keepends=True);row=0
    for i,line in enumerate(lines):
        matches=list(re.finditer(r'0\.\d{4}',line))
        if len(matches)!=7:continue
        assert row<11 and [m.group() for m in matches]==[f'{v:.4f}' for v in values[row]]
        col=iter(range(7))
        def mark(match):
            j=next(col);value=match.group();return r'\PaperBestMean{'+value+'}' if best[row,j] else value
        lines[i]=re.sub(r'0\.\d{4}',mark,line);row+=1
    assert row==11
    table2=''.join(lines)
    summary=read(ROOT/'paper/www/audit/optimization_20261115/summary.json');c01=read(ROOT/'experiments/runs/C01_v1/results.json')
    budgets=[24,48,96];train_arms=['documents','content','outlet']
    A=np.array([[summary['training_budget'][str(e)][a]['error']['title_date'] for e in budgets] for a in train_arms])
    B=np.array([summary['matched_placebo']['averaged_error']['title_date'],
        c01['laws']['title_date']['unrestricted_first_five'],c01['laws']['title_date']['class_stratified_permutation']])
    raw=np.r_[A.ravel(),B];marked=np.r_[(A==A.min(0)).ravel(),False,*(B[1:]==B[1:].min())]
    table3=unmark(before[files[2]]);seen=re.findall(r'0\.\d{4}',table3)
    assert seen==[f'{v:.4f}' for v in raw],(seen,raw)
    count=iter(range(12))
    def mark3(match):
        j=next(count);return r'\PaperBestMean{'+match.group()+'}' if marked[j] else match.group()
    table3=re.sub(r'0\.\d{4}',mark3,table3)
    assert unmark(table2)==unmark(before[files[1]]) and unmark(table3)==unmark(before[files[2]])
    (paper/files[1]).write_text(table2,encoding='utf-8');(paper/files[2]).write_text(table3,encoding='utf-8')
    style=paper/'paper_tables.sty';sty=style.read_text(encoding='utf-8');original_style=sty
    declaration_macro=r'\newcommand{\PaperBestMean}[1]{{\color{black}\bfseries #1}}'
    macro=r'\newcommand{\PaperBestMean}[1]{\textcolor{black}{\textbf{#1}}}'
    if declaration_macro in sty:sty=sty.replace(declaration_macro,macro)
    elif macro not in sty:
        assert r'\PaperBestMean' not in sty
        sty+='\n% Numerical rank only; no uncertainty or neighboring cell inherits this weight.\n'+macro+'\n'
    style.write_text(sty,encoding='utf-8')
    source=paper/'paper_short_polished.tex';text=source.read_text(encoding='utf-8')
    old=r'\caption{Title-and-day error over five shuffle seeds. (a) Training budgets. (b) 24-epoch controls, averaged over the indicated permutations.}'
    previous_caption=r'\caption{Title-and-day error over five shuffle seeds. (a) Training budgets. (b) 24-epoch controls, averaged over the indicated permutations. Bold marks the lowest unrounded mean per budget in (a), or among the five-permutation controls in (b).}'
    new=r'\caption{Mean title-and-day errors; five shuffles and, in (b), listed permutations. Bold uses unrounded column minima (a) or five-permutation minima (b).}'
    assert old in text or previous_caption in text or new in text
    text=text.replace(old,new).replace(previous_caption,new);source.write_text(text,encoding='utf-8')
    colors=['FFFFFF','E5EFF5','F2F7FA','F7F7F7','FCF6EE'];ratios={c:contrast('000000',c) for c in colors}
    assert min(ratios.values())>=4.5
    audit={'scope':'native main-text result tables 1--3; current four-page follow-up',
      'raw_results_changed':False,'numeric_precision_and_signs_changed':False,'headers_backgrounds_widths_fonts_unchanged':True,
      'ranking_uses':'exact stored unrounded seed means; exact numerical ties all marked; rank does not imply significance',
      'secondary_rank_enabled':False,'delta_columns_present':False,'new_delta_columns':False,
      'numeric_colors':'absolute means black; no red/blue without a delta; no uncertainty fields present in these tables',
      'table1':'descriptive cluster counts/repetition measures; no performance direction and no ranking',
      'table2':{'direction':'error down','group':'11 training weightings, same 24-epoch learner/data/split/five seeds; per evaluation column',
        'winners':{law:[{'arm':arms[i],'unrounded_mean':float(values[i,j])} for i in range(11) if best[i,j]] for j,law in enumerate(laws)},
        'numeric_cells':77,'bold_means':int(best.sum()),'unchanged_original_winner_count':7},
      'table3':{'direction':'error down','a_groups':'separate 24/48/96 budgets; compare three training arms within each column',
        'a_unrounded_means':A.tolist(),'a_bold_means':int((A==A.min(0)).sum()),
        'b_groups':'compare only matched five-permutation controls; ten-permutation mean includes overlapping assignments and is excluded from rank',
        'b_unrounded_means':B.tolist(),'b_bold_means':int(marked[-3:].sum()),'numeric_cells':12},
      'new_bold_means':4,'numeric_source_normalized_sha256':normalized,'black_text_contrast_by_background':ratios,
      'semantic_macro':'PaperBestMean; inline textcolor/textbf black mean; uncertainty and neighboring cells untouched',
      'source_after_sha256':{name:hashlib.sha256((paper/name).read_bytes()).hexdigest() for name in files},
      'result_sources':['paper/www/audit/training_evaluation.json','paper/www/audit/optimization_20261115/summary.json','experiments/runs/C01_v1/results.json']}
    # Keep an inspectable pre-edit record, including the original mark positions.
    audit_path=RUN/'table_numbers_audit.json';previous=read(audit_path) if audit_path.exists() else {}
    def tokens(text):
        return [{'display':m.group(2),'bold':bool(m.group(1))} for m in re.finditer(r'(\\(?:textbf|PaperBestMean)\{)?(0\.\d{4})',text)]
    initial=previous.get('before_tokens')
    if initial is None:
        initial={name:tokens((ROOT/'paper/www'/name).read_text(encoding='utf-8') if paper.resolve()!=(ROOT/'paper/www').resolve() else before[name]) for name in files[1:]}
    audit['before_tokens']=initial;audit['after_tokens']={name:tokens((paper/name).read_text(encoding='utf-8')) for name in files[1:]}
    for name in files[1:]:assert [x['display'] for x in initial[name]]==[x['display'] for x in audit['after_tokens'][name]]
    audit['precision']={'table1_cluster_counts':'comma-separated integers','table1_rates':1,'table2_means':4,'table3_means':4}
    audit['raw_source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in audit['result_sources']}
    (RUN/'table_numbers_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'table2_verified_best_means':int(best.sum()),'table3_new_best_means':int(marked.sum()),
        'all_105_numeric_cells_unchanged':True,'minimum_black_text_contrast':min(ratios.values()),'new_colors_or_deltas':False},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paper-dir',type=Path,default=ROOT/'paper/www');a=p.parse_args();main(a.paper_dir)
