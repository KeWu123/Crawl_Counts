"""Portable offline entry for the anonymous counting-target artifact."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def coverage(root):
    with np.load(root / 'content_laws.npz') as laws, np.load(root / 'reference_evaluation.npz') as ref:
        target = ref['reference'].copy()
        topics = sorted(np.unique(ref['topic']).tolist())
        expected = json.loads((root / 'expected_results.json').read_text(encoding='utf-8'))['coverage']
        rows, sets = [], []
        for spec in expected['sets']:
            names = spec['candidate_names']
            candidates = np.array([laws['law_' + name] for name in names])
            center = laws['center_' + spec['set']]
            assert np.all(candidates >= 0) and np.allclose(candidates.sum(1), 1)
            tv = lambda p, q: float(0.5 * np.abs(p-q).sum())
            item = {'set': spec['set'], 'candidates': names,
                    'R': max(tv(center, p) for p in candidates),
                    'reference_TV': tv(center, target), 'epsilon': min(tv(p, target) for p in candidates),
                    'outside_count': 0, 'topic_count': len(topics)}
            for topic in topics:
                f = ref['topic'] == topic
                means = candidates @ f
                lo, hi = float(min(means)), float(max(means))
                truth, mean = float(target @ f), float(center @ f)
                gap = max(lo-truth, truth-hi, 0.0)
                outside = gap > 1e-12
                direction = 'underestimate' if truth > hi + 1e-12 else 'overestimate' if truth < lo - 1e-12 else 'inside'
                assert np.max(np.abs(np.array([lo,hi])-spec['candidate_ranges'][topic])) < 1e-12
                row = {'candidate_set': spec['set'], 'topic': topic,
                       'candidate_min_pct': 100*lo, 'candidate_max_pct': 100*hi,
                       'story_reference_pct': 100*truth, 'distance_to_range_pp': 100*gap,
                       'direction': direction, 'reference_outside': outside}
                rows.append(row)
                item['outside_count'] += int(outside)
                if topic == 'health':
                    item['health'] = {**row, 'center_pct': 100*mean,
                        'radius_band_low_pct': 100*max(0,mean-item['R']),
                        'radius_band_high_pct': 100*min(1,mean+item['R']),
                        'all_candidate_means_below_reference': bool(np.all(means < truth)),
                        'minimum_shortfall_pp': 100*max(truth-hi, 0)}
            sets.append(item)
    return {'sets': sets, 'topics': topics, 'rows': rows,
            'selection': 'Health is an exploratory illustration from the baseline audit; all six native topics are reported.',
            'interpretation': 'For each topic separately, mixture expectations fill the interval between candidate means. A positive distance is unavoidable for every mixture in that declared set; zero does not establish a jointly attainable six-topic vector.',
            'units': 'Shares are percent; scalar distances and shortfalls are percentage points; R/reference_TV/epsilon are probability units.',
            'new_training_fits': 0}


def write_report(result, out):
    out.mkdir(parents=True, exist_ok=True)
    (out/'coverage_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (out/'all_topic_ranges.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(result['rows'][0]))
        writer.writeheader();writer.writerows(result['rows'])
    lines=['# Counting-target audit: core results', '',
           'Recomputed from frozen per-record laws, topic labels and the supplied story reference. No fitting or threshold search is performed.', '',
           '## Candidate coverage', '',
           '| Candidate set | R (%) | Center-to-reference TV (%) | Topics outside / 6 | Minimum health shortfall (pp) |',
           '| --- | ---: | ---: | ---: | ---: |']
    for s in result['sets']:
        lines.append(f"| {s['set']} | {100*s['R']:.6f} | {100*s['reference_TV']:.6f} | {s['outside_count']} | {s['health']['minimum_shortfall_pp']:.6f} |")
    lines += ['', 'For a topic indicator f, E_mix[f] is the convex combination of candidate means. Its attainable scalar interval is [min mean, max mean]. The reference health share is above every candidate in each tested set, so every allowed mixture underestimates it. This concerns these declared finite candidate sets.', '',
              result['selection'], '', '## All native topics', '',
              '| Set | Topic | Candidate min (%) | Candidate max (%) | Story reference (%) | Distance (pp) | Direction |',
              '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in result['rows']:
        lines.append(f"| {r['candidate_set']} | {r['topic']} | {r['candidate_min_pct']:.6f} | {r['candidate_max_pct']:.6f} | {r['story_reference_pct']:.6f} | {r['distance_to_range_pp']:.6f} | {r['direction']} |")
    lines += ['', result['interpretation'], '',
              '## Distinct diagnostics', '',
              'Linkage precision/recall compares partition membership. TV compares per-record counting weights, which depend on K and the size m(i) of each record\'s group. Two different pairings of four records into two size-two groups have identical 1/4 record weights and zero counting TV even with disjoint linked pairs.', '',
              'Exact candidate ranges, center-plus/minus-radius bands, and paired bootstrap intervals are different quantities. The cosine .50 center radius band covers health; its exact candidate range still excludes the reference.', '',
              '## Reproduce', '',
              'Run `python reproduce_core.py` from the extracted artifact. It verifies payload checksums, recomputes all 77 Table 2 cells and all four H intervals (5,000 joint test-cluster resamples each), plus the original controls and extension results, then writes this diagnostic under `recomputed/`.', '',
              'Optional: `python reproduce_core.py --coverage-only --plot` produces a standalone PNG/SVG rendering of the Figure 2(a,b) data. It uses the frozen arrays; the manuscript figure remains the selected original.', '',
              'H is a post-hoc conditional description with models, partitions and splits fixed. All six topics are reported without selecting outcomes. Raw article text is not redistributed.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,
                         'axes.linewidth':.7,'svg.fonttype':'none','svg.hashsalt':'counting-audit'})
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.4),layout='constrained')
    labels=['Documents + publisher','+ cosine .50','+ cosine .70']
    yy=np.arange(3)
    ax=axes[0]
    ax.barh(yy-.16,[100*s['R'] for s in result['sets']],height=.28,color='#9AA3A7',edgecolor='#30363A',label='Radius R')
    ax.barh(yy+.16,[100*s['reference_TV'] for s in result['sets']],height=.28,color='#315F78',label='Reference distance')
    ax.set(yticks=yy,yticklabels=labels,xlabel='TV distance (%)',xlim=(0,15),title='(a) Candidate coverage')
    ax.invert_yaxis();ax.legend(frameon=False,fontsize=7,loc='lower right')
    ax=axes[1]
    base=result['sets'][0]['health']
    ax.barh(0,base['radius_band_high_pct']-base['radius_band_low_pct'],left=base['radius_band_low_pct'],height=.22,color='#CCD2D5')
    for i,s in enumerate(result['sets']):
        h=s['health'];ax.plot([h['candidate_min_pct'],h['candidate_max_pct']],[i,i],color='#30363A',lw=2)
        ax.plot(h['center_pct'],i,'o',mfc='white',mec='#30363A',ms=4)
    ax.axvline(base['story_reference_pct'],color='#315F78',ls='--',lw=1)
    ax.set(yticks=yy,yticklabels=labels,xlabel='Health share (%)',xlim=(7.8,14),title='(b) Exact candidate ranges')
    ax.invert_yaxis();ax.text(.02,.02,'Gray: baseline center +/- R; not a CI',transform=ax.transAxes,fontsize=6.7)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'figure2ab.png',dpi=600)
    fig.savefig(out/'figure2ab.svg',metadata={'Date':None})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--coverage-only',action='store_true')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if not args.coverage_only:
        import sys
        sys.path.insert(0,str(root))
        import recompute
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):recompute.main(root)
    out=args.output or root/'recomputed'
    result=coverage(root);write_report(result,out)
    if args.plot:plot(result,out)
    print(json.dumps({'coverage_rows':len(result['rows']),
                     'outside_counts':[s['outside_count'] for s in result['sets']],
                     'health_minimum_gaps_pp':[s['health']['minimum_shortfall_pp'] for s in result['sets']],
                     'training_started':False}))


if __name__=='__main__':main()
