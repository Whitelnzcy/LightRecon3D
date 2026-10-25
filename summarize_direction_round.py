"""Audited paired development controls; no cherry-picked best-scene summary."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_line_support_refinement import read_json,write_json,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);rows=[]
    for p in args.runs:
        run=Path(p);protocol=read_json(run/'protocol.json');completion=read_json(run/'completion.json');audit=read_json(run/'audit.json')
        assert completion['status']=='complete' and set(completion['aggregate'])=={'none','learned'}
        assert len(audit)==2 and all(r['source_and_indices_verified'] for r in audit.values())
        rows.append(dict(run=str(run),seed=protocol['seed'],steps=protocol['steps'],boundary_target=protocol['boundary_target'],
            balance=protocol.get('direction_balance','uniform'),aggregate=completion['aggregate'],completion_sha256=sha(run/'completion.json')))
    fig,axes=plt.subplots(1,3,figsize=(15,max(4,len(rows)*.62)),sharey=True)
    labels=[f'{r["boundary_target"]} / {r["balance"]}\n{r["steps"]} steps, seed {r["seed"]}' for r in rows]
    for ax,key,title in zip(axes,['after_errors','boundary_after','damaged'],['Total wrong points','Boundary wrong points (<=5px)','Newly damaged correct points']):
        for y,r in enumerate(rows):
            a,b=[r['aggregate'][arm][key] for arm in ['none','learned']]
            ax.plot([a,b],[y,y],color='#a8b5c7',lw=2)
            ax.scatter(a,y,color='#888888',s=40,label='No propagation' if y==0 else None,zorder=3)
            ax.scatter(b,y,color='#2375bb',s=40,label='Learned propagation' if y==0 else None,zorder=3)
            ax.annotate(f'{b-a:+,}',(max(a,b),y),xytext=(8,-4),textcoords='offset points',fontsize=9)
        ax.set_title(title+' | lower is better',fontsize=11);ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.margins(x=.35)
    axes[0].set_yticks(np.arange(len(rows)),labels,fontsize=9);axes[0].invert_yaxis();axes[0].legend(loc='best',fontsize=8)
    fig.suptitle('Fixed TRAIN4 / historical DEV2 | paired controls, not independent test results',fontsize=13)
    fig.text(.5,.01,'Numbers are learned minus matched no-propagation. Negative = improvement; positive = regression. Fixed XYZ and evaluation domain.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(out/'paired_controls.png',dpi=180);plt.close(fig)
    write_json(out/'report.json',dict(scope='full fixed historicalDEV2, development only',rows=rows))


if __name__=='__main__':main()
