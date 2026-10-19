"""Summarize the completed three-sample small/large circle inverse experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads((args.out/'results.json').read_text())
    rows=result['samples']
    if len(rows)!=36 or any(row['iteration_ms'] is None for row in rows):
        raise RuntimeError('Wait for all 36 reconstructions and isolated timing measurements')
    with (args.out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    methods=('D-Flow','P-Flow','DLR D-Flow')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,problem in enumerate(('ct10','mri4')):
        for column,key in enumerate(('image_error_percent','iteration_ms')):
            ax=axes[row,column]
            for size,offset,color,label in [('large',-.18,'#737373','472k model'),('small',.18,'#2478b4','22k model')]:
                groups=[[v[key] for v in rows if (v['size'],v['problem'],v['method'])==(size,problem,method)] for method in methods]
                medians=list(map(statistics.median,groups))
                positions=np.arange(3)+offset
                ax.bar(positions,medians,width=.34,color=color,alpha=.8,label=label)
                for x,values in zip(positions,groups):
                    ax.scatter(x+np.linspace(-.06,.06,3),values,s=25,color=color,edgecolors='black',linewidths=.5,zorder=3)
                for x,value in zip(positions,medians):
                    ax.annotate(f'{value:.2f}',(x,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(3),methods)
            ax.set_ylabel('Relative image L2 error (%)' if column==0 else 'Compute time per iteration (ms)')
            ax.set_title(('CT, 10 angles' if row==0 else 'MRI, 4× acceleration')+(' — accuracy' if column==0 else ' — no line search'))
            ax.margins(y=.22);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0,0].legend()
    fig.suptitle('64² single circles · 3 paired samples · noiseless · 5 Euler steps\nBars: medians; dots: individual samples',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(args.out/'accuracy_speed_comparison.png',dpi=170)
    fig.savefig(args.out/'accuracy_speed_comparison.pdf')
    plt.close(fig)
    print(json.dumps(result['aggregate'],indent=2))


if __name__=='__main__':
    main()
