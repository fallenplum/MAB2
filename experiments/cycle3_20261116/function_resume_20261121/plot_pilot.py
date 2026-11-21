"""Plot every preregistered comparison on the same 16 pilot tasks."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CONTROLS = ("SINGLE_Q", "SINGLE_C", "BOK3", "BOK3_EARLY", "REPAIR3")
NAMES = ("Single 3B", "Single coder 7B", "Best of 3", "Best of 3, early stop", "Repair, up to 3")


def draw(result, out):
    assert result["tasks"] == 16 and result["seeds"] == 3
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.45), sharey=True)
    colors = ("#236b86", "#32897b", "#9a6328")
    seeds = sorted(result["conditions"]["VS_GRAPH"]["by_seed"])
    for ax, key, factor, threshold, xlabel in zip(axes, ("quality","utility"), (100,1), (-2,.01),
            ("Task success difference (percentage points)", "Net utility difference")):
        ax.axvline(0, color="#747474", linewidth=.9, zorder=0)
        ax.axvline(threshold, color="#9d3948", linestyle="--", linewidth=1, zorder=0)
        for i,control in enumerate(CONTROLS):
            for j,seed in enumerate(seeds):
                scores=result["conditions"]
                delta=(scores["VS_GRAPH"]["by_seed"][seed][key]-scores[control]["by_seed"][seed][key])*factor
                ax.scatter(delta,i+(j-1)*.13,s=22,color=colors[j],zorder=3)
            mean=result["paired_comparisons"][control][key+"_difference"]*factor
            ax.scatter(mean,i,s=51,marker="D",facecolor="white",edgecolor="black",linewidth=1.1,zorder=5)
        ax.set_xlabel(xlabel,fontsize=9)
        ax.grid(axis="y",alpha=.22)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.margins(x=.16)
    axes[0].set_yticks(range(5),NAMES)
    axes[0].invert_yaxis()
    handles=[Line2D([],[],marker="D",markerfacecolor="white",color="black",lw=0,label="Three-seed mean")]
    handles.extend(Line2D([],[],marker="o",color=c,lw=0,label=f"Seed {s[-3:]}") for s,c in zip(seeds,colors))
    handles.append(Line2D([],[],ls="--",color="#9d3948",label="Prespecified threshold"))
    fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=8,frameon=False)
    fig.suptitle("Visible-test graph selection versus each fixed control",fontsize=12)
    fig.subplots_adjust(left=.19,right=.985,top=.86,bottom=.23,wspace=.13)
    for fmt in ("pdf","png"):
        fig.savefig(out / ("PILOT_COMPARISONS."+fmt),dpi=220,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("summary",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();draw(json.loads(a.summary.read_bytes()),a.out)
