"""GT-only direction disagreement, NOT a reconstructed result."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_line_support_refinement import read_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();run=Path(args.run)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,name in enumerate(['final_016_scene_00199','final_008_scene_00190']):
        r=read_json(Path('local_outputs/independent_evidence_train4_dev2_20261024_v1')/f'{name}_prediction_inputs.json')
        v=r['views'][0]['transform'];h,w=v['pointmap_hw']
        with np.load(r['source']) as z:xy=z['pixel_xy'];views=z['source_views']
        with np.load(run/f'{name}_supervision_diagnostic.npz') as z:
            ix=z['source_indices'];mask=views[ix]==v['alignment_view_index'];coords=xy[ix[mask]]
            boundary=np.full((h,w),np.nan);angle=boundary.copy()
            boundary[coords[:,1],coords[:,0]]=z['boundary'][mask]
            q=z['direction_angle_degrees'][mask].copy();q[~z['eligible'][mask]]=np.nan
            angle[coords[:,1],coords[:,0]]=q
        axes[row,0].imshow(boundary,cmap='gray',vmin=0,vmax=1)
        im=axes[row,1].imshow(angle,cmap='magma',vmin=0,vmax=90)
        axes[row,0].set_title(name+' | GT k4 boundary');axes[row,1].set_title('Global vs same-plane target angle')
        for ax in axes[row]:ax.axis('off')
    fig.colorbar(im,ax=axes[:,1].tolist(),label='degrees (clipped display at90)')
    fig.suptitle('Supervision diagnostic only: GT used, not model predictions',fontsize=13)
    fig.savefig(run/'direction_target_diagnostic.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
