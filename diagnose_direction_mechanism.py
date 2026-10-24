"""Separate small-fit and GT-direction interventions; NOT deployable predictions."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from audit_line_support_refinement import read_json,write_json,sha,convert,metrics
from spatial_support_data import load_prediction_scene,attach_supervision,tile_boxes
from spatial_direction_data import DirectionInputs,attach_direction_supervision,direction_batch,direction_loss
from spatial_context_data import context_batch
from models.spatial_direction_refiner import SpatialDirectionRefiner
from models.line_support_patches import fuse_patch_logits
from train_spatial_line_ablation import predict,draw_comparison
from train_spatial_support_pilot import input_receipt


def expand_direction(target,k):
    return target[:,None].expand(-1,k,-1,-1,-1).reshape(-1,3,*target.shape[-2:])


def quality(d,t,y):
    v=y>=0;selected=d.gather(1,y.clamp_min(0)[:,None,None].expand(-1,1,3,-1,-1)).squeeze(1)
    ok=v&(torch.linalg.vector_norm(t,dim=1)>.5)
    angle=torch.rad2deg(torch.nn.functional.cosine_similarity(selected,t,dim=1).clamp(-1,1).acos())
    return dict(mse=float(direction_loss(d,t,y).detach().cpu()),
        median_angle=float(angle[ok].median().cpu()),valid=int(v.sum().cpu()))


@torch.inference_mode()
def intervened(model,s,mode):
    contributions=[]
    for v in s['views']:
        boxes=tile_boxes(v['registry'].shape)
        for start in range(0,len(boxes),4):
            batch=boxes[start:start+4];c,k,e=context_batch(v,batch,'cuda',0,False)
            if mode=='oracle':t=direction_batch(v,batch,'cuda')
            elif mode=='zero':t=torch.zeros_like(c[:,:3])
            else:
                t=torch.from_numpy(np.stack([v['random_direction'][:,y0:y1,x0:x1] for x0,y0,x1,y1 in batch])).cuda()
            replacement=expand_direction(t,k.shape[1])
            hook=model.line_direction.register_forward_hook(lambda module,inputs,output:replacement)
            try:logits=model(c,k,e).cpu().numpy()
            finally:hook.remove()
            for box,score in zip(batch,logits):
                x0,y0,x1,y1=box;indices=v['registry'][y0:y1,x0:x1];ok=indices>=0
                contributions.append((indices[ok],score.transpose(1,2,0)[ok]))
    return fuse_patch_logits(s['old']['point_plane_ids'],s['old']['plane_ids'],contributions)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    threadpool_limits(1);torch.set_num_threads(4);torch.manual_seed(42)
    base=Path('local_outputs/direction_train4_dev2_20261024_v2');models={}
    for arm in ['DS','DC']:
        m=SpatialDirectionRefiner(arm).cuda();m.load_state_dict(torch.load(base/f'{arm}.pt',map_location='cpu',weights_only=False)['model']);models[arm]=m.eval()
    write_json(out/'protocol.json',dict(schema_version=1,scope='TRAIN overfit + historicalDEV GT interventions ONLY',
        checkpoints={a:sha(base/f'{a}.pt') for a in models},overfit_steps=200,seed=42,
        overfit='TRAIN0 view0 fixed proposals0,1,64,65; directionMSE only; copied weights; never applied toDEV',
        intervention='frozen original DS/DC; oracle globalGT direction broadcast identically to ALL candidates; zero; fixed random3D unit direction',
        caveat='Post-training distribution shift; oracle replacement is NOT theoretical upper bound or retrained performance; no model selection'))
    provider=DirectionInputs(out)
    s=load_prediction_scene('local_outputs/gct3/items/train_000_scene_00000');provider.attach(s);attach_supervision(s);attach_direction_supervision(s)
    v=s['views'][0];boxes=[v['patches'][i]['box_xyxy'] for i in [0,1,64,65]]
    c,k,e,y,_=context_batch(v,boxes,'cuda');target=direction_batch(v,boxes,'cuda')
    overfit={}
    for arm,original in models.items():
        m=copy.deepcopy(original).train();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4)
        with torch.no_grad():_,d=m(c,k,e,return_details=True);before=quality(d,target,y)
        history=[]
        for step in range(200):
            opt.zero_grad();_,d=m(c,k,e,return_details=True);loss=direction_loss(d,target,y)
            loss.backward();opt.step()
            if step%25==0:history.append(dict(step=step+1,loss=float(loss.detach().cpu())))
        m.eval()
        with torch.no_grad():_,d=m(c,k,e,return_details=True);after=quality(d,target,y)
        overfit[arm]=dict(before=before,after=after,history=history,boxes=boxes)
        torch.save(dict(schema_version=1,model=m.state_dict(),scope='4patch direction-only overfit, not deployed'),out/f'{arm}_overfit_only.pt')
        write_json(out/'overfit.json',overfit);print('OVERFIT',arm,before,after,flush=True)
    del s,c,k,e,y,target
    report={};gtroot=Path('local_outputs/dense_boundary_competition_20260918_v1')
    for name in ['final_016_scene_00199','final_008_scene_00190']:
        s=load_prediction_scene(Path('local_outputs/b12r2/items')/name);path=provider.attach(s)
        write_json(out/f'{name}_prediction_inputs.json',dict(**input_receipt(s),lines_path=str(path),lines_sha256=sha(path)))
        predictions={a+'_learned':predict(m,s)[0] for a,m in models.items()}
        with np.load(base/f'{name}_predictions.npz') as z:
            assert all(np.array_equal(predictions[a+'_learned'],z[a]) for a in models)
        np.savez_compressed(out/f'{name}_learned_before_GT.npz',schema_version=1,**predictions)
        with np.load(gtroot/f'{name}_eval_only.npz') as z:
            np.testing.assert_array_equal(z['pixel_xy'],s['old']['pixel_xy']);np.testing.assert_array_equal(z['view_indices'],s['old']['source_views'])
            s.update(truth=z['raw_dense_gt'],distance=z['boundary_distance'],mapping=read_json(gtroot/'report.json')[name]['fixed_identity_map'])
        attach_direction_supervision(s)
        for v in s['views']:
            rng=np.random.default_rng(42+int(v['transform']['alignment_view_index']))
            random=rng.normal(size=(3,*v['registry'].shape)).astype(np.float32)
            v['random_direction']=random/np.maximum(np.linalg.norm(random,axis=0,keepdims=True),1e-12)
        for a,m in models.items():
            for mode in ['oracle','zero','random']:predictions[a+'_'+mode]=intervened(m,s,mode)
        before=convert(s['old']['point_plane_ids'],s['mapping']);row={}
        for arm,p in predictions.items():
            np.testing.assert_array_equal(p[s['old']['point_plane_ids']<0],s['old']['point_plane_ids'][s['old']['point_plane_ids']<0])
            assert np.isin(p[p>=0],s['old']['plane_ids']).all()
            row[arm]=metrics(before,convert(p,s['mapping']),s['truth'],s['distance'])
        np.savez_compressed(out/f'{name}_GT_DIAGNOSTIC.npz',schema_version=1,source_indices=np.arange(len(before)),original_ids=s['old']['point_plane_ids'],**predictions)
        report[name]=row;write_json(out/'interventions.json',report)
        for a in models:draw_comparison(s,{mode:predictions[a+'_'+mode] for mode in ['learned','oracle','zero','random']},out/'figures'/f'{name}_{a}.png','GT DIAGNOSTIC ONLY',
            'oracle uses GT directions for ALL candidates; frozen weights; NOT deployable results or an upper bound.')
        print('INTERVENTION',name,row,flush=True)
    aggregate={a:{key:sum(row[a][key] for row in report.values()) for key in next(iter(report.values()))[a]} for a in next(iter(report.values()))}
    write_json(out/'completion.json',dict(status='complete',scope='diagnostic only',aggregate=aggregate,overfit=overfit))


if __name__=='__main__':main()
