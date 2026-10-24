# Direction propagation reach ablation

Continue the fixed research line. Frozen checkpoint sensitivity on first16 GT-free
tiles/view shows ~2% feature change, <=0.07% boundary label flips on zero direction.
Only change grid step 1/4/8, four axial neighbors+center. Preserve normalized3D
radius.02, bias+2, losses, candidate pool, XYZ and frozen RGB prior. TRAIN4,
historicalDEV2,200steps,batch4,paired initialization/batches, seeds42 then43.
No GT in prediction. No promotion on one seed. Track damage and per-scene errors.
