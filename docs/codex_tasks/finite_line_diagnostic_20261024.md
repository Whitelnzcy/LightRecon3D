# Finite 3D line diagnostic

User: continue, the learned RGB boundary looks accurate. Verify correspondence
to 3D plane pairs before more training. Fixed XYZ/labels/equations and RGB prior.
Two historical DEV scenes, all nonparallel pairs,256 samples in overlap of
support along-line1%-99% quantiles. Fixed diagnostic support radius2% robust
scene diagonal, probability0.5, pixel proximity1.5px, XYZ observation proxy.
No GT, no ownership edits, no threshold sweep or mainline replacement.
Save continuous support distances/rejection reasons and global segment endpoints.
If no lines survive, diagnose rather than widen thresholds for appearance.
