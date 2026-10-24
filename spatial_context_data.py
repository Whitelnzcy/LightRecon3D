"""Read larger input halos without changing center supervision or output pixels.

Coordinates remain integer-center xy in the explicitly registered pointmap.
No resizing/rotation: outside-image context is zero, including observed mask.
"""
import numpy as np
import torch
from spatial_support_data import patch_batch
from spatial_line_features import line_batch


def extract_context(array, boxes, margin):
    if not isinstance(margin, int) or margin < 0:
        raise ValueError('Nonnegative integer context margin required')
    h,w=array.shape[-2:];result=[]
    for x0,y0,x1,y1 in boxes:
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError('Center box outside registered image')
        left,top,right,bottom=x0-margin,y0-margin,x1+margin,y1+margin
        src_x0,src_y0,src_x1,src_y1=max(0,left),max(0,top),min(w,right),min(h,bottom)
        patch=np.zeros((*array.shape[:-2],bottom-top,right-left),dtype=array.dtype)
        patch[...,src_y0-top:src_y1-top,src_x0-left:src_x1-left]=array[...,src_y0:src_y1,src_x0:src_x1]
        result.append(patch)
    if not result:raise ValueError('Empty patch list')
    return np.stack(result)


def center_crop(tensor, margin):
    if margin<0 or 2*margin>=min(tensor.shape[-2:]):raise ValueError('Invalid center crop')
    return tensor[...,margin:-margin,margin:-margin] if margin else tensor


def context_batch(view, boxes, device, margin=0, with_targets=True):
    if margin==0:
        batch=patch_batch(view,boxes,device,with_targets)
        return batch[:2]+[line_batch(view,boxes,device)]+batch[2:]
    result=[torch.from_numpy(extract_context(view[field],boxes,margin)).to(device)
            for field in ['common','candidates','lines']]
    if with_targets:
        for field in ['target','original_columns']:
            # Identical center targets, NOT supervision of the added halo.
            result.append(torch.from_numpy(extract_context(view[field],boxes,0)).to(device))
    return result
