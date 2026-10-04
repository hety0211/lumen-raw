"""Offline FFDNet color denoising, overlapping tiles and optional ONNX CUDA."""
from pathlib import Path
import numpy as np
from .i18n import tr


def session(cuda=True):
    from . import compute
    path=Path(__file__).resolve().parents[1]/'assets/models/ffdnet-color.onnx'
    return compute.session(path,cuda)


def noise_level(rgb):
    """Robust Haar high-frequency noise estimate from full-resolution patches."""
    h,w=rgb.shape[:2];estimates=[]
    for cy in (.2,.5,.8):
        for cx in (.2,.5,.8):
            y=max(0,min(h-256,round(h*cy)-128));x=max(0,min(w-256,round(w*cx)-128))
            patch=rgb[y:y+256,x:x+256]
            ph,pw=patch.shape[:2];patch=patch[:ph//2*2,:pw//2*2]
            if patch.size:
                hh=(patch[::2,::2]+patch[1::2,1::2]-patch[::2,1::2]-patch[1::2,::2])*.5
                estimates.append(float(np.median(np.abs(hh))/.67448975))
    return float(np.clip(np.percentile(estimates,30) if estimates else 1/255,.5/255,60/255))


def process(rgb,amount=35,cuda=True,progress=None,cancel=None,noise_reference=None):
    if not 0<=amount<=100:raise ValueError(tr('去杂色强度需为 0–100。'))
    if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消去杂色'))
    if not amount:return rgb.copy(),tr('FFDNet · 强度 0')
    from .large_image import allocate,validate_size
    validate_size(rgb.shape)
    sess=session(cuda);h,w=rgb.shape[:2];output=allocate(rgb.shape)
    # FFDNet receptive radius is 24 source pixels. Even tile origins preserve
    # pixel-unshuffle alignment; 32px overlap keeps seams outside copied centers.
    tile,pad=256,32;total=((h+tile-1)//tile)*((w+tile-1)//tile);done=0
    reference=noise_level(rgb) if noise_reference is None else float(noise_reference)
    if not np.isfinite(reference) or not 0<=reference<=1:raise ValueError(tr('无效噪声参考。'))
    strength=min(75/255,reference*amount/35)
    sigma=np.full((1,1,1,1),strength,np.float32)
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消去杂色'))
            ey,ex=min(h,y+tile),min(w,x+tile)
            sy,sx=max(0,y-pad),max(0,x-pad);ty,tx=min(h,ey+pad),min(w,ex+pad)
            patch=rgb[sy:ty,sx:tx]
            patch=np.pad(patch,((0,patch.shape[0]%2),(0,patch.shape[1]%2),(0,0)),mode='edge')
            result=sess.run(None,{'image':patch.transpose(2,0,1)[None].copy(),'sigma':sigma})[0]
            if result.shape!=(1,3,*patch.shape[:2]) or not np.isfinite(result).all():raise ValueError(tr('去杂色模型输出无效。'))
            result=result[0].transpose(1,2,0)
            output[y:ey,x:ex]=result[y-sy:ey-sy,x-sx:ex-sx]
            done+=1
            if progress:progress(done,total)
    return np.clip(output,0,1,out=output),tr('FFDNet · 噪声参考 {v:.1f}/255 · ', v=strength*255)+sess.get_providers()[0]
