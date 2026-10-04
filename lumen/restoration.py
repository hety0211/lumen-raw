"""Photo-oriented NAFNet SIDD and RRDB Real-ESRGAN, with feathered tile assembly."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from . import large_image,compute
from .i18n import tr


@lru_cache(maxsize=2)
def session(kind,cuda=True):
    filename={'super':'realesrgan-x4plus.onnx','denoise':'nafnet-sidd.onnx','drunet':'drunet-color.onnx'}[kind]
    path=Path(__file__).resolve().parents[1]/'assets/models'/filename
    return compute.session(path,cuda)


def axis_tiles(length,tile,overlap):
    if length<=tile:return [(0,length,np.ones(length,np.float32))]
    starts=list(range(0,length-tile+1,tile-overlap))
    if starts[-1]+tile<length:starts.append(starts[-1]+tile-overlap)
    result=[]
    for start in starts:
        end=min(length,start+tile);weight=np.ones(end-start,np.float32)
        # Positive raised-cosine windows, normalized after accumulation. Truncated
        # final tiles and irregular image dimensions never leave uncovered pixels.
        n=min(overlap,len(weight));ramp=(.5-.5*np.cos(np.pi*(np.arange(n)+1)/(n+1))).astype(np.float32)
        if start:weight[:n]*=ramp
        if end<length:weight[-n:]*=ramp[::-1]
        result.append((start,end,weight))
    return result


def tiled(rgb,kind,scale=1,cuda=True,progress=None,cancel=None,noise=None,region=None):
    h,w=rgb.shape[:2]
    rx0,ry0,rx1,ry1=region or (0,0,w,h)
    if not (0<=rx0<rx1<=w and 0<=ry0<ry1<=h):raise ValueError(tr('无效预览区域。'))
    large_image.validate_size(((ry1-ry0)*scale,(rx1-rx0)*scale))
    if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消 AI 运算'))
    sess=session(kind,cuda)
    tile,overlap,pad=(128,32,32) if kind=='super' else (256,64,32)
    ys=axis_tiles(h,tile,overlap);xs=axis_tiles(w,tile,overlap)
    wy=np.zeros(h,np.float32);wx=np.zeros(w,np.float32)
    for start,end,weight in ys:wy[start:end]+=weight
    for start,end,weight in xs:wx[start:end]+=weight
    out=large_image.allocate(((ry1-ry0)*scale,(rx1-rx0)*scale,3),zeros=True)
    ys=[item for item in ys if item[0]<ry1 and item[1]>ry0]
    xs=[item for item in xs if item[0]<rx1 and item[1]>rx0]
    native=4 if kind=='super' else 1;done=0;total=len(ys)*len(xs)
    for y,ey,yweight in ys:
        for x,ex,xweight in xs:
            if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消 AI 运算'))
            sy,sx=max(0,y-pad),max(0,x-pad);ty,tx=min(h,ey+pad),min(w,ex+pad)
            patch=rgb[sy:ty,sx:tx].transpose(2,0,1)[None].copy()
            inputs={'image':patch}
            if kind=='drunet':inputs['sigma']=np.full((1,1,1,1),noise,np.float32)
            result=sess.run(None,inputs)[0]
            expected=(1,3,(ty-sy)*native,(tx-sx)*native)
            if result.shape!=expected or not np.isfinite(result).all():raise ValueError(tr('AI 模型输出无效。'))
            result=result[0].transpose(1,2,0)
            if native!=scale:result=cv2.resize(result,((tx-sx)*scale,(ty-sy)*scale),interpolation=cv2.INTER_AREA)
            iy0,ix0=max(y,ry0),max(x,rx0);iy1,ix1=min(ey,ry1),min(ex,rx1)
            center=result[(iy0-sy)*scale:(iy1-sy)*scale,(ix0-sx)*scale:(ix1-sx)*scale]
            weight=np.repeat(yweight[iy0-y:iy1-y],scale)[:,None]*np.repeat(xweight[ix0-x:ix1-x],scale)[None,:]
            out[(iy0-ry0)*scale:(iy1-ry0)*scale,(ix0-rx0)*scale:(ix1-rx0)*scale]+=center*weight[...,None]
            done+=1
            if progress:progress(done,total)
    wx=np.repeat(wx[rx0:rx1],scale);wy=np.repeat(wy[ry0:ry1],scale)
    for y,block in large_image.strips(out):
        if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消 AI 运算'))
        block/=wy[y:y+len(block),None,None]*wx[None,:,None]
        np.clip(block,0,1,out=block)
    return out,sess.get_providers()[0]


def super_resolution(rgb,scale=2,cuda=True,progress=None,cancel=None,region=None):
    if scale not in (2,4):raise ValueError(tr('高画质超分支持 2× / 4×。'))
    out,provider=tiled(rgb,'super',scale,cuda,progress,cancel,region=region)
    return out,'Real-ESRGAN x4plus · RRDB · '+provider


def denoise(rgb,amount=70,cuda=True,progress=None,cancel=None,noise_reference=None,region=None):
    from .denoise import noise_level
    if not 0<=amount<=100:raise ValueError(tr('去杂色强度需为 0–100。'))
    reference=noise_level(rgb) if noise_reference is None else float(noise_reference)
    if not np.isfinite(reference) or not 0<=reference<=1:raise ValueError(tr('无效噪声参考。'))
    if not amount:
        if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消 AI 运算'))
        x0,y0,x1,y1=region or (0,0,rgb.shape[1],rgb.shape[0])
        return rgb[y0:y1,x0:x1].copy(),tr('NAFNet SIDD · 强度 0')
    out,provider=tiled(rgb,'denoise',1,cuda,progress,cancel,region=region)
    if region:
        x0,y0,x1,y1=region;rgb=rgb[y0:y1,x0:x1]
    blend=amount/100*float(np.clip(reference*255/2,.1,1))
    for y,block in large_image.strips(out):block[:]=block*blend+rgb[y:y+len(block)]*(1-blend)
    return out,tr('NAFNet SIDD · 混合 {v:.0f}% · ', v=blend*100)+provider


def drunet_denoise(rgb,amount=35,cuda=True,progress=None,cancel=None,noise_reference=None,region=None):
    from .denoise import noise_level
    if not 0<=amount<=100:raise ValueError(tr('去杂色强度需为 0–100。'))
    reference=noise_level(rgb) if noise_reference is None else float(noise_reference)
    if not np.isfinite(reference) or not 0<=reference<=1:raise ValueError(tr('无效噪声参考。'))
    if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消 AI 运算'))
    if not amount:
        x0,y0,x1,y1=region or (0,0,rgb.shape[1],rgb.shape[0])
        return rgb[y0:y1,x0:x1].copy(),tr('DRUNet · 强度 0')
    noise=min(50/255,reference*amount/35)
    out,provider=tiled(rgb,'drunet',1,cuda,progress,cancel,noise,region)
    return out,tr('DRUNet · 噪声参考 {v:.1f}/255 · ', v=noise*255)+provider
