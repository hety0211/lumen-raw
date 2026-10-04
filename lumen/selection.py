"""Offline open-source semantic / saliency masks and contiguous color selection."""
import base64
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from .i18n import tr


def encode(alpha):
    ok, data = cv2.imencode('.png', np.round(np.clip(alpha,0,1)*255).astype(np.uint8))
    if not ok:
        raise ValueError(tr('无法保存蒙版像素。'))
    return base64.b64encode(data).decode('ascii')


@lru_cache(maxsize=40)
def decode(encoded):
    if not isinstance(encoded,str) or len(encoded) > 8_000_000:
        raise ValueError(tr('蒙版数据过大。'))
    data = base64.b64decode(encoded,validate=True)
    # Validate dimensions before handing compressed bytes to the decoder.
    import struct
    if data[:8] != b'\x89PNG\r\n\x1a\n' or len(data) < 24:
        raise ValueError(tr('蒙版不是有效 PNG。'))
    w,h = struct.unpack('>II',data[16:24])
    if not 1 <= w <= 2048 or not 1 <= h <= 2048:
        raise ValueError(tr('蒙版分辨率超出限制。'))
    result = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
    if result is None:
        raise ValueError(tr('蒙版像素损坏。'))
    result = result.astype(np.float32)/255
    result.setflags(write=False)
    return result


@lru_cache(maxsize=4)
def session(kind, cuda=False):
    from . import compute
    filename = {'sky':'skyseg.onnx','person':'person-deeplab.onnx','background':'u2netp.onnx',
                'subject':'u2netp.onnx','foreground':'midas-small.onnx'}[kind]
    path = Path(__file__).resolve().parents[1]/'assets/models'/filename
    return compute.session(path,cuda)


def refine(alpha, rgb):
    # Guided filtering aligns coarse network boundaries to image luminance.
    h,w = rgb.shape[:2]
    p = cv2.resize(alpha,(w,h),interpolation=cv2.INTER_LINEAR)
    guide = cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY)
    box = lambda a: cv2.boxFilter(a,-1,(9,9),normalize=True,borderType=cv2.BORDER_REFLECT)
    mean_i,mean_p = box(guide),box(p)
    var = box(guide*guide)-mean_i*mean_i
    cov = box(guide*p)-mean_i*mean_p
    a = cov/(var+.002)
    b = mean_p-a*mean_i
    return np.clip(box(a)*guide+box(b),0,1)


def automatic(rgb, kind, cuda=False):
    sess = session('background' if kind=='subject' else kind,cuda)
    size = 256 if kind == 'foreground' else (384 if kind == 'person' else 320)
    x = cv2.resize(rgb,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)
    if kind not in ('person','foreground'):
        x /= max(float(x.max()),1e-5)
    x = (x - np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    prediction = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name:x.transpose(2,0,1)[None].copy()})[0].squeeze()
    if not np.isfinite(prediction).all():
        raise ValueError(tr('识别模型返回无效结果。'))
    if kind=='foreground':
        # MiDaS predicts relative inverse depth: larger values are nearer.
        low,high=np.percentile(prediction,[10,95])
        relative=np.clip((prediction-low)/max(float(high-low),1e-6),0,1)
        prediction=np.clip((relative-.48)/.22,0,1)
        prediction=prediction*prediction*(3-2*prediction)
    alpha = refine(np.clip(prediction,0,1),rgb)
    if kind == 'background':
        alpha = 1-alpha
    return alpha, sess.get_providers()[0]


def color_region(rgb, point, tolerance=18):
    """Fixed seed Lab distance plus 8-connected flood fill; no remote inference."""
    h,w = rgb.shape[:2]
    x,y = round(point[0]*(w-1)),round(point[1]*(h-1))
    lab = cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2LAB)
    distance = np.linalg.norm(lab-lab[y,x],axis=2)
    allowed = (distance <= tolerance).astype(np.uint8)
    # Flood-fill only the component containing the seed, not disconnected colors.
    flooded = allowed.copy()
    cv2.floodFill(flooded,None,(x,y),2,loDiff=0,upDiff=0,flags=8|cv2.FLOODFILL_FIXED_RANGE)
    return (flooded==2).astype(np.float32)


def raster_alpha(mask, shape, area=None):
    """Stored mask resized to a frame of ``shape``; with ``area`` only that block of it (1.4.0)."""
    h,w = shape[:2]
    feather = mask.get('feather',0)/100*min(h,w)*.008
    if area is None or area.shape == (h,w):
        alpha = cv2.resize(decode(mask['raster']),(w,h),interpolation=cv2.INTER_LINEAR)
        if feather > .2:
            alpha = cv2.GaussianBlur(alpha,(0,0),feather)
        return alpha
    from .engine import resize_region
    # OpenCV's float Gaussian kernel reaches about 4 sigma; read that much context.
    outer = area.grow(int(np.ceil(feather*4))+2 if feather > .2 else 0)
    alpha = resize_region(decode(mask['raster']),(w,h),outer)
    if feather > .2:
        alpha = cv2.GaussianBlur(alpha,(0,0),feather)
    return np.ascontiguousarray(alpha[outer.inner(area)])
