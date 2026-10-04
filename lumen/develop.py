"""Per-photo, monotonic baseline rendering derived from the camera JPEG."""
import io
import numpy as np
import cv2
from PIL import Image, ImageOps
from .i18n import N_


IDENTITY = [[0., 0.], [1., 1.]]


def luminance(rgb):
    return rgb[..., 0]*.2126 + rgb[..., 1]*.7152 + rgb[..., 2]*.0722


def camera_curve(raw, linear, path=None):
    from .engine import to_linear, resize_limit
    from . import previews
    label = N_('相机预览亮度参考')  # kept in the recipe; translated where shown
    try:
        try:
            thumb = raw.extract_thumb()
        except Exception:
            # 1.4.1: Canon HDR PQ files embed HEVC previews that LibRaw cannot return.
            hevc = previews.hevc_preview(path) if path else None
            if hevc is None:
                raise
            thumb, label = None, N_('相机 HDR 预览亮度参考（PQ 转 SDR）')
            preview = hevc.astype(np.float32)/255
        if thumb is not None and isinstance(thumb.data, bytes):
            with Image.open(io.BytesIO(thumb.data)) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((800, 800))
                preview = np.asarray(im, np.float32)/255
        elif thumb is not None:
            preview = np.asarray(thumb.data, np.float32)/255
        a = luminance(resize_limit(linear, 800))
        b = luminance(to_linear(resize_limit(preview, 800)))
        quantiles = [0.5, 2, 5, 10, 20, 35, 50, 65, 80, 90, 95, 98, 99.5]
        xs, ys = np.percentile(a, quantiles), np.percentile(b, quantiles)
        points = [[0., 0.]]
        for x, y in zip(xs, ys):
            x = float(x)
            # Guard pathological thumbnails and preserve strictly ordered anchors.
            if x > points[-1][0] + .0001 and x < .999:
                y = float(np.clip(y, max(points[-1][1], x/8), min(1., x*16)))
                points.append([x,y])
        points.append([1.,1.])
        return points, label
    except Exception:
        # Fallback only when no embedded JPEG can be decoded; do not invent metadata.
        high = float(np.percentile(luminance(resize_limit(linear, 800)), 99.5))
        gain = float(np.clip(.8/max(high, .02), 1, 8))
        return [[0.,0.],[.8/gain,.8],[1.,1.]], N_('标准亮度（相机预览不可用）')


def apply(source, settings):
    if settings.get('mode') != 'camera' or settings.get('curve', IDENTITY) == IDENTITY:
        return source
    from .curves import evaluate
    lum = luminance(source)
    axis = np.linspace(0,1,16385)
    mapped = np.interp(lum,axis,evaluate(settings['curve'],axis)).astype(np.float32)
    ratio = np.clip(mapped/np.maximum(lum,1e-7),0,16)
    return source * ratio[...,None]
