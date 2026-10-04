"""Bounded-memory image buffers and strip encoding for up to 400 megapixels."""
import os,shutil,sys,tempfile,weakref
from pathlib import Path
import numpy as np
from .i18n import tr

MAX_PIXELS=400_000_000
MAP_BYTES=256*1024*1024
STRIP_ROWS=64

def validate_size(shape):
    h,w=map(int,shape[:2])
    if h<1 or w<1 or h*w>MAX_PIXELS:
        raise ValueError(tr('图像超过 4 亿像素，请裁切或降低输出倍率。'))

def _cleanup(mapping,path):
    try:mapping.close()
    finally:
        try:Path(path).unlink(missing_ok=True)
        except OSError:pass

def allocate(shape,dtype=np.float32,zeros=False):
    validate_size(shape)
    size=int(np.prod(shape))*np.dtype(dtype).itemsize
    if size<=MAP_BYTES:return np.zeros(shape,dtype) if zeros else np.empty(shape,dtype)
    folder=Path(tempfile.gettempdir())/'LumenRAW-cache';folder.mkdir(exist_ok=True)
    if shutil.disk_usage(folder).free<size+256*1024*1024:
        raise ValueError(tr('大图缓存需要至少 {v:.1f} GB 可用磁盘空间。', v=size/2**30+.25))
    fd,path=tempfile.mkstemp(prefix='image-',suffix='.tmp',dir=folder);os.close(fd)
    try:data=np.memmap(path,mode='w+',dtype=dtype,shape=shape)
    except BaseException:
        Path(path).unlink(missing_ok=True);raise
    weakref.finalize(data,_cleanup,data._mmap,path)
    # New mappings are zero-filled by the OS without touching the entire buffer.
    return data

def strips(image):
    for y in range(0,image.shape[0],STRIP_ROWS):yield y,image[y:y+STRIP_ROWS]

def finite(image):
    return all(np.isfinite(block).all() for _,block in strips(image))

def encode_strips(image,linear=False,bits=16):
    from .engine import to_linear
    maximum=(1<<bits)-1
    dtype=np.uint16 if bits==16 else np.uint8
    for _,block in strips(image):
        if not np.isfinite(block).all():raise ValueError(tr('图像包含无效像素。'))
        if linear:block=to_linear(block)
        yield np.round(np.clip(block,0,1)*maximum).astype(dtype).tobytes()

def transform(image,edits):
    """Copy crop/rotation in strips, keeping output buffers backed by disk."""
    from . import engine
    if abs(edits.get('straighten',0))>.001:
        # OpenCV affine output can use a mapped destination directly.
        import cv2
        crop=dict(edits,straighten=0.,rotation=0)
        image=transform(image,crop)
        h,w=image.shape[:2];theta=np.deg2rad(abs(edits['straighten']))
        scale=max(np.cos(theta)+h/w*np.sin(theta),np.cos(theta)+w/h*np.sin(theta))
        matrix=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),edits['straighten'],scale)
        result=allocate(image.shape)
        cv2.warpAffine(image,matrix,(w,h),dst=result,flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        image=result
    elif edits.get('crop'):
        h,w=image.shape[:2];a,b,c,d=edits['crop']
        x0,y0=min(w-1,int(a*w)),min(h-1,int(b*h))
        x1,y1=max(x0+1,min(w,round(c*w))),max(y0+1,min(h,round(d*h)))
        image=image[y0:y1,x0:x1]
    view=np.rot90(image,-edits['rotation'])
    if view.flags.c_contiguous:return view
    output=allocate(view.shape)
    for y,block in strips(view):output[y:y+len(block)]=block
    return output

def _macos_available_memory():
    """Free plus inactive pages (as Activity Monitor and psutil count them) via host_statistics64."""
    import ctypes,ctypes.util
    fields=[(k,ctypes.c_uint32) for k in ('free','active','inactive','wire')]
    fields+=[(k,ctypes.c_uint64) for k in ('zero_fill','reactivations','pageins','pageouts','faults','cow_faults','lookups','hits','purges')]
    fields+=[(k,ctypes.c_uint32) for k in ('purgeable','speculative')]
    fields+=[(k,ctypes.c_uint64) for k in ('decompressions','compressions','swapins','swapouts')]
    fields+=[(k,ctypes.c_uint32) for k in ('compressor','throttled','external','internal')]
    fields+=[('uncompressed',ctypes.c_uint64)]
    class Statistics(ctypes.Structure):
        _fields_=fields
    try:
        libc=ctypes.CDLL(ctypes.util.find_library('c') or '/usr/lib/libSystem.B.dylib')
        libc.mach_host_self.restype=ctypes.c_uint32
        libc.host_statistics64.argtypes=(ctypes.c_uint32,ctypes.c_int,ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32))
        statistics=Statistics();count=ctypes.c_uint32(ctypes.sizeof(Statistics)//4)
        if libc.host_statistics64(libc.mach_host_self(),4,ctypes.byref(statistics),ctypes.byref(count)):return None  # HOST_VM_INFO64
        return (statistics.free+statistics.inactive)*os.sysconf('SC_PAGE_SIZE')
    except (OSError,AttributeError,ValueError):
        return None

def available_memory():
    if sys.platform=='darwin':return _macos_available_memory()
    if os.name!='nt':return None
    import ctypes
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','avail','page','freepage','virt','freevirt','ext')]
    status=Status();status.length=ctypes.sizeof(Status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):return status.avail

def fits_in_memory(shape):
    """Whole-image processing needs roughly 100 bytes per pixel of free memory."""
    available=available_memory()
    return available is None or shape[0]*shape[1]*100<=available*.85

def process(source,edits,backend,apply_crop,detail_scale,cache=None):
    """Stream point operations. Spatial edits retain the original whole-image math."""
    from . import engine,lens
    if lens.active(edits.get('lens')):
        # 1.5.1: correct the lens once (disk-backed for large frames), then process without it.
        source=lens.apply(source,edits['lens'])
        edits=dict(edits,lens=dict(edits['lens'],enabled=False))
    a=edits['adjustments']
    spatial=any(a[k] for k in ('dehaze','clarity','texture','sharpness','denoise','color_noise'))
    spatial|=any(m['enabled'] and any(m['adjustments'].values()) for m in edits['masks'])
    spatial|=any(o['enabled'] and o['opacity'] for o in edits.get('retouch',[]))
    spatial|=bool(edits.get('effects',{}).get('vignette') or edits.get('effects',{}).get('grain'))
    if spatial:
        available=available_memory();required=source.shape[0]*source.shape[1]*100
        if available is not None and required>available*.85:
            raise ValueError(tr('当前空间类编辑预计需要约 {v:.1f} GB 可用内存。可先裁切、关闭质感／蒙版／修复，或在内存更大的电脑处理。4 亿像素的基础调色、AI 副本和 DNG 导出使用分块缓存。', v=required/2**30))
        return engine.process(source,edits,backend,apply_crop,detail_scale,_stream=False,cache=cache)
    output=allocate(source.shape)
    for y,block in strips(source):
        output[y:y+len(block)]=engine.process(block,edits,backend,False,detail_scale,_stream=False)
    return transform(output,edits) if apply_crop else output
