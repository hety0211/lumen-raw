"""Bounded-memory image buffers and strip encoding for up to 400 megapixels."""
import os,shutil,sys,tempfile,weakref
import cv2
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

def encode_strips(image,linear=False,bits=16,space=None):
    """Integer strips of display values; ``linear`` decodes them first with the transfer curve
    of the output colour space ``space`` (sRGB by default)."""
    from .engine import to_linear
    from . import color
    maximum=(1<<bits)-1
    dtype=np.uint16 if bits==16 else np.uint8
    for _,block in strips(image):
        if not np.isfinite(block).all():raise ValueError(tr('图像包含无效像素。'))
        if linear:block=color.decode(block,space) if color.wide(space) else to_linear(block)
        yield np.round(np.clip(block,0,1)*maximum).astype(dtype).tobytes()

def transform(image,edits):
    """Copy crop/rotation in strips, keeping output buffers backed by disk."""
    from . import engine
    if abs(edits.get('straighten',0))>.001:
        # OpenCV affine output can use a mapped destination directly.
        import cv2
        crop=dict(edits,straighten=0.,rotation=0,flip=[False,False])
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
    view=engine.flipped(np.rot90(image,-edits['rotation']),edits)
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

def memory_status():
    """Windows: ``{physical_total, physical_available, commit_limit, commit_available}`` in bytes,
    else None.  Windows grants memory against the *commit limit* (physical memory plus the page
    file), not against free physical memory: other programs commit far more than they touch
    (here 32 GB committed with 17 GB in use), so with a small or disabled page file an allocation
    fails while Task Manager still shows gigabytes of free memory."""
    if os.name!='nt':return None
    import ctypes
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in ('total','avail','page','freepage','virt','freevirt','ext')]
    status=Status();status.length=ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):return None
    return dict(physical_total=status.total,physical_available=status.avail,commit_limit=status.page,
                commit_available=status.freepage)

def available_memory():
    """Bytes this process can still use without paging or failing, or None when unknown.
    2.3.0: on Windows the smaller of free physical memory and the remaining commit."""
    if sys.platform=='darwin':return _macos_available_memory()
    if os.name!='nt':
        # Linux: MemAvailable already accounts for reclaimable page cache and swap headroom.
        try:
            with open('/proc/meminfo',encoding='ascii') as stream:
                for line in stream:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1])*1024
        except (OSError,ValueError,IndexError):
            pass
        return None
    status=memory_status()
    return min(status['physical_available'],status['commit_available']) if status else None

#: Bytes per pixel a whole-frame render with spatial edits needs at most (2.3.0: measured 55–76 on a
#: 61 MP ARW with clarity, texture, sharpening, two masks and dehaze; 140 before 2.3.0).
FRAME_BYTES=100

def fits_in_memory(shape,per_pixel=FRAME_BYTES):
    """Whether a whole-frame render of ``shape`` fits the memory available now."""
    available=available_memory()
    return available is None or shape[0]*shape[1]*per_pixel<=available*.85

def block_size(margin,available=None):
    """Side of the square blocks a frame is rendered in when it does not fit memory: a block and
    its ``margin`` take at most a quarter of the available memory."""
    available=available or available_memory() or 4*2**30
    side=int((available*.25/FRAME_BYTES)**.5)-2*margin
    return max(512,min(8192,side))

def out_of_memory(error):
    """Whether an exception (or its text) reports that memory ran out: NumPy, OpenCV, ONNX
    Runtime / DirectML (E_OUTOFMEMORY) and Windows (the page file is too small)."""
    if isinstance(error,MemoryError):return True
    text=str(error).lower()
    return any(k in text for k in ('insufficient memory','unable to allocate','bad_alloc','bad allocation',
        'out of memory','outofmemory','8007000e','e_outofmemory','paging file is too small',
        'not enough memory','页面文件太小','内存资源不足','内存不足'))

def explain(error):
    """The text to show for an exception: plain words and the real numbers when memory ran out."""
    text=str(error) or type(error).__name__
    if not out_of_memory(error):return text
    detail=text.strip().splitlines()[-1][:160] if text.strip() else type(error).__name__
    lowered=text.lower()
    if any(k in lowered for k in ('dml','directml','d3d12','gpu','cuda')):
        return tr('显卡内存不足，无法完成这一步（{detail}）。可关闭其他占用显卡的程序，或在右侧面板底部改用“CPU 模式”后重试。',detail=detail)
    status=memory_status()
    if status:
        G=2**30
        used=(status['commit_limit']-status['commit_available'])/G
        return tr('内存不足，无法完成这一步（{detail}）。\nWindows 按“已提交内存”分配，上限是物理内存加页面文件：当前已提交 {used:.1f} GB，上限 {limit:.1f} GB，'
                  '可用物理内存 {free:.1f} GB。已提交接近上限时，即使任务管理器显示还有空闲内存，分配也会失败。\n'
                  '可在“系统 → 系统信息 → 高级系统设置 → 性能 → 设置 → 高级 → 虚拟内存”中把页面文件改为“系统管理的大小”，或关闭其他程序后重试。',
                  detail=detail,used=used,limit=status['commit_limit']/G,free=status['physical_available']/G)
    return tr('内存不足，无法完成这一步（{detail}）。请关闭其他程序，或先裁切后再处理。',detail=detail)

def process(source,edits,backend,apply_crop,detail_scale,cache=None,space=None):
    """Stream point operations. Spatial edits retain the original whole-image math when the frame
    fits memory; 2.3.0: otherwise they are rendered in blocks, as the 100 % view renders them."""
    from . import engine,lens,tone,portrait
    if lens.active(edits.get('lens')):
        # 1.5.1: correct the lens once (disk-backed for large frames), then process without it.
        source=lens.correct(source,edits)
        edits=dict(edits,lens=dict(edits['lens'],enabled=False))
    a=edits['adjustments']
    spatial=any(a[k] for k in ('dehaze','clarity','texture','sharpness','denoise','color_noise'))
    spatial|=any(m['enabled'] and any(m['adjustments'].values()) for m in edits['masks'])
    spatial|=any(o['enabled'] and o['opacity'] for o in edits.get('retouch',[]))
    spatial|=bool(portrait.active(edits.get('portrait')))
    spatial|=bool(edits.get('effects',{}).get('vignette') or edits.get('effects',{}).get('grain'))
    if spatial and fits_in_memory(source.shape):
        return engine.process(source,edits,backend,apply_crop,detail_scale,_stream=False,cache=cache,space=space)
    output=allocate(source.shape)
    h,w=source.shape[:2]
    context=None
    if engine.process_version(edits)==2 and tone.needs_context(a):
        # 1.6.0: local tone mapping reads the whole frame once, at its low resolution.
        ratio=min(1.,tone.CONTEXT_EDGE/max(h,w))
        small=cv2.resize(source,(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA) if ratio<1 else np.asarray(source)
        context=tone.context(small*engine._gain(edits),a)
    if spatial:
        # 2.3.0: the frame does not fit memory: blocks with the margins their neighbourhood tools
        # need (engine.process_region, exact for any block).  Dehaze statistics come from a
        # 3000-pixel copy (they are measured at 900 pixels anyway).
        import logging
        dehaze=None
        if a.get('dehaze',0)>0:
            ratio=min(1.,3000/max(h,w))
            reduced=cv2.resize(np.asarray(source),(max(1,round(w*ratio)),max(1,round(h*ratio))),interpolation=cv2.INTER_AREA)
            dehaze=engine.dehaze_context_for(reduced,edits,backend,detail_scale*ratio,context)
        masks=[m for m in edits['masks'] if m['enabled'] and any(m['adjustments'].values())]
        margin=engine.spatial_margin(a,(h,w),detail_scale)+max([engine.spatial_margin(m['adjustments'],(h,w),detail_scale) for m in masks],default=0)
        side=block_size(margin)
        logging.getLogger(__name__).info('rendering %d x %d in %d-pixel blocks (not enough memory for the whole frame)',w,h,side)
        for y in range(0,h,side):
            for x in range(0,w,side):
                rect=(x,y,min(w,x+side),min(h,y+side))
                output[rect[1]:rect[3],rect[0]:rect[2]]=engine.process_region(source,edits,backend,rect,detail_scale,dehaze,
                    tone_context=context,space=space)
    elif context is not None:
        for y in range(0,h,STRIP_ROWS):
            output[y:y+STRIP_ROWS]=engine.process_region(source,edits,backend,(0,y,w,min(h,y+STRIP_ROWS)),detail_scale,tone_context=context,space=space)
    else:
        for y,block in strips(source):
            output[y:y+len(block)]=engine.process(block,edits,backend,False,detail_scale,_stream=False,space=space)
    return transform(output,edits) if apply_crop else output
