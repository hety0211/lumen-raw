"""DNG 1.4 LinearRaw export of edited scene-linear sRGB (not sensor mosaics)."""
from pathlib import Path
import numpy as np
import tifffile
from .i18n import tr


MODEL = 'Lumen Rendered Linear sRGB'


def is_rendered(path):
    try:
        with tifffile.TiffFile(path) as f:
            tag = f.pages[0].tags.get(50708)
            return bool(tag and tag.value == MODEL)
    except (OSError, ValueError, tifffile.TiffFileError):
        return False


def dimensions(path):
    with tifffile.TiffFile(path) as f:
        return f.pages[0].imagewidth,f.pages[0].imagelength

def read(path,preview_limit=None):
    from . import large_image
    import cv2
    width,height=dimensions(path);large_image.validate_size((height,width))
    data=tifffile.memmap(path,mode='r')
    try:
        if data.ndim != 3 or data.shape[-1] != 3 or data.dtype != np.uint16:
            raise ValueError(tr('不支持的 Lumen DNG 像素格式。'))
        if preview_limit and max(height,width)>preview_limit:
            # Sample all source pixels through OpenCV's area reducer without a
            # full-frame float conversion or an extra full-resolution array.
            ratio=preview_limit/max(height,width)
            return cv2.resize(data,(max(1,round(width*ratio)),max(1,round(height*ratio))),interpolation=cv2.INTER_AREA).astype(np.float32)/65535
        out=large_image.allocate(data.shape)
        for y,block in large_image.strips(data):out[y:y+len(block)]=block.astype(np.float32)/65535
        return out
    finally:data._mmap.close()


def write(path, srgb, photo=None, provenance=None):
    from .engine import to_linear
    h,w = srgb.shape[:2]
    matrix = [3.2406,-1.5372,-.4986, -.9689,1.8758,.0415, .0557,-.2040,1.0570]
    rational = [v for n in matrix for v in (round(n*1000000),1000000)]
    tags = [(254,'I',1,0,False), (274,'H',1,1,False),
        (50706,'B',4,(1,4,0,0),False), (50707,'B',4,(1,3,0,0),False),
        (50708,'s',0,MODEL,False), (50714,'H',1,0,False), (50717,'I',1,65535,False),
        (50718,'2I',2,(1,1,1,1),False), (50719,'I',2,(0,0),False),
        (50720,'I',2,(w,h),False), (50721,'2i',9,rational,False),
        (50728,'2I',3,(1,1,1,1,1,1),False), (50730,'2i',1,(0,1),False),
        (50778,'H',1,21,False), (50829,'I',4,(0,0,h,w),False)]
    from . import large_image
    large_image.validate_size(srgb.shape)
    from .photo_metadata import description
    tifffile.imwrite(str(path),large_image.encode_strips(srgb,linear=True),shape=srgb.shape,dtype=np.uint16,
                    rowsperstrip=large_image.STRIP_ROWS,photometric=34892,planarconfig='contig',
                    metadata=None,software='LUMEN RAW 1.2.2',extratags=tags,
                    description=description(photo,provenance) if photo is not None else None)
