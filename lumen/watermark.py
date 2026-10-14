"""Photo borders: preserve the floating-point image and render text only in margins."""
import base64
import io
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .i18n import tr

PRESETS = {'gallery':(tr('画廊白'),(250,249,246),(39,43,39)),
           'noir':(tr('暗夜黑'),(20,23,23),(231,234,225)),
           'travel':(tr('旅行纸'),(238,231,214),(74,71,54)),
           'sage':(tr('山野绿'),(220,228,214),(41,62,43))}
LABELS = {'date':tr('拍摄时间'),'body':tr('机身'),'lens':tr('镜头'),'aperture':tr('光圈'),'shutter':tr('快门'),'iso':'ISO','focal':tr('焦距'),'brand':tr('品牌文字')}


#: Built-in camera and lens logos (assets/logos, see its README.txt): ``camera_logo`` / ``lens_logo``
#: ``'auto'`` picks them from the photo's EXIF, ``''`` shows none; a base64 PNG is a logo imported
#: by 1.6.1 or earlier, still drawn as it was.
AUTO = 'auto'
LOGOS = Path(__file__).resolve().parents[1] / 'assets' / 'logos'


def defaults():
    return dict(enabled=False,preset='gallery',sides=['bottom'],size=14.,
                fields=list(LABELS),camera_logo=AUTO,lens_logo=AUTO)


@lru_cache(maxsize=1)
def builtin():
    """{key: entry} of the built-in logos: brand name, role (camera / lens / both), colour handling,
    height share of the border and the EXIF patterns that select it."""
    try:
        return json.loads((LOGOS / 'logos.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def match(text, role):
    """Key of the first built-in logo for ``role`` ('camera' or 'lens') whose pattern occurs in ``text``."""
    text = f' {str(text or "").upper()} '
    for key, entry in builtin().items():
        if entry['role'] in (role, 'both') and any(pattern in text for pattern in entry['patterns']):
            return key
    return None


def brand_logos(photo):
    """(camera, lens) built-in logo keys for a photo's EXIF, None where there is none; a maker's own
    lens shows no second copy of the camera's logo (Nikon lenses show NIKKOR)."""
    camera = match(f"{photo.get('body', '')} {photo.get('brand', '')}", 'camera')
    lens = match(f"{photo.get('lens_brand', '')} {photo.get('lens', '')}", 'lens')
    return camera, (lens if lens != camera else None)


def _luminance(rgb):
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return float(linear @ np.array([.2126, .7152, .0722]))


@lru_cache(maxsize=64)
def builtin_logo(key, ink, paper):
    """RGBA image of a built-in logo for a border of colour ``paper`` with text colour ``ink``.

    'ink' logos: the neutral dark parts take ``ink``, so black wordmarks stay readable on dark
    borders, while colour parts (Fujifilm's red mark) keep their colour.  'colour' wordmarks keep
    their colours unless they would be hard to read on ``paper`` (contrast below 2.2:1, e.g. dark
    blue on black); then they are drawn in ``ink``.  'badge' logos keep their colours always."""
    with Image.open(LOGOS / f'{key}.png') as im:
        rgba = np.asarray(im.convert('RGBA')).copy()
    kind = builtin()[key]['colour']
    rgb = rgba[..., :3].astype(np.int16)
    if kind == 'ink':
        neutral_dark = (rgb.max(axis=2) - rgb.min(axis=2) < 48) & (rgb.mean(axis=2) < 140)
        rgba[neutral_dark, :3] = ink
    elif kind == 'colour':
        solid = rgba[..., 3] > 128
        if solid.any():
            # Most of the logo's area decides (Olympus: blue letters over a thin yellow line).
            linear = np.where(rgb[solid] / 255 <= .04045, rgb[solid] / 255 / 12.92, ((rgb[solid] / 255 + .055) / 1.055) ** 2.4)
            low, high = sorted((float(np.median(linear @ np.array([.2126, .7152, .0722]))),
                                _luminance(np.asarray(paper) / 255)))
            if (high + .05) / (low + .05) < 2.2:
                rgba[..., :3] = ink
    return Image.fromarray(rgba, 'RGBA')


@lru_cache(maxsize=16)
def logo(data):
    raw = base64.b64decode(data,validate=True)
    with Image.open(io.BytesIO(raw)) as im:
        if im.format!='PNG' or max(im.size)>2048: raise ValueError(tr('标志需要不超过 2048px 的 PNG。'))
        return im.convert('RGBA')


def validate(data):
    if not isinstance(data,dict): raise ValueError(tr('水印设置无效。'))
    result=defaults()
    result['enabled']=bool(data.get('enabled',False))
    result['preset']=data.get('preset','gallery')
    if result['preset'] not in PRESETS: raise ValueError(tr('未知水印预设。'))
    for key,allowed in [('sides',('top','bottom','left','right')),('fields',LABELS)]:
        value=data.get(key,result[key])
        if not isinstance(value,list) or any(v not in allowed for v in value): raise ValueError(tr('无效水印项目。'))
        result[key]=list(dict.fromkeys(value))
    value=data.get('size',14.)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not 8<=value<=35: raise ValueError(tr('边框宽度需为 8–35%。'))
    result['size']=float(value)
    for key in ('camera_logo','lens_logo'):
        value=data.get(key,'')
        if not isinstance(value,str) or len(value)>3_000_000: raise ValueError(tr('标志文件过大。'))
        if value and value!=AUTO: logo(value)
        result[key]=value
    return result


def lines(settings,photo,hidden=()):
    """Text rows of a border; brand names in ``hidden`` (upper case) are drawn as logos instead."""
    fields=settings['fields']
    def val(k): return photo.get(k) or tr('{v}未记录', v=LABELS[k])
    output=[]
    if 'brand' in fields:
        brands=list(dict.fromkeys(p.upper() for p in (photo.get('brand'),photo.get('lens_brand')) if p and p.upper() not in hidden))
        if brands: output.append(' / '.join(brands))
    for key in ('body','lens'):
        if key in fields: output.append(val(key))
    tech='  ·  '.join(val(k) for k in ('focal','aperture','shutter','iso') if k in fields)
    if tech: output.append(tech)
    if 'date' in fields: output.append(val('date'))
    return output


def band(length,thickness,settings,photo):
    _,bg,fg=PRESETS[settings['preset']]
    strip=Image.new('RGB',(length,thickness),bg)
    draw=ImageDraw.Draw(strip)
    margin=max(3,round(thickness*.15))
    reserved=[margin,margin]
    automatic=brand_logos(photo);hidden=set()
    brand,lens_brand=(str(photo.get(k) or '').upper() for k in ('brand','lens_brand'))
    for side,key in enumerate(('camera_logo','lens_logo')):
        if not settings[key]:continue
        if settings[key]==AUTO:
            found=automatic[side]
            if found is None:continue
            emblem=builtin_logo(found,tuple(fg),tuple(bg)).copy()
            height=max(1,round((thickness-2*margin)*builtin()[found]['height']))
            # The logo stands for the maker's name: the camera's make and a native lens's make;
            # a lens logo for the lens's make.
            if side==0:
                hidden.add(brand)
                if lens_brand and found in (match(lens_brand,'camera'),match(lens_brand,'lens')):hidden.add(lens_brand)
            else:hidden.add(lens_brand)
        else:
            emblem=logo(settings[key]).copy();height=max(1,thickness-2*margin)
        emblem.thumbnail((max(1,length//5),height),Image.Resampling.LANCZOS)
        x=margin if side==0 else length-margin-emblem.width
        strip.paste(emblem,(x,(thickness-emblem.height)//2),emblem)
        reserved[side]+=emblem.width+margin
    rows=lines(settings,photo,hidden)
    available=max(1,length-sum(reserved))
    fontpath=str(Path(__file__).resolve().parents[1]/'assets/NotoSansSC.ttf')
    fontsize=max(2,int((thickness-2*margin)/max(1,len(rows))*.76))
    while True:
        font=ImageFont.truetype(fontpath,fontsize)
        try:font.set_variation_by_axes([420])
        except (OSError,AttributeError):pass
        if fontsize<=2 or all(draw.textlength(t,font=font)<=available for t in rows): break
        fontsize-=1
    lineheight=max(fontsize+1,(thickness-2*margin)/max(1,len(rows)))
    for i,text in enumerate(rows):
        x=reserved[0]+available/2
        y=thickness/2+(i-(len(rows)-1)/2)*lineheight
        draw.text((x,y),text,font=font,fill=fg,anchor='mm')
    return np.asarray(strip,dtype=np.float32)/255


def dimensions(shape,settings):
    h,w=shape[:2]
    pad=max(1,round(min(w,h)*settings['size']/100))
    sides=settings['sides'] if settings['enabled'] else []
    return (w+pad*sum(s in sides for s in ('left','right')),
            h+pad*sum(s in sides for s in ('top','bottom')))


def apply(rgb,settings,photo,space=None):
    """Borders around ``rgb``; with an output colour space other than sRGB the border colours
    (and logos) are converted into it, so they look as in an sRGB export."""
    from . import color
    settings=validate(settings)
    if not settings['enabled'] or not settings['sides']: return rgb
    h,w=rgb.shape[:2];width,height=dimensions(rgb.shape,settings)
    from . import large_image
    if width*height>large_image.MAX_PIXELS: raise ValueError(tr('加水印后超过 4 亿像素，请降低输出尺寸或边框宽度。'))
    pad=max(1,round(min(w,h)*settings['size']/100))
    sides=settings['sides'];left=pad if 'left' in sides else 0;top=pad if 'top' in sides else 0
    out=large_image.allocate((height,width,3))
    out[:]=color.reencode(np.array([[PRESETS[settings['preset']][1]]],np.float32)/255,space)[0,0]
    out[top:top+h,left:left+w]=rgb
    for side in sides:
        if side in ('top','bottom'):
            y=0 if side=='top' else top+h
            out[y:y+pad,left:left+w]=color.reencode(band(w,pad,settings,photo),space)
        else:
            x=0 if side=='left' else left+w
            out[top:top+h,x:x+pad]=np.rot90(color.reencode(band(h,pad,settings,photo),space),1 if side=='left' else -1)
    return out
