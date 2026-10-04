"""Photo borders: preserve the floating-point image and render text only in margins."""
import base64
import io
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


def defaults():
    return dict(enabled=False,preset='gallery',sides=['bottom'],size=14.,
                fields=list(LABELS),camera_logo='',lens_logo='')


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
        if value: logo(value)
        result[key]=value
    return result


def lines(settings,photo):
    fields=settings['fields']
    def val(k): return photo.get(k) or tr('{v}未记录', v=LABELS[k])
    output=[]
    if 'brand' in fields:
        brands=list(dict.fromkeys(p.upper() for p in (photo.get('brand'),photo.get('lens_brand')) if p))
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
    rows=lines(settings,photo)
    margin=max(3,round(thickness*.15))
    reserved=[margin,margin]
    for side,key in enumerate(('camera_logo','lens_logo')):
        if settings[key]:
            emblem=logo(settings[key]).copy()
            emblem.thumbnail((max(1,length//5),max(1,thickness-2*margin)),Image.Resampling.LANCZOS)
            x=margin if side==0 else length-margin-emblem.width
            strip.paste(emblem,(x,(thickness-emblem.height)//2),emblem)
            reserved[side]+=emblem.width+margin
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


def apply(rgb,settings,photo):
    settings=validate(settings)
    if not settings['enabled'] or not settings['sides']: return rgb
    h,w=rgb.shape[:2];width,height=dimensions(rgb.shape,settings)
    from . import large_image
    if width*height>large_image.MAX_PIXELS: raise ValueError(tr('加水印后超过 4 亿像素，请降低输出尺寸或边框宽度。'))
    pad=max(1,round(min(w,h)*settings['size']/100))
    sides=settings['sides'];left=pad if 'left' in sides else 0;top=pad if 'top' in sides else 0
    out=large_image.allocate((height,width,3));out[:]=np.array(PRESETS[settings['preset']][1])/255
    out[top:top+h,left:left+w]=rgb
    for side in sides:
        if side in ('top','bottom'):
            y=0 if side=='top' else top+h
            out[y:y+pad,left:left+w]=band(w,pad,settings,photo)
        else:
            x=0 if side=='left' else left+w
            out[top:top+h,x:x+pad]=np.rot90(band(h,pad,settings,photo),1 if side=='left' else -1)
    return out
