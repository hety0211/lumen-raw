"""Exposure curve: drag anywhere on the tone curve, sliders stay linked (1.4.1).

The curve shows exposure, contrast and the black / shadow / highlight / white
sliders (``evaluate``) followed by a fine tone curve (``tone``, stored in the
edits as ``tone_curve``).  A drag bends the curve locally around the pointer and
the point under the pointer follows it exactly, like a Lightroom point curve.

The four sliders have broad, overlapping effects, so they cannot bend the curve
locally on their own: before 1.4.1 a small drag drove them to ±100 and moved
other tones more than the dragged one.  Now only the one or two sliders of the
dragged tone range move, in the drag direction, as far as their least-squares
fit to the new curve goes; the fine tone curve carries the remainder, so the
image follows the drawn curve and the sliders still read like the curve.
"""
import copy
import numpy as np
from PySide6.QtCore import Qt,Signal,QPointF,QRectF
from PySide6.QtGui import QPainter,QPainterPath,QPen,QColor
from PySide6.QtWidgets import QWidget
from .curves import IDENTITY,tone
from .i18n import tr

KEYS=('blacks','shadows','highlights','whites')
LABELS=(tr('黑色'),tr('暗部'),tr('亮部'),tr('白色'))
#: Input tones the four sliders mainly describe; they decide which sliders a drag moves.
ZONES=np.array([.08,.33,.67,.92])
#: Gaussian falloff of a drag along the input tone axis.
WIDTH=.11
#: Pull of the previous slider values in the fit (larger keeps the sliders calmer).
PRIOR=2e-5
SAMPLES=np.linspace(0,1,513)
KNOTS=97

def linear(x):
    x=np.asarray(x,dtype=float)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def srgb(x):
    return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,0)**(1/2.4)-.055)

def basis(x,a):
    p=np.clip(linear(x)*2**a['exposure'],0,1)**.45
    return np.stack([(1-p)**6/85,(1-p)**2/65,p**3/65,p**6/85],axis=-1)

def evaluate(a,x):
    """Exposure, contrast and the four sliders on a neutral input tone (as the tonal stage)."""
    value=linear(x)*2**a['exposure']*2**(basis(x,a)@np.array([a[k] for k in KEYS]))
    return np.clip((srgb(value)-.5)*(1+a['contrast']/125)+.5,0,1)

def display(a,points,x):
    """The drawn curve: what a neutral input tone becomes after both stages."""
    return tone(points,evaluate(a,x))

def membership(x):
    g=np.exp(-.5*((x-ZONES)/.14)**2)
    return g/g.sum()

def fit(a,target,x,delta):
    """Slider values that best follow ``target`` (drawn curve over SAMPLES).

    Only sliders whose tone range covers ``x`` are free, and they only move with the drag.
    The fit is linear in photographic stops: the sliders scale linear light by 2^(basis·values).
    """
    lin=linear(SAMPLES)*2**a['exposure']
    gain=1+a['contrast']/125
    flat=(target-.5)/gain+.5
    valid=(SAMPLES>.01)&(target>.003)&(target<.997)&(flat>1e-4)&(flat<1)&(lin>1e-6)
    wanted=linear(np.clip(flat,1e-4,1))
    stops=np.log2(wanted)-np.log2(np.maximum(lin,1e-12))
    B=basis(SAMPLES,a)
    # Weight stops by how far they move the displayed value.
    slope=np.where(wanted<=.0031308,12.92,1.055/2.4*np.maximum(wanted,1e-9)**(1/2.4-1))
    weight=(np.log(2)*wanted*slope*gain)**2*valid
    previous=np.array([a[k] for k in KEYS],float)
    values=previous.copy()
    free=membership(x)>=.2
    for _ in range(4):
        if not free.any():
            break
        matrix=(B[:,free].T*weight)@B[:,free]+PRIOR*np.eye(int(free.sum()))
        rhs=(B[:,free].T*weight)@(stops-B[:,~free]@values[~free])+PRIOR*previous[free]
        values[free]=np.linalg.solve(matrix,rhs)
        against=free&(np.sign(values-previous)*np.sign(delta)<0)
        values[against]=previous[against]
        values=np.clip(values,-100,100)
        if not against.any():
            break
        free&=~against
    return {k:round(float(v),2) for k,v in zip(KEYS,values)}

def residual(a,target,x=None):
    """Fine tone curve that maps the sliders' curve onto ``target``, knots where the curve lies:
    evenly along the input tones, at the drag point and where the target starts or stops clipping."""
    u=np.maximum.accumulate(evaluate(a,SAMPLES))
    index=list(np.round(np.linspace(0,len(SAMPLES)-1,KNOTS)).astype(int))
    for hit in (np.flatnonzero(target>=.999)[:1],np.flatnonzero(target<=.001)[-1:]):
        index+=[int(i) for i in hit]+[int(i)-1 for i in hit if i>0]+[int(i)+1 for i in hit if i<len(SAMPLES)-1]
    if x is not None:
        index.append(int(np.argmin(np.abs(SAMPLES-x))))
    index=np.unique(index)
    u,v=u[index],np.maximum.accumulate(target[index])
    keep=np.concatenate([[True],np.diff(u)>2e-5])
    u,v=u[keep],v[keep]
    # Tones beyond the sliders' range (e.g. whites pulled down) continue with slope 1.
    if u[0]>1e-5:
        u,v=np.concatenate([[0.],u]),np.concatenate([[max(0.,v[0]-u[0])],v])
    else:
        u[0]=0.
    if u[-1]<1-1e-5:
        u,v=np.concatenate([u,[1.]]),np.concatenate([v,[min(1.,v[-1]+1-u[-1])]])
    else:
        u[-1]=1.
    if np.abs(v-u).max()<5e-4:
        return copy.deepcopy(IDENTITY)
    return [[round(float(p),6),round(float(q),6)] for p,q in zip(u,np.clip(v,0,1))]

def bend(a,points,x,delta):
    """Drag the curve at input tone ``x`` by ``delta``: new ``(adjustments, tone_curve)``."""
    current=display(a,points,SAMPLES)
    bump=np.exp(-.5*((SAMPLES-x)/WIDTH)**2)
    target=np.clip(current+delta*bump,0,1)
    # A tone curve never falls: a lifted point carries brighter tones up with it, a lowered
    # point carries darker tones down, so the point under the pointer always follows.
    target=np.maximum.accumulate(target) if delta>=0 else np.minimum.accumulate(target[::-1])[::-1]
    result=dict(copy.deepcopy(a),**fit(a,target,x,delta))
    # The sliders' curve must never fall (strong negative whites can fold it over) and must rise
    # wherever the target rises, or no tone curve could follow it; ease the slider move back until so.
    rising=np.diff(target)>1e-6
    for _ in range(14):
        step=np.diff(evaluate(result,SAMPLES))
        if np.all(step>=-1e-9) and np.all(step[rising]>1e-7):
            break
        for k in KEYS:
            result[k]=round(a[k]+(result[k]-a[k])*.7,2)
    return result,residual(result,target,x)

def expose(a,points,x,target):
    """Shift-drag: global exposure so that the curve passes through ``target`` at ``x``."""
    result=copy.deepcopy(a)
    def error(value):
        result['exposure']=value
        return abs(float(display(result,points,x))-target)
    # Sample the full legal range, then refine the closest bracket; extreme slider settings
    # need not respond monotonically.
    grid=np.linspace(-5,5,201)
    i=int(np.argmin([error(v) for v in grid]))
    lo,hi=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
    for _ in range(18):
        first,second=lo+(hi-lo)/3,hi-(hi-lo)/3
        if error(first)<=error(second):hi=second
        else:lo=first
    result['exposure']=float(np.clip(round((lo+hi)/2,2),-5,5))
    return result

def drag(a,x,target,global_exposure=False,points=None):
    """Move the curve point at input tone ``x`` to ``target``: ``(adjustments, tone_curve)``."""
    points=copy.deepcopy(points) if points else copy.deepcopy(IDENTITY)
    x=float(np.clip(x,1/255,1));target=float(np.clip(target,0,1))
    if global_exposure:
        return expose(a,points,x,target),points
    return bend(a,points,x,target-float(display(a,points,x)))


class ExposureCurve(QWidget):
    changed=Signal(dict,list)
    committed=Signal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.values=dict(exposure=0.,contrast=0.,**{k:0. for k in KEYS})
        self.points=copy.deepcopy(IDENTITY)
        self.drag_state=None;self.hover=None
        self.setFixedHeight(165)
        self.setMouseTracking(True)
        self.setToolTip(tr('在曲线任意位置上下拖动：该亮度附近平滑弯曲，黑色／暗部／亮部／白色随之联动。\n按住 Shift 拖动调整整体曝光。双击还原曲线。'))

    def area(self):return QRectF(18,12,self.width()-36,self.height()-42)

    def set_values(self,values,points=None):
        self.values=copy.deepcopy(values)
        self.points=copy.deepcopy(points) if points else copy.deepcopy(IDENTITY)
        self.update()

    def screen(self,x,y):
        a=self.area();return QPointF(a.left()+x*a.width(),a.bottom()-y*a.height())

    def tone_at(self,position):
        return float(np.clip((position.x()-self.area().left())/self.area().width(),1/255,1))

    def mousePressEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton or not self.area().adjusted(-6,-6,6,6).contains(event.position()):return
        x=self.tone_at(event.position())
        self.drag_state=(copy.deepcopy(self.values),copy.deepcopy(self.points),x,event.position().y(),
                         float(display(self.values,self.points,x)),bool(event.modifiers()&Qt.KeyboardModifier.ShiftModifier))
        self.update()

    def mouseMoveEvent(self,event):
        if self.drag_state is None:
            inside=self.area().adjusted(-6,-6,6,6).contains(event.position())
            self.hover=self.tone_at(event.position()) if inside else None
            self.update();return
        values,points,x,start_y,start_value,global_exposure=self.drag_state
        target=start_value+(start_y-event.position().y())/self.area().height()
        self.values,self.points=drag(values,x,target,global_exposure,points)
        self.hover=x
        self.changed.emit(copy.deepcopy(self.values),copy.deepcopy(self.points));self.update()

    def mouseReleaseEvent(self,event):
        if self.drag_state is not None:self.drag_state=None;self.committed.emit();self.update()

    def leaveEvent(self,event):
        self.hover=None;self.update()

    def mouseDoubleClickEvent(self,event):
        if event.button()!=Qt.MouseButton.LeftButton:return
        self.drag_state=None
        for key in (*KEYS,'exposure'):self.values[key]=0.
        self.points=copy.deepcopy(IDENTITY)
        self.changed.emit(copy.deepcopy(self.values),copy.deepcopy(self.points));self.committed.emit();self.update()

    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.RenderHint.Antialiasing);a=self.area()
        painter.fillRect(a,QColor('#161c17'))
        active=self.hover if self.hover is not None else (self.drag_state[2] if self.drag_state else None)
        if active is not None:
            # The tone range a drag here bends.
            band=QRectF(self.screen(max(0,active-2*WIDTH),1),self.screen(min(1,active+2*WIDTH),0))
            painter.fillRect(band,QColor(124,150,104,34))
        painter.setPen(QPen(QColor('#354033'),1))
        for i in range(5):
            t=i/4;painter.drawLine(self.screen(t,0),self.screen(t,1));painter.drawLine(self.screen(0,t),self.screen(1,t))
        painter.setPen(QPen(QColor('#596552'),1,Qt.PenStyle.DashLine));painter.drawLine(a.bottomLeft(),a.topRight())
        axis=np.linspace(0,1,512);values=display(self.values,self.points,axis);path=QPainterPath(self.screen(0,float(values[0])))
        for x,y in zip(axis,values):path.lineTo(self.screen(x,y))
        painter.setPen(QPen(QColor('#c6d9b5'),2));painter.drawPath(path)
        if active is not None:
            painter.setBrush(QColor('#202c1c'));painter.setPen(QPen(QColor('#e3f0d4'),1.5))
            painter.drawEllipse(self.screen(active,float(display(self.values,self.points,active))),4,4)
        weights=membership(active) if active is not None else np.zeros(4)
        for i,text in enumerate(LABELS):
            rect=QRectF(a.left()+i*a.width()/4,a.bottom()+6,a.width()/4,22)
            painter.setPen(QColor('#dcebcc') if weights[i]>=.2 else QColor('#94a68c'))
            painter.drawText(rect,Qt.AlignmentFlag.AlignCenter,text)
