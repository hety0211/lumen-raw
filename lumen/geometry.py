"""Source/display coordinate transforms identical to engine.crop_rotate."""
import cv2
import numpy as np


def frame(edits, size):
    w,h = size
    a,b,c,d = edits.get('crop') or [0,0,1,1]
    x0,y0 = min(w-1,int(a*w)),min(h-1,int(b*h))
    x1,y1 = max(x0+1,min(w,round(c*w))),max(y0+1,min(h,round(d*h)))
    cw,ch = x1-x0,y1-y0
    matrix = np.array([[w-1,0,-x0],[0,h-1,-y0],[0,0,1.]],dtype=float)
    angle = edits.get('straighten',0)
    scale = 1.
    if abs(angle)>.001:
        theta=np.deg2rad(abs(angle))
        scale=max(np.cos(theta)+ch/cw*np.sin(theta),np.cos(theta)+cw/ch*np.sin(theta))
        affine=cv2.getRotationMatrix2D(((cw-1)/2,(ch-1)/2),angle,scale)
        matrix=np.vstack([affine,[0,0,1]])@matrix
    rotation=edits.get('rotation',0)%4
    if rotation==1:
        matrix=np.array([[0,-1,ch-1],[1,0,0],[0,0,1]])@matrix
    elif rotation==2:
        matrix=np.array([[-1,0,cw-1],[0,-1,ch-1],[0,0,1]])@matrix
    elif rotation==3:
        matrix=np.array([[0,1,0],[-1,0,cw-1],[0,0,1]])@matrix
    ow,oh=(ch,cw) if rotation%2 else (cw,ch)
    # 2.3.0: mirrored output (engine.flipped).
    horizontal,vertical=(edits.get('flip') or (False,False))[:2]
    if horizontal:
        matrix=np.array([[-1,0,ow-1],[0,1,0],[0,0,1.]])@matrix
    if vertical:
        matrix=np.array([[1,0,0],[0,-1,oh-1],[0,0,1.]])@matrix
    matrix=np.diag([1/max(1,ow-1),1/max(1,oh-1),1])@matrix
    return matrix,(ow,oh),scale*min(w,h)/max(1,min(ow,oh))


def point(matrix,p):
    result=matrix@np.array([p[0],p[1],1.])
    return (result[:2]/result[2]).tolist()
