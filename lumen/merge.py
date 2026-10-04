"""Photo stacks and spherical panoramas. Inputs/outputs are display-referred RGB.

HDR uses OpenCV's multiscale Mertens exposure fusion, not radiance-map HDR.
Panorama camera estimation uses OpenCV; full-resolution spherical projection and
feather blending are evaluated in tiles, before allocating the bounded canvas.
"""
import math
import cv2
import numpy as np
from . import large_image
from .i18n import tr

PANORAMA_MAX_PIXELS=200_000_000
METHODS={'focus':tr('景深合成'),'hdr':tr('HDR堆栈'),'panorama':tr('全景合成')}


def check(cancel):
    if cancel is not None and cancel.is_set():raise InterruptedError(tr('已取消合成'))


def notify(progress,value,message,cancel=None):
    check(cancel)
    if progress:progress(value,message)


def small(image,limit=1400):
    h,w=image.shape[:2];scale=min(1.,limit/max(h,w))
    out=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else image
    return out,np.diag([out.shape[1]/w,out.shape[0]/h,1.])


def gray(image):return cv2.cvtColor(np.asarray(image,np.float32),cv2.COLOR_RGB2GRAY)


def normalized(image):
    value=gray(image);lo,hi=np.percentile(value,[2,98])
    return np.clip((value-lo)/max(.01,hi-lo),0,1).astype(np.float32)


def check_memory(pixels,count):
    # OpenCV Mertens and focus analysis allocate native full-frame temporaries.
    available=large_image.available_memory();required=pixels*(120+count*80)
    if available and required>available*.8:
        raise ValueError(tr('此组堆栈预计需要约 {v:.1f} GiB 可用内存。请减少照片数量、裁切后再合成，或使用内存更大的电脑。', v=required/2**30))


def stack_transforms(images,reference,cancel=None):
    previews=[small(im) for im in images];ref,sref=previews[reference]
    sift=cv2.SIFT_create(nfeatures=6000);clahe=cv2.createCLAHE(2.,(8,8))
    features=[]
    for im,_ in previews:
        check(cancel);features.append(sift.detectAndCompute(clahe.apply(np.uint8(normalized(im)*255)),None))
    keyref,desref=features[reference];transforms=[]
    for index,(im,scale) in enumerate(previews):
        check(cancel)
        if index==reference:transforms.append(np.eye(3));continue
        key,des=features[index];matrix=None
        if des is not None and desref is not None:
            pairs=cv2.BFMatcher().knnMatch(des,desref,k=2)
            matches=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
            if len(matches)>=10:
                a=np.float32([key[m.queryIdx].pt for m in matches]);b=np.float32([keyref[m.trainIdx].pt for m in matches])
                affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=3000)
                if affine is not None and int(inliers.sum())>=10:matrix=np.vstack([affine,[0,0,1]])
        if matrix is None:
            # ECC helps low-texture and defocused frames; this matrix maps the
            # reference to the source, so invert it for our forward convention.
            resized=cv2.resize(normalized(im),(ref.shape[1],ref.shape[0]))
            warp=np.eye(2,3,dtype=np.float32)
            try:
                score,warp=cv2.findTransformECC(normalized(ref),resized,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),None,5)
                if score>.6:
                    adjust=np.diag([ref.shape[1]/im.shape[1],ref.shape[0]/im.shape[0],1.])
                    matrix=np.linalg.inv(np.vstack([warp,[0,0,1]]))@adjust
            except cv2.error:pass
        if matrix is None:
            raise ValueError(tr('第 {v} 张无法对齐。堆栈需要相同构图、足够的重叠和可识别细节。', v=index+1))
        full=np.linalg.inv(sref)@matrix@scale
        determinant=np.linalg.det(full[:2,:2])
        if not np.isfinite(full).all() or not .01<determinant<100:raise ValueError(tr('第 {v} 张对齐结果无效。', v=index+1))
        transforms.append(full)
    return transforms


def aligned_stack(images,reference,progress=None,cancel=None):
    matrices=stack_transforms(images,reference,cancel);h,w=images[reference].shape[:2]
    aligned=[];valid=large_image.allocate((h,w),np.uint8,True);valid[:]=255
    for index,(im,matrix) in enumerate(zip(images,matrices)):
        notify(progress,25+int(20*index/len(images)),tr('对齐照片 {v} / {n_images}', v=index+1, n_images=len(images)),cancel)
        result=large_image.allocate((h,w,3));mask=large_image.allocate((h,w),np.uint8)
        source_mask=np.full(im.shape[:2],255,np.uint8)
        for y in range(0,h,256):
            check(cancel);height=min(256,h-y);shift=np.array([[1,0,0],[0,1,-y],[0,0,1.]])@matrix
            result[y:y+height]=cv2.warpPerspective(im,shift,(w,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            mask[y:y+height]=cv2.warpPerspective(source_mask,shift,(w,height),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        np.minimum(valid,mask,out=valid);aligned.append(result)
    if np.count_nonzero(valid)<h*w*.15:raise ValueError(tr('对齐后的共同区域太小，请选择相同构图的照片。'))
    return aligned,valid


def inner_crop(valid):
    """Largest all-valid rectangle on a conservative <=1024-cell min-pooled grid."""
    h,w=valid.shape;step=max(1,math.ceil(max(h,w)/1024))
    reduced=np.minimum.reduceat(valid,np.arange(0,h,step),axis=0)
    grid=np.minimum.reduceat(reduced,np.arange(0,w,step),axis=1)>0
    heights=np.zeros(grid.shape[1],np.int32);best=(0,0,0,0,0)
    for y,row in enumerate(grid):
        heights=np.where(row,heights+1,0);stack=[]
        for x in range(len(heights)+1):
            value=int(heights[x]) if x<len(heights) else 0;start=x
            while stack and stack[-1][1]>value:
                left,height=stack.pop();area=height*(x-left)
                if area>best[0]:best=(area,left,y-height+1,x,y+1)
                start=left
            if not stack or stack[-1][1]<value:stack.append((start,value))
    if not best[0]:raise ValueError(tr('没有可用的完整矩形区域，请关闭自动裁边或重新选择照片。'))
    _,x0,y0,x1,y1=best
    return x0*step,y0*step,min(w,x1*step),min(h,y1*step)


def ghost_mask(images,reference,level):
    if level not in ('low','medium','high'):raise ValueError(tr('无效去伪影档位'))
    threshold,radius={'low':(.20,2),'medium':(.12,5),'high':(.065,9)}[level]
    ref=gray(images[reference]);refblur=cv2.GaussianBlur(ref,(0,0),1.2)
    sampling=max(1,math.ceil(max(ref.shape)/1000));percentiles=np.linspace(2,98,33)
    target=np.percentile(ref[::sampling,::sampling],percentiles)
    motion=np.zeros(ref.shape,np.float32)
    for index,im in enumerate(images):
        if index==reference:continue
        lum=gray(im);values=np.percentile(lum[::sampling,::sampling],percentiles)
        unique,positions=np.unique(values,return_index=True)
        mapped=np.interp(lum,unique,target[positions]).astype(np.float32)
        error=np.abs(cv2.GaussianBlur(mapped,(0,0),1.2)-refblur)
        # Clipped exposures lack comparison detail; they are not evidence of motion.
        usable=(lum>.015)&(lum<.985)&(ref>.015)&(ref<.985)
        np.maximum(motion,(error>threshold)*usable,out=motion)
    radius=max(1,round(radius*max(ref.shape)/1800))
    motion=cv2.dilate(motion,np.ones((radius*2+1,radius*2+1),np.uint8))
    return np.clip(cv2.GaussianBlur(motion,(0,0),max(1,radius/2)),0,1)


def fuse_hdr(images,reference,ghost='medium',cancel=None):
    check(cancel)
    motion=ghost_mask(images,reference,ghost)
    ref=images[reference];step=max(1,math.ceil(max(ref.shape[:2])/1000))
    static=motion[::step,::step]<.01
    if static.sum()<64:static=np.ones_like(static,dtype=bool)
    quantiles=np.linspace(0,100,129);mappings=[]
    for channel in range(3):
        values=np.percentile(ref[::step,::step,channel][static],quantiles)
        unique,positions=np.unique(values,return_index=True);mappings.append((unique,positions))
    # MergeMertens internally divides its float inputs by 255; keeping float
    # data in that range avoids quantizing 16-bit RAW data to 8-bit first.
    # Substitute motion before pyramid fusion, mapping the reference content
    # back to each exposure. Pasting the bright reference over a fused image
    # afterwards would leave an obvious bright rectangle around moving objects.
    inputs=[]
    for index,im in enumerate(images):
        check(cancel);prepared=large_image.allocate(im.shape)
        target=[np.percentile(im[::step,::step,c][static],quantiles)[mappings[c][1]] for c in range(3)]
        for y,block in large_image.strips(prepared):
            original=im[y:y+len(block)]
            if index==reference:block[:]=original*255;continue
            alpha=motion[y:y+len(block),:,None]
            replacement=np.stack([np.interp(ref[y:y+len(block),:,c],mappings[c][0],target[c]) for c in range(3)],axis=2)
            block[:]=(original*(1-alpha)+replacement*alpha)*255
        inputs.append(prepared)
    output=cv2.createMergeMertens(1.,1.,1.).process(inputs);del inputs
    check(cancel)
    return np.clip(output,0,1).astype(np.float32)


def fuse_focus(images,cancel=None):
    h,w=images[0].shape[:2];best=np.full((h,w),-1.,np.float32);winner=np.zeros((h,w),np.uint8)
    for index,im in enumerate(images):
        check(cancel);lum=gray(im)
        score=np.zeros((h,w),np.float32)
        for sigma in (.7,1.4,2.8):
            lap=np.abs(cv2.Laplacian(cv2.GaussianBlur(lum,(0,0),sigma),cv2.CV_32F,ksize=3))
            score+=cv2.GaussianBlur(lap,(0,0),2.5)/sigma
        choose=score>best;winner[choose]=index;np.maximum(best,score,out=best)
    # Median labels suppress isolated noisy picks. Feather the boundaries only;
    # broad averaging would reintroduce the very defocus being removed.
    winner=cv2.medianBlur(winner,5)
    output=large_image.allocate((h,w,3),zeros=True);weights=np.zeros((h,w),np.float32)
    for index,im in enumerate(images):
        check(cancel);weight=cv2.GaussianBlur((winner==index).astype(np.float32),(0,0),1.5);weights+=weight
        for y,block in large_image.strips(output):block+=im[y:y+len(block)]*weight[y:y+len(block),:,None]
    for y,block in large_image.strips(output):block/=np.maximum(weights[y:y+len(block),:,None],1e-6)
    return output


def bounded_rois(sizes,matrices,rotations,scale,maximum=PANORAMA_MAX_PIXELS):
    for _ in range(8):
        warper=cv2.PyRotationWarper('spherical',float(scale))
        rois=[warper.warpRoi((w,h),k.astype(np.float32),r.astype(np.float32)) for (h,w),k,r in zip(sizes,matrices,rotations)]
        x0=min(r[0] for r in rois);y0=min(r[1] for r in rois)
        x1=max(r[0]+r[2] for r in rois);y1=max(r[1]+r[3] for r in rois)
        w,h=x1-x0,y1-y0
        if w<=0 or h<=0 or not np.isfinite(scale):raise ValueError(tr('全景投影无效。'))
        if w*h<=maximum:return rois,(x0,y0,w,h),scale
        scale*=math.sqrt(maximum/(w*h))*.998
    raise ValueError(tr('无法将全景画布限制在 2 亿像素内。'))


def spherical_map(rect,scale,k,r):
    x,y,w,h=rect;u=(np.arange(w,dtype=np.float32)+x)/scale;v=(np.arange(h,dtype=np.float32)+y)/scale
    sinv=np.sin(v)[:,None]
    rays=np.stack(np.broadcast_arrays(sinv*np.sin(u)[None,:],-np.cos(v)[:,None],sinv*np.cos(u)[None,:]),axis=2)
    camera=rays@r;projected=camera@k.T;z=projected[...,2]
    good=z>1e-6;z=np.where(good,z,1.)
    return np.where(good,projected[...,0]/z,-1).astype(np.float32),np.where(good,projected[...,1]/z,-1).astype(np.float32)


def panorama(images,reference,progress=None,cancel=None,maximum=PANORAMA_MAX_PIXELS):
    previews=[small(im,1200) for im in images]
    notify(progress,22,tr('估算全景相机位置与球面投影…'),cancel)
    # SIFT gives more stable focal/camera estimation than the high-level
    # Stitcher's ORB defaults on low-contrast photographic terrain.
    finder=cv2.SIFT_create(nfeatures=6000);features=[]
    cv2.setRNGSeed(121)
    for im,_ in previews:
        check(cancel);features.append(cv2.detail.computeImageFeatures2(finder,np.uint8(np.clip(im[...,::-1],0,1)*255)))
    if any(len(feature.keypoints)<10 for feature in features):
        raise ValueError(tr('全景匹配失败：部分照片缺少可识别细节，请移除纯色、严重模糊或过曝的照片。'))
    # The default cutoff of 3 treats exceptionally consistent matches as
    # duplicate images. Dense, well-aligned textures also exceed that cutoff;
    # keep those valid pairs (the confidence formula is bounded by 10/3).
    matcher=cv2.detail.BestOf2NearestMatcher_create(False,.3,6,6,3.5)
    matches=matcher.apply2(features);matcher.collectGarbage();check(cancel)
    edges={i:{i} for i in range(len(images))}
    for match in matches:
        if match.confidence>.6 and match.num_inliers>=10:
            edges[match.src_img_idx].add(match.dst_img_idx);edges[match.dst_img_idx].add(match.src_img_idx)
    connected={reference};previous=set()
    while previous!=connected:
        previous=connected.copy()
        for i in previous:connected.update(edges[i])
    if len(connected)!=len(images):raise ValueError(tr('全景匹配失败：部分照片没有足够重叠或可识别细节，请移除这些照片后重试。'))
    success,cameras=cv2.detail_HomographyBasedEstimator().apply(features,matches,None)
    if not success:raise ValueError(tr('无法估算全景相机参数，请选择同一视点、约 30% 以上重叠的照片。'))
    for camera in cameras:camera.R=camera.R.astype(np.float32)
    adjuster=cv2.detail_BundleAdjusterRay();adjuster.setConfThresh(.6)
    refinement=np.zeros((3,3),np.uint8);refinement[0,0]=refinement[0,1]=refinement[0,2]=refinement[1,1]=refinement[1,2]=1
    adjuster.setRefinementMask(refinement);success,cameras=adjuster.apply(features,matches,cameras);check(cancel)
    if not success:raise ValueError(tr('全景相机优化失败。请避免明显视差、模糊或重复纹理，并增加照片间的重叠。'))
    corrected=cv2.detail.waveCorrect([camera.R.copy() for camera in cameras],cv2.detail.WAVE_CORRECT_HORIZ)
    for camera,rotation in zip(cameras,corrected):camera.R=rotation
    component=list(range(len(images)));ordered=images
    matrices=[np.linalg.inv(scale)@camera.K() for (_,scale),camera in zip(previews,cameras)]
    rotations=[cam.R for cam in cameras];sizes=[im.shape[:2] for im in ordered]
    scale=float(np.median([k[0,0] for k in matrices]))
    rois,bounds,scale=bounded_rois(sizes,matrices,rotations,scale,maximum)
    x0,y0,w,h=bounds
    if w*h>maximum:raise ValueError(tr('全景超过 2 亿像素。'))
    notify(progress,35,tr('全景画布 {w} × {h}，开始分块融合…', w=w, h=h),cancel)
    output=large_image.allocate((h,w,3));valid=large_image.allocate((h,w),np.uint8,True)
    # Estimate scalar exposure gains from low-resolution overlaps, anchored to
    # the chosen reference. No geometrically disconnected photo is silently lost.
    ratio=min(1.,900/max(w,h));sample_w=max(1,round(w*ratio));sample_h=max(1,round(h*ratio))
    samples=[];masks=[]
    for im,k,r in zip(ordered,matrices,rotations):
        mx,my=spherical_map((x0*ratio,y0*ratio,sample_w,sample_h),scale*ratio,k,r)
        mask=(mx>=1)&(my>=1)&(mx<im.shape[1]-2)&(my<im.shape[0]-2)
        samples.append(gray(cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)));masks.append(mask)
    equations=[];values=[]
    for i in range(len(images)):
        for j in range(i+1,len(images)):
            mask=masks[i]&masks[j]&(samples[i]>.04)&(samples[j]>.04)&(samples[i]<.9)&(samples[j]<.9)
            if mask.sum()<80:continue
            row=np.zeros(len(images));row[i]=1;row[j]=-1;equations.append(row)
            values.append(float(np.median(np.log(samples[j][mask]/samples[i][mask]))))
    row=np.zeros(len(images));row[component.index(reference)]=10;equations.append(row);values.append(0.)
    gains=np.exp(np.clip(np.linalg.lstsq(equations,values,rcond=None)[0],-.5,.5))
    tile=512;total=math.ceil(h/tile)*math.ceil(w/tile);done=0
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            check(cancel);th,tw=min(tile,h-y),min(tile,w-x)
            accum=np.zeros((th,tw,3),np.float32);weight=np.zeros((th,tw),np.float32)
            for im,k,r,roi,gain in zip(ordered,matrices,rotations,rois,gains):
                if roi[0]+roi[2]<x+x0 or roi[0]>x+x0+tw or roi[1]+roi[3]<y+y0 or roi[1]>y+y0+th:continue
                mx,my=spherical_map((x+x0,y+y0,tw,th),scale,k,r)
                ih,iw=im.shape[:2];inside=(mx>=0)&(my>=0)&(mx<=iw-1)&(my<=ih-1)
                edge=np.minimum(np.minimum(mx+1,iw-mx),np.minimum(my+1,ih-my))
                alpha=np.clip(edge/(min(ih,iw)*.12),.001,1)*inside
                warped=cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                accum+=warped*(alpha*gain)[...,None];weight+=alpha
            output[y:y+th,x:x+tw]=np.clip(accum/np.maximum(weight[...,None],1e-8),0,1)
            valid[y:y+th,x:x+tw]=np.uint8(weight>0)*255
            done+=1;notify(progress,35+int(55*done/total),tr('全景融合分块 {done} / {total}', done=done, total=total),cancel)
    return output,valid,dict(canvas=[w,h],projection='spherical',pixel_limit=maximum,gains=gains.tolist())


def merge_images(images,kind,reference=0,ghost='medium',crop=True,align=True,progress=None,cancel=None):
    if kind not in METHODS or not 2<=len(images)<=32:raise ValueError(tr('合成需选择 2–32 张照片。'))
    if not 0<=reference<len(images):raise ValueError(tr('无效参考照片。'))
    for image in images:
        large_image.validate_size(image.shape)
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError(tr('合成需要 RGB 照片。'))
    check(cancel)
    if kind=='panorama':output,valid,info=panorama(images,reference,progress,cancel)
    else:
        h,w=images[reference].shape[:2];check_memory(h*w,len(images))
        if align:images,valid=aligned_stack(images,reference,progress,cancel)
        else:
            if any(im.shape!=images[reference].shape for im in images):raise ValueError(tr('关闭对齐时，堆栈照片尺寸必须相同。'))
            valid=np.full((h,w),255,np.uint8)
        notify(progress,50,tr('Mertens 曝光融合与去伪影…') if kind=='hdr' else tr('分析多尺度清晰度并合成景深…'),cancel)
        output=fuse_hdr(images,reference,ghost,cancel) if kind=='hdr' else fuse_focus(images,cancel)
        info=dict(algorithm='Mertens exposure fusion' if kind=='hdr' else 'multiscale focus selection',deghost=ghost if kind=='hdr' else None)
    check(cancel)
    if crop:
        x0,y0,x1,y1=inner_crop(valid);output=output[y0:y1,x0:x1];info['crop']=[x0,y0,x1,y1]
    if not large_image.finite(output):raise ValueError(tr('合成结果包含无效像素。'))
    notify(progress,94,tr('合成完成，准备保存…'),cancel)
    return output,info
