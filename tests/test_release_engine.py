import copy
from pathlib import Path
import numpy as np
import cv2
import pytest
import rawpy
import tifffile
from lumen import engine,model,develop,selection,geometry,dng


def test_linear_dng_is_three_channel_16bit_and_independently_readable(tmp_path):
    rgb=np.random.default_rng(5).random((64,89,3),dtype=np.float32)*.82
    path=tmp_path/'成片.dng'
    engine.export_image(path,rgb)
    with tifffile.TiffFile(path) as f:
        assert len(f.pages)==1 and f.pages[0].shape==rgb.shape
        assert f.pages[0].samplesperpixel==3 and f.pages[0].dtype==np.uint16
        assert f.pages[0].photometric==34892
        assert f.pages[0].tags[50706].value==bytes([1,4,0,0])
    read,info=engine.load_image(path,None)
    assert not info['raw'] and dng.is_rendered(path)
    np.testing.assert_allclose(read,engine.to_linear(rgb),atol=1/65535)
    with rawpy.imread(str(path)) as raw:
        other=raw.postprocess(use_camera_wb=True,no_auto_bright=True,adjust_maximum_thr=0,
            output_bps=16,gamma=(1,1),output_color=rawpy.ColorSpace.sRGB).astype(np.float32)/65535
    np.testing.assert_allclose(other,engine.to_linear(rgb),atol=.00015)


def test_develop_curve_preserves_hue_is_saved_and_not_synced():
    r=model.recipe()
    r['develop']=dict(mode='camera',curve=[[0,0],[.2,.45],[.7,.9],[1,1]],source='test')
    src=np.array([[[.05,.1,.15],[.1,.2,.3]]],np.float32)
    out=develop.apply(src,r['develop'])
    assert np.mean(out)>np.mean(src)
    np.testing.assert_allclose(out[...,0]/out[...,1],src[...,0]/src[...,1],atol=1e-5)
    # 2.3.1: validation adds the camera colour reference, 1 (colours as decoded) when absent.
    assert model.validate(r)['develop']==dict(r['develop'],chroma=1.)
    selected=model.apply_look(r,model.builtin_presets()[1]['look'])
    assert selected['develop']==r['develop']
    old=model.recipe();old['version']=3;old.pop('develop')
    assert model.validate(old)['develop']['mode']=='linear'


def test_color_flood_selects_only_connected_similar_region():
    rgb=np.full((90,120,3),[.1,.5,.2],np.float32)
    rgb[:,55:65]=[.8,.1,.1]
    a=selection.color_region(rgb,[.2,.5],10)
    assert a[40,20]==1 and a[40,80]==0 and a[40,60]==0
    rgb[40:50,55:65]=[.1,.5,.2]
    b=selection.color_region(rgb,[.2,.5],10)
    assert b[40,80]==1


def test_raster_masks_roundtrip_refine_and_scaling(tmp_path):
    alpha=np.zeros((90,120),np.float32);alpha[:45]=1
    mask=model.new_mask('sky',1)
    mask.update(raster=selection.encode(alpha),feather=0)
    mask['adjustments']['exposure']=1
    r=model.recipe();r['masks']=[mask]
    model.save_project(tmp_path/'天空.lumen',tmp_path/'raw.arw',r)
    _,loaded=model.load_project(tmp_path/'天空.lumen')
    assert loaded['masks'][0]['raster']==mask['raster']
    hi=engine.mask_alpha(mask,(180,240,3))
    assert hi[30,30]==1 and hi[150,30]==0
    mask['strokes']=[dict(points=[[.3,.2]],radius=.08,erase=True)]
    a=engine.mask_alpha(mask,(90,120,3))
    assert a[18,36]<.01 and a[10,10]==1
    mask['invert']=True
    b=engine.mask_alpha(mask,(90,120,3))
    np.testing.assert_allclose(a+b,1,atol=1e-6)


@pytest.mark.parametrize('rotation',[0,1,2,3])
@pytest.mark.parametrize('angle',[0,5,-4])
def test_crop_mapping_matches_rendered_pixels(rotation,angle):
    r=model.recipe();r.update(crop=[.1,.2,.9,.85],rotation=rotation,straighten=angle)
    h,w=201,301
    yy,xx=np.mgrid[0:h,0:w]
    src=np.stack([xx/(w-1),yy/(h-1),np.zeros_like(xx)],axis=2).astype(np.float32)
    out=engine.crop_rotate(src,r)
    mat,size,_=geometry.frame(r,(w,h))
    assert size==(out.shape[1],out.shape[0])
    inverse=np.linalg.inv(mat)
    for u,v in [(.2,.2),(.5,.5),(.8,.8)]:
        x,y=round(u*(size[0]-1)),round(v*(size[1]-1))
        expected=geometry.point(inverse,[x/(size[0]-1),y/(size[1]-1)])
        np.testing.assert_allclose(out[y,x,:2],expected,atol=.0005)
        np.testing.assert_allclose(geometry.point(mat,expected),[x/(size[0]-1),y/(size[1]-1)],atol=1e-10)


def test_mask_rejects_invalid_raster_before_decode():
    r=model.recipe();m=model.new_mask('person',1);m['raster']='invalid';r['masks']=[m]
    with pytest.raises((ValueError,TypeError)):model.validate(r)


def test_real_person_and_background_networks_return_probabilities():
    rgb=np.full((160,240,3),.5,np.float32)
    for kind in ['person','background']:
        result,provider=selection.automatic(rgb,kind)
        assert result.shape==(160,240) and np.isfinite(result).all()
        assert result.min()>=0 and result.max()<=1 and 'CPU' in provider
