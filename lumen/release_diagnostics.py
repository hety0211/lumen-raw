"""Headless integration check of the 1.0 workflow using user-selected RAW files."""
def run(output, sources):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import copy
    import hashlib
    import json
    import time
    import traceback
    from pathlib import Path
    import numpy as np
    from PySide6.QtCore import Qt,QPointF
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication,QMessageBox
    from .app import MainWindow,STYLE
    from . import model,engine,selection
    destination=Path(output);destination.mkdir(parents=True,exist_ok=True)
    report={}
    errors=[]
    QMessageBox.warning=lambda *args:errors.append(str(args[2]))
    QMessageBox.information=lambda *args:None
    paths=[str(Path(p).resolve()) for p in sources]
    before={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    try:
        app=QApplication.instance() or QApplication([])
        app.setStyle('Fusion');app.setStyleSheet(STYLE)
        w=MainWindow();w.show()
        def settle(timeout=90):
            deadline=time.monotonic()+timeout
            while w.loading or w.render_running or w.timer.isActive() or w.jobs or w.full_busy or w.detail_timer.isActive():
                app.processEvents();time.sleep(.01)
                if time.monotonic()>deadline:raise TimeoutError('1.0 integration timed out')
            app.processEvents()
            if errors:raise RuntimeError('; '.join(errors))
        w.import_paths(paths);settle()
        assert len(w.documents)==len(paths)
        w.grab().save(str(destination/'filmstrip.png'))
        rows=[]
        for index,path in enumerate(paths):
            w.canvas.fit()
            w.open_path(path);settle()
            median=float(np.median(w.rendered@np.array([.2126,.7152,.0722])))
            started=time.monotonic()
            w.canvas.actual_size();settle()
            # 1.4.0: the visible part is rendered as original-resolution tiles over the preview.
            view=w.detail_view()
            assert view is not None and view.level==0 and w.detail_ready()
            assert view.size==(w.full_source.shape[1],w.full_source.shape[0])
            assert abs(w.canvas.image_rect().width()*w.canvas.devicePixelRatioF()-view.size[0])<.1
            from .viewport import tile_array
            tiles=[w.canvas.detail.main[i] for i in view.tiles]
            pixels=np.concatenate([tile_array(t).reshape(-1,3) for t in tiles])
            full_median=float(np.median(pixels[::4]@np.array([.2126,.7152,.0722])))/255
            # Compare with the same part of the preview.
            x0,y0=min(t.x0 for t in tiles)/view.size[0],min(t.y0 for t in tiles)/view.size[1]
            x1,y1=max(t.x0+t.width for t in tiles)/view.size[0],max(t.y0+t.height for t in tiles)/view.size[1]
            ph,pw=w.rendered.shape[:2]
            part=w.rendered[round(y0*ph):round(y1*ph),round(x0*pw):round(x1*pw)]
            median=float(np.median(part@np.array([.2126,.7152,.0722])))
            assert abs(full_median-median)<.025
            w.canvas.offset+=QPointF(w.canvas.width()/2,w.canvas.height()/2)-w.canvas.screen([.4,.65])
            w.canvas.viewport_changed.emit();settle()
            assert w.detail_ready()
            if index==len(paths)-1:w.grab().save(str(destination/'original-100.png'))
            rows.append(dict(file=Path(path).name,dimensions=[w.info['width'],w.info['height']],
                preview_median=median,full_median=full_median,detail_seconds=round(time.monotonic()-started,2)))
        w.canvas.fit();settle()
        report['packaged_mask_models']={}
        for kind in ('person','background'):
            alpha,provider=selection.automatic(w.rendered,kind,False)
            assert alpha.shape==w.rendered.shape[:2] and np.isfinite(alpha).all()
            report['packaged_mask_models'][kind]=provider
        w.show_tab('masks')
        w.create_auto_mask('sky');settle()
        assert w.selected_mask()['kind']=='sky'
        sky=engine.mask_alpha(w.selected_mask(),w.source.shape)
        assert float(sky[:sky.shape[0]//4].mean())>.85
        assert float(sky[-sky.shape[0]//4:].mean())<.1
        w.grab().save(str(destination/'sky-mask.png'))
        w.local_controls['exposure'].spin.setValue(-.2);settle()
        w.start_crop()
        w.edits['crop']=[.1,.15,.9,.9];w.update_display()
        w.canvas.setFocus();QTest.keyClick(w.canvas,Qt.Key.Key_Return);settle()
        assert w.final_view.isChecked() and w.canvas.crop is None
        w.grab().save(str(destination/'crop-confirmed.png'))
        w.controls['contrast'].spin.setValue(10);settle()
        for i in range(w.filmstrip.count()):w.filmstrip.item(i).setSelected(True)
        w.sync_look()
        for path in paths[:-1]:
            record=w.documents[path]
            assert record['edits']['adjustments']['contrast']==10
            assert record['edits']['crop'] is None and not record['edits']['masks']
        out=destination/'edited.dng'
        w.start_export(str(out));settle()
        rgb,info=engine.load_image(out,None)
        assert rgb.shape[1]>1600 and rgb.dtype==np.float32
        report['dng_dimensions']=[rgb.shape[1],rgb.shape[0]]
        del rgb
        assert w.save_album(path=str(destination/'旅行选片.lumenalbum'))
        w.open_album(str(destination/'旅行选片.lumenalbum'));settle()
        assert len(w.documents)==len(paths)
        w.show_tab('color');w.tabs.widget(w.tab_index('color')).ensureWidgetVisible(w.curve);app.processEvents()
        w.grab().save(str(destination/'workspace.png'))
        for path in paths:assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==before[path]
        report.update(ok=True,photos=rows,source_hashes_unchanged=True,source_hashes=before,sky=True,album=True,full_resolution=True)
        w.save_album(path=str(destination/'旅行选片.lumenalbum'))
        w.close();app.processEvents()
    except Exception:
        report.update(ok=False,error=traceback.format_exc(),ui_errors=errors)
    (destination/'release-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if report.get('ok') else 1
