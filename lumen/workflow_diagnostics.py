"""Exercise the 1.1 desktop workflow and real full-resolution AI copies."""
def run(destination,primary,others):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import copy,hashlib,json,time,traceback
    from pathlib import Path
    import numpy as np
    from PySide6.QtWidgets import QApplication,QMessageBox
    from .app import MainWindow,STYLE
    from .ai_dialog import EnhancementDialog
    from .watermark_dialog import WatermarkDialog
    from . import engine,watermark
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    report={};errors=[]
    QMessageBox.warning=lambda *a:errors.append(str(a[2]))
    QMessageBox.information=lambda *a:None
    try:
        paths=[str(Path(p).resolve()) for p in [primary,*others]]
        original_hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
        app=QApplication.instance() or QApplication([]);app.setStyle('Fusion');app.setStyleSheet(STYLE)
        w=MainWindow();w.show();app.processEvents()
        for i in range(w.tabs.count()):
            w.tabs.setCurrentIndex(i);app.processEvents()
            assert w.tabs.tabBar().isEnabled() and not w.tabs.widget(i).widget().isEnabled()
        w.show_tab('details');w.grab().save(str(out/'empty-browse.png'))
        def settle(timeout=300):
            end=time.monotonic()+timeout
            while w.loading or w.jobs or w.timer.isActive() or w.detail_timer.isActive() or w.ai_busy:
                app.processEvents();time.sleep(.01)
                if time.monotonic()>end:raise TimeoutError('工作流超时')
            app.processEvents()
            if errors:raise RuntimeError('; '.join(errors))
        w.import_paths(paths);settle()
        brands=[]
        for path in paths:
            w.open_path(path);settle()
            assert w.info['raw'] and w.info['photo'].get('body')
            brands.append(dict(file=Path(path).name,shape=list(w.source.shape),photo=w.info['photo']))
        w.open_path(paths[0]);settle();w.show_tab('masks')
        report['mask_results']={}
        for kind in ('subject','foreground'):
            w.create_auto_mask(kind);settle()
            mask=w.selected_mask()
            found=mask is not None and mask['kind']==kind
            if not found:assert '没有识别到明确区域' in w.statusBar().currentMessage()
            report['mask_results'][kind]='created' if found else 'no clear region, correctly reported'
            w.grab().save(str(out/(kind+'-mask.png')))
        w.edits['masks']=[];w.current_mask=-1
        w.edits['watermark'].update(enabled=True,preset='gallery',size=18.)
        w.refresh();w.changed();settle()
        dialog=WatermarkDialog(w);dialog.show();app.processEvents();dialog.grab().save(str(out/'watermark-dialog.png'));dialog.reject()
        w.grab().save(str(out/'workspace.png'))
        copies=[]
        for kind in ('denoise','super'):
            w.open_path(paths[0]);settle()
            dialog=EnhancementDialog(w,kind);dialog.folder.setText(str(out));dialog.show();app.processEvents()
            dialog.start(True);settle()
            assert dialog.after.pixmap() is not None
            dialog.grab().save(str(out/(kind+'-dialog.png')))
            started=time.monotonic();dialog.start(False);settle(1800)
            assert dialog.output_path and Path(dialog.output_path).exists()
            assert w.source_path==dialog.output_path and not w.info['raw']
            full,info=engine.load_image(dialog.output_path,None)
            expected=[w.documents[paths[0]]['edits']['watermark']['enabled'],True]
            assert expected[0] and w.edits['watermark']['enabled']
            assert info['photo']['date']==brands[0]['photo']['date']
            assert w.edits['crop'] is None and not w.edits['masks']
            copies.append(dict(kind=kind,file=Path(dialog.output_path).name,dimensions=[full.shape[1],full.shape[0]],seconds=round(time.monotonic()-started,2),photo=info['photo']))
            del full
        w.grab().save(str(out/'copies-filmstrip.png'))
        w.start_export(str(out/'watermarked.jpg'));settle()
        w.save_album(path=str(out/'多品牌与AI副本.lumenalbum'));w.open_album(str(out/'多品牌与AI副本.lumenalbum'));settle()
        assert len(w.documents)==len(paths)+2
        for p,digest in original_hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
        report.update(ok=True,brands=brands,copies=copies,original_hashes=original_hashes,source_unchanged=True,album=True,watermark=True)
        w.save_album(path=str(out/'多品牌与AI副本.lumenalbum'));w.close();app.processEvents()
    except Exception:report.update(ok=False,error=traceback.format_exc(),ui_errors=errors)
    (out/'workflow-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return 0 if report.get('ok') else 1
