"""Portable-package smoke test: python main.py --smoke-test sample.ARW output_dir."""
def run(source, output):
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import json
    import traceback
    from pathlib import Path
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    # 2.0.0: the smoke test's photo library and preview cache are its own, never the user's.
    os.environ.setdefault('LUMEN_LIBRARY', str(destination / 'library.db'))
    os.environ.setdefault('LUMEN_PREVIEW_CACHE', str(destination / 'previews'))
    os.environ.setdefault('LUMEN_CATALOG', str(destination / 'catalog.jsonl'))
    report = {}
    try:
        import copy
        import numpy as np
        from PySide6.QtWidgets import QApplication
        from .app import MainWindow, STYLE
        from . import engine, model, compute
        import time
        app = QApplication.instance() or QApplication([])
        app.setStyle('Fusion')
        app.setStyleSheet(STYLE)
        w = MainWindow()
        w.show()
        def settle():
            deadline = time.monotonic() + 40
            while w.loading or w.render_running or w.timer.isActive() or w.jobs:
                app.processEvents()
                time.sleep(.01)
                if time.monotonic() > deadline:
                    raise TimeoutError('Desktop diagnostics timed out')
            app.processEvents()
        w.open_path(str(Path(source).resolve()))
        settle()
        w.choose_preset(0)
        w.preset_list.setCurrentRow(0)
        settle()
        w.add_snapshot('自然风光 · 初稿')
        w.grab().save(str(destination / 'interface.png'))
        w.tabs.setCurrentIndex(1);w.tabs.widget(1).ensureWidgetVisible(w.curve)
        w.curve.set_output(60, 72)
        w.curve.set_output(190, 205)
        settle()
        app.processEvents()
        w.grab().save(str(destination / 'curves.png'))
        w.tabs.setCurrentIndex(6)
        w.set_wheel('shadows', 210., 20.)
        w.set_wheel('highlights', 40., 25.)
        settle()
        w.grab().save(str(destination / 'grading.png'))
        w.split_check.setChecked(True)
        app.processEvents()
        w.grab().save(str(destination / 'compare.png'))
        w.split_check.setChecked(False)
        w.tabs.setCurrentIndex(4)
        w.add_mask('radial')
        w.selected_mask()['adjustments']['exposure'] = .3
        w.refresh()
        app.processEvents()
        w.grab().save(str(destination / 'masks.png'))
        w.add_mask('luminance')
        w.range_value('low', 65)
        w.selected_mask()['adjustments']['highlights'] = -30
        w.changed()
        settle()
        w.grab().save(str(destination / 'luminance.png'))
        w.tabs.setCurrentIndex(8)
        w.add_retouch(dict(kind='heal', points=[[.2,.8]], radius=.008, erase=False))
        w.add_retouch(dict(kind='clone', points=[[.8,.75]], radius=.012, offset=[-.05,0], erase=False))
        settle()
        w.grab().save(str(destination / 'retouch.png'))
        from .enhance_dialog import ExportDialog
        dialog = ExportDialog(w, w.source_path, True)
        dialog.show()
        dialog.preview()
        settle()
        if dialog.after.pixmap() is None:
            raise RuntimeError('Real neural preview failed: ' + dialog.status.text())
        dialog.grab().save(str(destination / 'enhance.png'))
        report['enhance'] = dialog.status.text()
        dialog.reject()
        w.tabs.setCurrentIndex(0)
        w.tabs.widget(0).verticalScrollBar().setValue(w.tabs.widget(0).verticalScrollBar().maximum())
        app.processEvents()
        w.grab().save(str(destination / 'white-balance.png'))
        report['natural_language'] = natural_language(w, app, settle, destination)
        report['lens'] = lens_corrections(w, app, settle, destination)
        report['languages'] = languages()
        report['library'] = library_and_culling(w, app, source, destination)
        report['models'] = models_load()
        report['video'] = video_encoding(destination)
        model.save_project(destination / 'smoke.lumen', source, w.edits, w.snapshots)
        engine.export_image(destination / 'smoke.tif', w.rendered)
        report.update(ok=True, info=w.info, preview_shape=w.rendered.shape, backend=w.backend.name,
                      compute=dict(zip(('provider','device','detail','warning'), compute.state.snapshot())),
                      finite=bool(np.isfinite(w.rendered).all()), qt='rendered', raw='decoded', tiff='exported')
        w.saved_edits = copy.deepcopy(w.edits)
        w.saved_snapshots = copy.deepcopy(w.snapshots)
        w.timer.stop()
        w.close()
        app.processEvents()
    except Exception:
        report.update(ok=False, error=traceback.format_exc())
    (destination / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report.get('ok') else 1


def lens_corrections(w, app, settle, destination):
    """1.5.1: bundled lensfun database, the photo's profile (or a fixed one) and a corrected render."""
    import numpy as np
    from . import lens
    db = lens.database()
    if not db.lenses:
        raise RuntimeError('lens database missing')
    w.commit()
    match = w.lens_match()
    if not match or not match.get('lens'):
        # Synthetic or unknown samples: exercise the renderer with a fixed profile.
        w.edits['lens']['manual'] = dict(maker='Nikon', model='Nikkor Z 14-24mm f/2.8 S')
        w.info['lens_match'] = dict(match or {}, focal=14, aperture=4, crop=1.)
    before = w.rendered.copy()
    w.tabs.setCurrentIndex(5)
    w.lens_enable.setChecked(True)
    settle()
    if w.edits['lens']['profile'] is None or np.array_equal(before, w.rendered):
        raise RuntimeError('lens correction not applied: ' + w.lens_status.text())
    w.tabs.widget(5).ensureWidgetVisible(w.lens_enable)
    app.processEvents()
    w.grab().save(str(destination / 'lens.png'))
    profile = w.edits['lens']['profile']
    result = dict(lenses=len(db.lenses), cameras=len(db.cameras), match=match, profile=profile.get('label'),
                  focal=profile.get('focal'), crop=profile.get('crop'),
                  has=[k for k in ('distortion', 'tca', 'vignetting') if profile.get(k)])
    w.undo(-1)
    settle()
    return result


def video_encoding(destination):
    """2.0.0: PyAV and its FFmpeg libraries are bundled: a short H.264 clip on the CPU encoder, and
    the encoder the graphics card would use."""
    import numpy as np
    import av
    from . import timelapse
    path = destination / 'timelapse-check.mp4'
    encoder = timelapse.Encoder(path, 'h264', (128, 96), 30, mbps=8, hardware=False)
    for level in (.2, .5, .8):
        encoder.write(np.full((96, 128, 3), level, np.float32))
    encoder.close()
    with av.open(str(path)) as container:
        frames = sum(1 for _ in container.decode(container.streams.video[0]))
    if frames != 3:
        raise RuntimeError(f'video check decoded {frames} frames')
    hardware = timelapse.Encoder(destination / 'timelapse-hardware.mp4', 'h264', (256, 144), 30, mbps=8, hardware=True)
    codec = hardware.codec
    hardware.abort()
    return dict(frames=frames, cpu='libx264', hardware=codec, ffmpeg=av.ffmpeg_version_info)


def models_load():
    """2.0.0: every bundled model opens in ONNX Runtime (a truncated file once reached a package)."""
    import onnxruntime as ort
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1] / 'assets' / 'models'
    loaded = {}
    for path in sorted(folder.glob('*.onnx')):
        if path.stat().st_size < 64 * 1024:
            raise RuntimeError(f'{path.name} is only {path.stat().st_size} bytes')
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        loaded[path.name] = [i.name for i in session.get_inputs()]
        del session
    if len(loaded) < 11:
        raise RuntimeError(f'only {len(loaded)} models bundled: {sorted(loaded)}')
    return {name: inputs for name, inputs in loaded.items()}


def library_and_culling(w, app, source, destination):
    """2.0.0: the library catalog (SQLite, ExifTool index), the preview cache, the face models of
    the culling hints, the grid and the culling view."""
    import time
    from . import culling, preview_cache
    db = w.library_db
    db.add([source])
    db.index_all(w.catalog)
    row = db.record(source)
    if row is None or not row['indexed']:
        raise RuntimeError('library index failed')
    analyser = culling.faces()
    if analyser is None:
        raise RuntimeError('face models missing')
    started = time.perf_counter()
    view = w.previews.view(source)
    result = culling.analyse(view)
    analysed = time.perf_counter() - started
    db.set_analysis(row['id'], result)
    if result['sharpness'] is None or result['faces'] is None:
        raise RuntimeError(f'culling analysis incomplete: {result}')
    w.previews.flush()
    started = time.perf_counter()
    cache = preview_cache.PreviewCache(w.previews.root, limit=1024)
    cached = cache.cached(source)
    reread = time.perf_counter() - started
    cache.close()
    if cached is None:
        raise RuntimeError('preview cache empty after opening the photo')
    w.set_mode('library')
    w.refresh_sources()
    w.refresh_library_grid()
    for _ in range(100):
        app.processEvents()
        time.sleep(.02)
    w.grab().save(str(destination / 'library.png'))
    w.select_library_path(row['path'])
    w.start_culling()
    for _ in range(100):
        app.processEvents()
        time.sleep(.02)
    if w.cull_window is None:
        raise RuntimeError('culling view did not open')
    w.cull_window.grab().save(str(destination / 'culling.png'))
    w.cull_window.close()
    w.set_mode('develop')
    app.processEvents()
    return dict(photos=db.count(), captured=row['captured'], camera=row['model'], view=list(view.shape),
                sharpness=round(result['sharpness'], 1), faces=result['faces'], analysis_ms=round(analysed * 1000),
                cached_preview_ms=round(reread * 1000), cache_bytes=w.previews.total_bytes())


def languages():
    """1.5.1: every interface catalog is bundled and complete."""
    from . import i18n
    counts = {code: len(i18n._load(code)) for code in i18n.CODES if code != i18n.SOURCE}
    if min(counts.values()) < 800:
        raise RuntimeError(f'interface catalogs incomplete: {counts}')
    from PySide6.QtCore import QLocale
    return dict(current=i18n.language(), catalogs=counts, restart=i18n.tr_in('en', '立即重新启动'),
                system=i18n.system_language(), macos=i18n.macos_languages(), qt=QLocale.system().uiLanguages())


def natural_language(w, app, settle, destination):
    """1.5.0: speech model (when downloaded, 1.8.0), Qt Multimedia, TLS and one edit through a local
    OpenAI-style server."""
    import json
    import ssl
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import numpy as np
    from PySide6.QtMultimedia import QMediaDevices
    from . import nl_edit, speech
    reply = json.dumps({'explanation': '冒烟测试', 'adjustments': {'exposure': .25, 'vibrance': 15},
                        'masks': [{'region': 'top', 'adjustments': {'highlights': -20}}]})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body = json.dumps({'choices': [{'message': {'content': reply}}]}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    # 1.8.0: packages download the speech model on first use; test it when this machine has it.
    started = time.monotonic()
    folder = speech.model_folder()
    text = speech.transcribe((np.random.default_rng(0).standard_normal(16000) * 30).astype(np.int16)) if folder else None
    recognizer = round(time.monotonic() - started, 2) if folder else None
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        w.nl_settings = nl_edit.default_settings()
        w.nl_settings.update(provider='local', timeout=20)
        w.nl_settings['providers']['local'].update(base_url=f'http://127.0.0.1:{server.server_address[1]}/v1', model='smoke')
        w.library_tabs.setCurrentIndex(1)
        w.nl_send('整体提亮一点')
        deadline = time.monotonic() + 30
        while w.nl_busy:
            app.processEvents()
            time.sleep(.01)
            if time.monotonic() > deadline:
                raise TimeoutError('natural-language request timed out')
        settle()
        if w.edits['adjustments']['exposure'] != .25 or w.edits['masks'][-1]['kind'] != 'linear':
            raise RuntimeError('natural-language edit not applied: ' + w.nl_status.text())
        w.grab().save(str(destination / 'natural-language.png'))
    finally:
        server.shutdown()
        server.server_close()
    return dict(speech_model=str(folder / speech.MODEL_NAME) if folder else None, speech_seconds=recognizer,
                speech_text=text, speech_download_mb=speech.download_megabytes(),
                microphones=[d.description() for d in QMediaDevices.audioInputs()],
                tls=ssl.OPENSSL_VERSION, status=w.nl_status.text())
