"""2.0.0: time-lapse videos of interval shots: keyframe blending, deflicker, sizes, encoders,
the job, the dialog in the window and the command."""
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lumen import model, timelapse

av = pytest.importorskip('av')


def shot(path, level, size=(320, 240), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    rgb = np.stack([xx / size[0], yy / size[1], .5 + .2 * np.sin(xx / 9)], axis=2) * level
    rgb = rgb + rng.normal(0, .01, rgb.shape)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path, quality=95)
    return str(Path(path).resolve())


def decoded(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format='rgb24') for f in container.decode(stream)]
        return stream, frames


# ----------------------------------------------------------------------------- blending

def test_numbers_blend_hues_take_the_short_way_and_the_rest_switches():
    a = dict(adjustments=dict(exposure=0., temperature=-20.), grading=dict(shadows=[350., 10.], balance=0.),
             crop=None, rotation=0, monochrome=False, curves=dict(RGB=[[0., 0.], [1., 1.]]), name='a')
    b = dict(adjustments=dict(exposure=-2., temperature=40.), grading=dict(shadows=[10., 30.], balance=20.),
             crop=[.1, .1, .9, .9], rotation=1, monochrome=True, curves=dict(RGB=[[0., 0.], [.5, .6], [1., 1.]]), name='b')
    half = timelapse.mix(a, b, .5)
    assert half['adjustments'] == dict(exposure=-1., temperature=10.)
    assert half['grading']['shadows'] == [0., 20.] and half['grading']['balance'] == 10.
    assert half['crop'] == [.1, .1, .9, .9] and half['rotation'] == 1 and half['monochrome'] is True
    quarter = timelapse.mix(a, b, .25)
    assert quarter['crop'] is None and quarter['rotation'] == 0 and quarter['curves'] == a['curves']
    assert quarter['grading']['shadows'][0] == pytest.approx(355.)
    assert timelapse.mix(a, b, 0) == a and timelapse.mix(a, b, 1) == b


def test_frames_between_keyframes_blend_and_outside_them_hold():
    first, last = model.recipe(), model.recipe()
    first['crop'] = [0., 0., 1., 1.]
    last['adjustments']['exposure'] = -2.
    last['adjustments']['temperature'] = 40.
    last['crop'] = [.2, .1, .8, .7]
    keyframes = {2: first, 6: last}
    assert timelapse.recipe_at(0, keyframes)['adjustments']['exposure'] == 0.
    middle = timelapse.recipe_at(4, keyframes)
    assert middle['adjustments']['exposure'] == pytest.approx(-1.) and middle['adjustments']['temperature'] == pytest.approx(20.)
    assert middle['crop'] == pytest.approx([.1, .05, .9, .85])                   # a slow pan and zoom
    assert timelapse.recipe_at(9, keyframes)['adjustments']['exposure'] == -2.
    assert timelapse.recipe_at(3, {})['adjustments'] == model.recipe()['adjustments']


def test_default_keyframes_are_the_ends_and_the_edited_photos():
    frames = [dict(edited=i == 3) for i in range(7)]
    assert timelapse.default_keyframes(frames) == [0, 3, 6]
    assert timelapse.default_keyframes([]) == []


def test_capture_order_and_file_name_order():
    paths = ['/t/b.jpg', '/t/a.jpg', '/t/c.jpg']
    records = {'/t/a.jpg': dict(captured_ts=3.), '/t/b.jpg': dict(captured_ts=1.), '/t/c.jpg': dict(captured_ts=2.)}
    assert timelapse.ordered(paths, records) == ['/t/b.jpg', '/t/c.jpg', '/t/a.jpg']
    assert timelapse.ordered(paths, {'/t/a.jpg': dict(captured_ts=1.)}) == ['/t/a.jpg', '/t/b.jpg', '/t/c.jpg']


def test_deflicker_removes_flicker_and_keeps_a_sunset():
    ramp = np.linspace(0, -4, 60)                       # the light fades by 4 stops
    flicker = np.where(np.arange(60) % 2, .25, -.25)
    corrections = np.array(timelapse.deflicker(ramp + flicker, 11))
    corrected = ramp + flicker + corrections
    assert np.std(np.diff(corrected[6:-6])) < .2 * np.std(np.diff((ramp + flicker)[6:-6]))
    assert corrected[0] - corrected[-1] > 3.4
    assert timelapse.deflicker([0., 1., 0.], 0) == [0., 0., 0.]
    assert max(abs(c) for c in timelapse.deflicker([0., 9., 0., 9., 0.], 3)) <= timelapse.DEFLICKER_LIMIT


def test_automatic_white_balance_jumps_are_smoothed_and_slow_drifts_kept():
    drift = np.linspace(0, .3, 40)
    jitter = np.where(np.arange(40) % 3 == 0, .04, -.02)
    multipliers = [[2.4 * np.exp(d + j), 1., 1.6 * np.exp(-d)] for d, j in zip(drift, jitter)]
    multipliers[5] = None                                         # a JPEG in the sequence
    factors = timelapse.stabilise_white_balance(multipliers, 11)
    corrected = [np.log(m[0] * f[0]) for m, f in zip(multipliers, factors) if m]
    assert factors[5] == [1., 1., 1.] and all(f[1] == pytest.approx(1.) for f in factors)
    assert np.std(np.diff(corrected[6:-6])) < .25 * np.std(np.diff(np.log([m[0] for m in multipliers if m]))[6:-6])
    assert corrected[-1] - corrected[0] > .25                     # the drift stays
    assert timelapse.stabilise_white_balance([[2., 1., 1.5]] * 4, 0) == [[1., 1., 1.]] * 4


@pytest.mark.parametrize('frame, resolution, aspect, fmt, expected', [
    ((6000, 4000), '4k', '16:9', 'h264', (3840, 2160)),
    ((6000, 4000), '4k', 'original', 'h264', (3840, 2560)),
    ((6000, 4000), 'fhd', '9:16', 'hevc', (1080, 1920)),
    ((6000, 4000), '2k', '1:1', 'av1', (2560, 2560)),
    ((6001, 4001), 'original', 'original', 'prores', (6000, 4000)),
])
def test_output_sizes(frame, resolution, aspect, fmt, expected):
    long_edge = dict((k, v) for k, _, v in timelapse.RESOLUTIONS)[resolution]
    ratio = dict((k, v) for k, _, v in timelapse.ASPECTS)[aspect]
    assert timelapse.output_size(*frame, long_edge, ratio, fmt) == expected


def test_original_size_stays_within_the_codec_levels():
    width, height = timelapse.output_size(9504, 6336, None, None, 'h264')
    assert width * height <= timelapse.LEVEL_PIXELS and width % 2 == 0 and height % 2 == 0
    assert abs(width / height - 1.5) < .01
    assert timelapse.output_size(9504, 6336, None, None, 'prores') == (9504, 6336)


def test_default_bit_rates_follow_size_and_codec():
    assert timelapse.default_mbps('h264', 3840, 2160) == 80 and timelapse.default_mbps('hevc', 3840, 2160) == 50
    assert timelapse.default_mbps('h264', 3840, 2566) == 95 and timelapse.default_mbps('av1', 3840, 2560) == 45
    assert timelapse.default_mbps('h264', 1920, 1080) == 20 and timelapse.default_mbps('prores', 3840, 2160) is None
    assert timelapse.default_mbps('h264', 8192, 5460) == 400


# ----------------------------------------------------------------------------- encoding

@pytest.mark.parametrize('fmt, codec, pixel', [('h264', 'h264', 'yuv420p'), ('hevc', 'hevc', 'yuv420p10le'),
                                                ('av1', None, 'yuv420p10le'), ('prores', 'prores', 'yuv422p10le')])
def test_cpu_encoders_write_bt709_video(tmp_path, fmt, codec, pixel):
    path = tmp_path / ('clip' + timelapse.FORMATS[fmt]['suffix'])
    encoder = timelapse.Encoder(path, fmt, (96, 64), 24, mbps=8, ten_bit=True, hardware=False)
    assert encoder.codec == timelapse.FORMATS[fmt]['software']
    gray = np.full((64, 96, 3), .5, np.float32)
    encoder.write(gray, repeat=2)
    encoder.write(gray * .4)
    assert encoder.close() == path and not encoder.part.exists()
    stream, frames = decoded(path)
    assert len(frames) == 3 and (stream.width, stream.height) == (96, 64) and stream.format.name == pixel
    if codec:
        assert stream.codec_context.name == codec
    assert int(stream.codec_context.colorspace) == 1                  # BT.709
    assert abs(frames[0].mean() - 128) < 6 and frames[2].mean() < frames[0].mean() - 50


def test_graphics_card_encoders_are_tried_first_and_the_cpu_takes_over(tmp_path, monkeypatch):
    tried = []
    original = timelapse.Encoder._probe

    def probe(self, codec):
        tried.append(codec)
        if codec != 'libx264':
            raise RuntimeError('no such device')
        return original(self, codec)
    monkeypatch.setattr(timelapse.Encoder, '_probe', probe)
    encoder = timelapse.Encoder(tmp_path / 'v.mp4', 'h264', (64, 64), 30, hardware=True)
    encoder.abort()
    assert tried == timelapse.platform_hardware(timelapse.FORMATS['h264']['hardware']) + ['libx264'] and not list(tmp_path.iterdir())
    monkeypatch.setattr(timelapse.Encoder, '_probe', lambda self, codec: (_ for _ in ()).throw(RuntimeError('broken')))
    with pytest.raises(timelapse.EncoderError):
        timelapse.Encoder(tmp_path / 'w.mp4', 'h264', (64, 64), 30)


# ----------------------------------------------------------------------------- the job

def sequence(folder, count=6):
    levels = [.8 if i % 2 else .6 for i in range(count)]          # flickering exposure
    return [shot(folder / f'IMG_{i:04d}.jpg', level, seed=i) for i, level in enumerate(levels)]


def test_a_time_lapse_blends_keyframes_and_deflickers(tmp_path):
    paths = sequence(tmp_path / 'seq')
    dark = model.recipe()
    dark['adjustments']['exposure'] = -2.
    frames = [dict(path=p, edits=None, initialized=False) for p in paths]
    frames[-1]['edits'] = dark
    output = tmp_path / 'out.mp4'
    reports = []
    result = timelapse.run(frames, dict(output=str(output), format='h264', resolution='original', aspect='original',
                                        fps=25, repeat=2, deflicker='off', hardware=False),
                           None, lambda done, total, text: reports.append((done, total)))
    stream, video = decoded(output)
    assert result['frames'] == 12 == len(video) and result['keyframes'] == [0, 5] and result['codec'] == 'libx264'
    assert (stream.width, stream.height) == (320, 240) and result['seconds'] == pytest.approx(12 / 25)
    means = [f.mean() for f in video[::2]]
    assert means[0] > means[2] > means[4] and means[1] > means[3] > means[5]     # darker towards the last keyframe
    assert reports[-1] == (6, 6)
    steady = timelapse.run([dict(path=p, edits=None, initialized=False) for p in paths],
                           dict(output=str(tmp_path / 'steady.mp4'), format='h264', resolution='original', aspect='original',
                                fps=25, deflicker='strong', hardware=False), None)
    flat = [f.mean() for f in decoded(tmp_path / 'steady.mp4')[1]]
    assert steady['frames'] == 6 and np.std(np.diff(flat)) < 6


def test_a_cancelled_time_lapse_leaves_no_file(tmp_path):
    paths = sequence(tmp_path / 'seq', 4)
    cancel = threading.Event()

    def progress(done, total, text):
        if done >= 2:
            cancel.set()
    with pytest.raises(InterruptedError):
        timelapse.run([dict(path=p) for p in paths], dict(output=str(tmp_path / 'gone.mp4'), resolution='original',
                                                          hardware=False), None, progress, cancel)
    assert not list(tmp_path.glob('gone*'))


def test_the_command_makes_a_time_lapse(tmp_path, monkeypatch):
    from lumen import catalog, commands, session
    monkeypatch.setenv('LUMEN_LIBRARY', str(tmp_path / 'tl-library.db'))
    paths = sequence(tmp_path / 'cmd', 3)
    context = session.Session(catalog.Catalog(tmp_path / 'tl.jsonl'))
    result = commands.run(context, 'video.timelapse', dict(output=str(tmp_path / 'cmd.mov'), paths=paths, format='prores',
                                                           resolution='original', fps=24, keyframes=[paths[1]], hardware=False))
    assert result['frames'] == 3 and result['keyframes'] == [paths[1]] and result['bytes'] > 0
    with pytest.raises(commands.CommandError):
        commands.run(context, 'video.timelapse', dict(output=str(tmp_path / 'x.mp4'), paths=paths[:1]))
    with pytest.raises(commands.CommandError):
        commands.run(context, 'video.timelapse', dict(output=str(tmp_path / 'x.mov'), paths=paths, format='h264'))
    from lumen import mcp
    assert 'make_timelapse' in {t['name'] for t in mcp.tools()}
    context.library_db.close()


# ----------------------------------------------------------------------------- the window

@pytest.fixture(scope='module')
def app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from lumen.app import STYLE
    instance = QApplication.instance() or QApplication([])
    instance.setStyle('Fusion')
    instance.setStyleSheet(STYLE)
    return instance


def wait_until(app, predicate, timeout=60):
    start = time.monotonic()
    while not predicate():
        app.processEvents()
        time.sleep(.01)
        if time.monotonic() - start > timeout:
            raise AssertionError('timed out')


def test_the_window_makes_a_time_lapse_of_the_selection(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog, QMessageBox
    from lumen.app import MainWindow
    from lumen import timelapse_dialog
    monkeypatch.setenv('LUMEN_LIBRARY', str(tmp_path / 'tlw-library.db'))
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'appdata'))
    shown = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: shown.append(a[2]))
    paths = sequence(tmp_path / 'win', 4)
    output = tmp_path / 'window.mp4'
    dialogs = []

    def accept(dialog):
        dialogs.append(dialog)
        assert dialog.list.count() == 4 and dialog.marks() == [0, 3]
        dialog.toggle(dialog.list.item(1))                   # a third keyframe
        dialog.resolution.setCurrentIndex(dialog.resolution.findData('original'))
        dialog.aspect.setCurrentIndex(dialog.aspect.findData('1:1'))
        dialog.hardware.setChecked(False)
        assert '240 × 240' in dialog.summary.text()
        dialog.output.setText(str(output))
        dialog.confirm()
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(timelapse_dialog.TimelapseDialog, 'exec', accept)
    w = MainWindow()
    try:
        w.show()
        w.add_documents(paths)
        w.filmstrip.selectAll()
        menu = w.film_context_menu()
        action = next(a for a in menu.actions() if a.text() == '制作延时视频…')
        assert action.isEnabled()
        action.trigger()
        wait_until(app, lambda: shown or not w.exporting and output.exists())
        wait_until(app, lambda: not w.exporting)
        assert output.exists() and shown and '240 × 240' in shown[0]
        assert w.timelapse_marks[tuple(paths)] == [0, 1, 3]
        stream, frames = decoded(output)
        assert len(frames) == 4 and stream.width == 240
        # The library grid starts one too: its photos join the 选片集 first.
        assert hasattr(w, 'timelapse_from_library')
    finally:
        w.saved_edits, w.saved_snapshots = w.edits, w.snapshots
        for document in w.documents.values():
            document['saved_edits'], document['saved_snapshots'] = document['edits'], document['snapshots']
        w.library_structure_dirty = False
        w.close()
        w.library_db.close()
