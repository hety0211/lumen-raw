"""2.0.0: the photo library (SQLite catalog, folders, keywords, smart albums, grid), culling
(full-screen view, comparison, bursts, sharpness and closed-eye hints) and the preview cache."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lumen import catalog, culling, library_db, preview_cache, previews

ROOT = Path(__file__).resolve().parents[1]
FACES = (ROOT / 'assets/models/face-yunet.onnx').is_file() and (ROOT / 'assets/models/face-landmarks.onnx').is_file()


def scene(width=360, height=240, seed=0, blur=0.):
    """A textured landscape-like test image (uint8 RGB)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    sky = np.stack([.45 + .2 * yy / height, .6 + .1 * yy / height, .85 - .1 * yy / height], axis=2)
    ridge = (yy > height * .45 + 18 * np.sin(xx / 23 + seed)).astype(np.float32)[..., None]
    rocks = .3 + .25 * rng.random((height, width, 1)) + .1 * np.sin(xx / 3)[..., None] * np.cos(yy / 4)[..., None]
    rgb = sky * (1 - ridge) + np.concatenate([rocks * 1.1, rocks * .9, rocks * .7], axis=2) * ridge
    image = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if blur:
        import cv2
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return image


def photo(path, seed=0, when=None, blur=0., size=(360, 240), make='TESTCAM', model='Body 1'):
    """Write a JPEG with capture data (``when`` = 'YYYY:MM:DD HH:MM:SS.fff')."""
    image = Image.fromarray(scene(*size, seed=seed, blur=blur))
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = make, model
    if when:
        main, _, fraction = when.partition('.')
        exif.get_ifd(0x8769)[0x9003] = main
        if fraction:
            exif.get_ifd(0x8769)[0x9291] = fraction
        exif.get_ifd(0x8769)[0x8827] = 400
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92, exif=exif)
    return str(Path(path).resolve())


@pytest.fixture
def db(tmp_path):
    database = library_db.LibraryDB(tmp_path / 'library.db')
    yield database
    database.close()


@pytest.fixture
def ratings(tmp_path):
    return catalog.Catalog(tmp_path / 'catalog.jsonl')


@pytest.fixture
def cache(tmp_path):
    store = preview_cache.PreviewCache(tmp_path / 'previews', limit=512)
    yield store
    store.close()


# ----------------------------------------------------------------------------- the preview cache

def test_packed_previews_keep_float16_precision_and_their_info():
    rgb = np.random.default_rng(1).normal(.3, .4, (50, 70, 3)).astype(np.float32)
    info = dict(name='a.ARW', width=7000, develop=dict(mode='camera', curve=[[0., 0.], [np.float32(.5), .6]]),
                camera_wb_gains=[np.float64(2.1), 1., 1.4, 0.], lens_match=None)
    back, data = preview_cache.unpack(preview_cache.pack(rgb, info))
    np.testing.assert_array_equal(back, preview_cache.quantize(rgb))
    np.testing.assert_allclose(back, rgb, rtol=6e-4, atol=1e-7)
    assert data['develop']['curve'][1] == [.5, .6] and data['camera_wb_gains'][0] == 2.1 and data['lens_match'] is None


def test_a_photo_decodes_once_and_then_comes_from_the_cache(tmp_path, cache, monkeypatch):
    from lumen import engine
    path = photo(tmp_path / 'a.jpg', when='2026:10:01 10:00:00')
    calls = []
    # 2.1.2: the cache decodes with _load_image (load_image adds the XMP sidecar's position).
    original = engine._load_image
    monkeypatch.setattr(engine, '_load_image', lambda *a, **k: calls.append(a) or original(*a, **k))
    rgb, info = cache.load(path)
    cache.flush()
    assert len(calls) == 1 and rgb.dtype == np.float32
    np.testing.assert_array_equal(rgb, preview_cache.quantize(original(path)[0]))
    fresh = preview_cache.PreviewCache(cache.root, limit=512)       # another process: from disk
    again, info_again = fresh.load(path)
    assert len(calls) == 1 and fresh.hits == 1
    np.testing.assert_array_equal(again, rgb)
    assert info_again == json.loads(json.dumps(preview_cache._plain(info)))
    again[:] = 0                                                     # callers get their own copy
    assert fresh.load(path)[0].max() > 0
    fresh.close()


def test_a_changed_file_is_decoded_again(tmp_path, cache):
    path = photo(tmp_path / 'b.jpg', seed=1)
    first, _ = cache.load(path)
    cache.flush()
    time.sleep(.02)
    photo(tmp_path / 'b.jpg', seed=7)
    os.utime(path, (time.time() + 5, time.time() + 5))
    second, _ = cache.load(path)
    assert not np.array_equal(first, second) and cache.misses == 2


def test_damaged_entries_are_dropped_and_the_cache_trims_to_its_limit(tmp_path, cache):
    paths = [photo(tmp_path / f'p{i}.png', seed=i, size=(400, 300)) for i in range(4)]
    for path in paths:
        cache.load(path)
    cache.flush()
    key, _ = preview_cache.stamp(paths[0])
    cache._file(key, 'd').write_bytes(b'LPV1garbage')
    cache._memory.clear()
    assert cache.cached(paths[0]) is None and not cache._file(key, 'd').exists()
    total = cache.total_bytes()
    assert total > 0
    cache.limit = total // 2
    cache.trim()
    assert 0 < cache.total_bytes() <= total // 2
    cache.clear()
    assert cache.total_bytes() == 0


def test_a_cache_turned_off_decodes_every_time(tmp_path):
    off = preview_cache.PreviewCache(tmp_path / 'off', limit=0)
    path = photo(tmp_path / 'c.jpg')
    from lumen import engine
    rgb, _ = off.load(path)
    np.testing.assert_array_equal(rgb, engine.load_image(path)[0])     # not rounded to float16
    assert not (tmp_path / 'off' / 'd').exists()
    off.close()


def test_views_and_thumbnails_come_from_the_camera_rendering(tmp_path, cache):
    path = photo(tmp_path / 'v.tif', seed=3, size=(900, 600))
    Image.open(tmp_path / 'v.tif').save(tmp_path / 'v.png')
    view = cache.view(str(tmp_path / 'v.png'))
    assert view.dtype == np.uint8 and view.shape == (600, 900, 3)
    data = cache.thumbnail(str(tmp_path / 'v.png'))
    cache.flush()
    with Image.open(__import__('io').BytesIO(data)) as thumb:
        assert max(thumb.size) == preview_cache.THUMB_EDGE
    assert cache.has_thumbnail(str(tmp_path / 'v.png'))


def test_embedded_previews_decode_at_a_reduced_scale_and_keep_their_orientation(tmp_path):
    big = Image.fromarray(scene(1600, 1000, seed=2))
    buffer = __import__('io').BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6                                   # rotated 90° clockwise
    big.save(buffer, 'JPEG', exif=exif)
    view = previews.jpeg_view(buffer.getvalue(), 400)
    assert view.shape == (400, 250, 3)
    plain = __import__('io').BytesIO()
    big.save(plain, 'JPEG')
    assert previews.jpeg_view(plain.getvalue(), 400, flip=6).shape == (400, 250, 3)
    assert previews.jpeg_view(plain.getvalue(), None).shape == (1000, 1600, 3)


# ----------------------------------------------------------------------------- the catalog

def test_importing_a_folder_scans_photos_and_skips_system_folders(tmp_path, db):
    trip = tmp_path / '旅行'
    photo(trip / 'day1' / 'a.jpg')
    photo(trip / 'day1' / 'b.jpg', seed=1)
    photo(trip / 'day2' / 'c.jpg', seed=2)
    photo(trip / '.hidden' / 'x.jpg')
    photo(trip / '$RECYCLE.BIN' / 'y.jpg')
    (trip / 'notes.txt').write_text('x', encoding='utf-8')
    added, ids = db.import_folder(str(trip))
    assert len(added) == 3 == db.count()
    assert db.import_folder(str(trip))[0] == []                            # nothing twice
    folders = {Path(f['path']).name: f for f in db.folders()}
    assert folders['旅行']['root'] == 1 and folders['day1']['count'] == 2
    whole = db.rows(('folder', folders['旅行']['id'], True))
    assert sorted(r['name'] for r in whole) == ['a.jpg', 'b.jpg', 'c.jpg']
    assert [r['name'] for r in db.rows(('folder', folders['day2']['id']))] == ['c.jpg']
    assert len(db.rows(('recent',))) == 3


def test_capture_data_keywords_and_xmp_ratings_are_indexed(tmp_path, db, ratings):
    path = photo(tmp_path / 'shot.jpg', when='2026:10:05 08:30:12.25', model='Body 7')
    Path(path).with_suffix('.xmp').write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmp:Rating="4" xmp:Label="Green"><dc:subject><rdf:Bag><rdf:li>海边</rdf:li><rdf:li>Sunset</rdf:li></rdf:Bag>'
        '</dc:subject></rdf:Description></rdf:RDF></x:xmpmeta>', encoding='utf-8')
    db.add([path])
    try:
        catalog._exiftool()
    except RuntimeError:
        pytest.skip('ExifTool not installed')
    assert db.index_all(ratings) == 1
    row = db.record(path)
    assert row['captured'] == '2026-10-05 08:30:12.25' and row['model'] == 'Body 7' and row['iso'] == 400
    import datetime
    moment = datetime.datetime(2026, 10, 5, 8, 30, 12, tzinfo=datetime.timezone.utc).timestamp()
    assert row['captured_ts'] == pytest.approx(moment + .25)
    assert sorted(row['keywords']) == ['Sunset', '海边']
    assert ratings.get(path) == dict(rating=4, flag='', label='green')


def test_keywords_smart_albums_and_queries(tmp_path, db, ratings):
    a, b, c = (photo(tmp_path / f'{n}.jpg', seed=i) for i, n in enumerate('abc'))
    db.add([a, b, c])
    ids = {Path(p).name: db.id_of(p) for p in (a, b, c)}
    db.add_keywords([ids['a.jpg'], ids['b.jpg']], '海边, Sunset，海边')
    db.add_keywords([ids['c.jpg']], ['New York'])
    assert {k['name']: k['count'] for k in db.keywords()} == {'New York': 1, 'Sunset': 2, '海边': 2}
    db.remove_keywords([ids['b.jpg']], ['sunset'])
    assert db.record(b)['keywords'] == ['海边']
    ratings.set([a], rating=4)
    ratings.set([b], flag='reject')
    names = lambda rows: sorted(r['name'] for r in rows)
    assert names(db.search(ratings, 'keyword:海边')) == ['a.jpg', 'b.jpg']
    assert names(db.search(ratings, 'keyword:"new york"')) == ['c.jpg']
    assert names(db.search(ratings, '海边 -flag:reject')) == ['a.jpg']          # a bare word finds keywords
    assert names(db.search(ratings, 'keyword:none')) == []
    assert names(db.search(ratings, 'rating>=3 keyword:sun*')) == ['a.jpg']
    collection = db.add_collection('好照片', 'rating>=3')
    assert db.collection(collection)['query'] == 'rating>=3'
    with pytest.raises(ValueError):
        db.add_collection('bad', 'rating>=x')
    db.update_collection(collection, name='精选', query='-flag:reject')
    assert [c['name'] for c in db.collections()] == ['精选']
    db.remove_collection(collection)
    assert db.collections() == []
    db.set_keywords([ids['c.jpg']], ['东京'])
    assert 'New York' not in {k['name'] for k in db.keywords()}
    db.remove([ids['c.jpg']])
    assert db.count() == 2 and db.id_of(c) is None


def test_query_fields_of_the_catalog():
    record = dict(keywords=['海边', 'Sunset'], make='SONY', model='ILCE-7RM5', lens='FE 24-70mm F2.8 GM II', iso=800.,
                  aperture=2.8, focal=35., captured='2026-10-05 08:30:12', faces=2, eyes_closed=1, sharpness=41.,
                  blurry=False, burst=3, best=True, missing=0)
    meta = catalog.empty()

    def match(query):
        return catalog.matches(catalog.parse_query(query), '/trip/DSC01.ARW', meta, False, record)
    for query in ('keyword:sunset', 'camera:7rm5', 'lens:"24-70"', 'iso>=800', 'f<=2.8', 'focal:35', 'date:2026-10',
                  'date>=2026-10-01', 'year:2026', 'faces>=2', 'eyes:closed', 'closed', 'sharpness<50', 'burst',
                  'burst:best', 'best', 'faces', '海边', 'sony', '-blurry', 'sharp', '-missing'):
        assert match(query), query
    for query in ('keyword:none', 'camera:nikon', 'iso>800', 'date<2026', 'eyes:open', 'blurry', 'missing', 'f>4'):
        assert not match(query), query
    # Without a catalog record (the 选片集 of 1.x), catalog fields never match.
    assert not catalog.matches(catalog.parse_query('keyword:any'), '/a.jpg', meta)
    assert catalog.matches(catalog.parse_query('-keyword:any'), '/a.jpg', meta)
    with pytest.raises(ValueError):
        catalog.parse_query('iso>=many')


def test_frames_of_a_burst_are_grouped_and_the_best_is_marked():
    def row(name, seconds, sharpness=60., closed=0, model='Z 8', serial='1', signature=None):
        return dict(name=name, path='/' + name, captured_ts=1000 + seconds, make='NIKON', model=model, serial=serial,
                    sharpness=sharpness, eyes_closed=closed, faces=1, signature=signature)
    rows = [row('a1', 0, 70), row('a2', .3, 80, closed=1), row('a3', .6, 75), row('a4', 1.4, 20),
            row('b1', 10), row('b2', 10.5, model='Z 6'),                     # another camera
            row('c1', 30), row('c1', 30), row('c1-Lumen', 30),               # a copy and an export
            row('d1', 40), row('d1', 40.3)]                                  # same name, different moments: a burst
    groups = library_db.annotate_bursts(rows)
    library_db.annotate_hints(rows)
    by = {}
    for r in rows:
        by.setdefault(r['name'], r)
    assert [len(g) for g in groups] == [4, 2]
    assert [by[n]['burst_index'] for n in ('a1', 'a2', 'a3', 'a4')] == [1, 2, 3, 4]
    assert by['a3']['best'] and not by['a2']['best']                          # eyes open beats sharper
    assert by['a4']['blurry'] and not by['a1']['blurry']                      # far softer than the best
    assert by['b1']['burst'] == 0 and by['c1']['burst'] == 0 and by['c1-Lumen']['burst'] == 0


def test_a_different_scene_ends_a_burst():
    a = culling.signature(scene(seed=1)[..., 0])
    a2 = culling.signature(np.clip(scene(seed=1)[..., 0].astype(int) + 30, 0, 255).astype(np.uint8))   # brighter
    b = culling.signature(np.rot90(scene(seed=5)[..., 0], 2).copy())
    assert library_db.signature_difference(a, a2) < library_db.BURST_DIFFERENCE < library_db.signature_difference(a, b)


# ----------------------------------------------------------------------------- culling hints

def test_sharpness_falls_with_blur_and_ignores_contrast():
    sharp = culling.sharpness(scene(1600, 1000, seed=4))
    soft = culling.sharpness(scene(1600, 1000, seed=4, blur=1.2))
    blurred = culling.sharpness(scene(1600, 1000, seed=4, blur=5))
    assert sharp > soft > blurred and blurred < library_db.BLURRY < sharp
    faint = (scene(1600, 1000, seed=4).astype(np.float32) * .25 + 90).astype(np.uint8)   # misty, low contrast
    assert abs(culling.sharpness(faint) - sharp) < 6
    import cv2
    moved = cv2.filter2D(scene(1600, 1000, seed=4), -1, np.full((1, 15), 1 / 15, np.float32))  # motion blur
    assert culling.sharpness(moved) < sharp - 15


def test_eye_opening_from_landmarks():
    points = np.zeros((478, 2), np.float32)
    def eye(indices, width, opening):
        outer, inner, *pairs = indices
        points[outer], points[inner] = (0, 0), (width, 0)
        for k, (upper, lower) in enumerate(pairs):
            x = width * (k + 1) / 4
            points[upper], points[lower] = (x, -opening / 2), (x, opening / 2)
    eye(culling.RIGHT_EYE, 30, 9)
    eye(culling.LEFT_EYE, 30, 2)
    assert culling.Faces.opening(points, culling.RIGHT_EYE) == pytest.approx(.3)
    assert culling.Faces.opening(points, culling.LEFT_EYE) < culling.CLOSED


@pytest.mark.skipif(not FACES, reason='face models not installed')
def test_the_face_models_load_and_find_no_face_in_a_landscape():
    result = culling.analyse(scene(1280, 853, seed=6))
    assert result['faces'] == 0 and result['eyes_closed'] == 0 and result['boxes'] == []
    assert len(result['signature']) == 256 and result['sharpness'] > library_db.BLURRY


# ----------------------------------------------------------------------------- commands

def test_catalog_commands_import_search_tag_and_analyse(tmp_path, monkeypatch):
    from lumen import commands, session
    monkeypatch.setenv('LUMEN_LIBRARY', str(tmp_path / 'cmd-library.db'))
    trip = tmp_path / 'trip'
    photo(trip / 'p1.jpg', when='2026:10:05 08:30:12.10')
    photo(trip / 'p2.jpg', when='2026:10:05 08:30:12.40', blur=4)
    photo(trip / 'p3.jpg', seed=9, when='2026:10:05 09:00:00')
    context = session.Session(catalog.Catalog(tmp_path / 'cmd.jsonl'))
    result = commands.run(context, 'catalog.import', dict(paths=[str(trip)]))
    assert result['added'] == 3 and result['photos'] == 3
    found = commands.run(context, 'catalog.search', dict(query='date:2026-10-05', limit=2))
    assert found['count'] == 3 and len(found['photos']) == 2
    with pytest.raises(commands.CommandError):
        commands.run(context, 'catalog.search', dict(folder=str(tmp_path / 'elsewhere')))
    p1 = str((trip / 'p1.jpg').resolve())
    tagged = commands.run(context, 'library.keyword', dict(add=['海边', 'sunset'], paths=[p1]))
    assert sorted(tagged['photos'][0]['keywords']) == ['sunset', '海边']
    assert [p['name'] for p in commands.run(context, 'catalog.search', dict(keyword='海边'))['photos']] == ['p1.jpg']
    analysed = commands.run(context, 'catalog.analyze', dict(query=''))
    photos = {p['name']: p for p in analysed['photos']}
    assert photos['p2.jpg']['sharpness'] < photos['p1.jpg']['sharpness']
    assert analysed['bursts'] and analysed['bursts'][0]['best'] == p1 and photos['p2.jpg']['blurry']
    names = {t['name'] for t in __import__('lumen.mcp', fromlist=['tools']).tools()}
    assert {'search_library', 'import_folder', 'tag_photos', 'analyze_photos'} <= names
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


def wait_until(app, predicate, timeout=30):
    start = time.monotonic()
    while not predicate():
        app.processEvents()
        time.sleep(.01)
        if time.monotonic() - start > timeout:
            raise AssertionError('timed out')


@pytest.fixture
def library_window(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from lumen import app as app_module
    monkeypatch.setenv('LUMEN_LIBRARY', str(tmp_path / 'window-library.db'))
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **k: None)
    w = app_module.MainWindow()
    w.resize(1500, 950)
    w.show()
    yield w
    if w.cull_window is not None:
        w.cull_window.close()
    w.saved_edits, w.saved_snapshots = w.edits, w.snapshots
    for document in w.documents.values():
        document['saved_edits'], document['saved_snapshots'] = document['edits'], document['snapshots']
    w.library_structure_dirty = False
    w.close()
    w.library_db.close()


def test_the_library_imports_a_folder_and_shows_it_in_the_grid(app, library_window, tmp_path):
    w = library_window
    trip = tmp_path / 'trip'
    for i in range(5):
        photo(trip / f'IMG_{i}.jpg', seed=i, when=f'2026:10:05 08:30:{10 + i:02d}.00')
    photo(trip / 'sub' / 'IMG_9.jpg', seed=9, when='2026:10:06 07:00:00')
    assert w.mode == 'develop'
    w.import_folder(str(trip))
    assert w.mode == 'library'
    wait_until(app, lambda: w.library_db.count() == 6 and len(w.library_model.rows) == 6)
    assert w.library_source[0] == 'folder' and w.source_tree.currentItem().text(0).startswith(str(trip.resolve()))
    wait_until(app, lambda: not w.library_worker.pending(), 60)
    w.refresh_library_grid()
    rows = w.library_model.rows
    assert [r['name'] for r in rows] == [f'IMG_{i}.jpg' for i in range(5)] + ['IMG_9.jpg']   # by capture time
    assert all(r['analysis'] == library_db.ANALYSIS and r['sharpness'] is not None for r in rows)
    assert rows[0]['burst'] and rows[0]['burst_size'] == 5                  # one-second steps: a burst
    wait_until(app, lambda: all(w.thumb_images.image(r['path']) is not None for r in rows))
    # Grid selection drives the information panel, ratings and keywords.
    w.grid.selectAll()
    app.processEvents()
    w.update_library_info()
    assert '6' in w.info_panel.title.text()
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    QTest.keyClick(w, Qt.Key.Key_3)
    app.processEvents()
    assert all(w.catalog.get(r['path'])['rating'] == 3 for r in rows)
    w.add_keywords_to_selection(['海边'])
    assert all(r['keywords'] == ['海边'] for r in w.library_model.rows)
    assert any(item.text(0).startswith('海边') for item in _tree_items(w.source_tree))
    w.library_search.setText('keyword:海边 rating>=3 -IMG_9')
    w.refresh_library_grid()
    assert len(w.library_model.rows) == 5
    w.library_search.setText('rating>=x')
    w.refresh_library_grid()
    assert w.library_search.property('invalid') is True and len(w.library_model.rows) == 5
    w.library_search.clear()
    w.collapse_bursts.setChecked(True)
    assert len(w.library_model.rows) == 2 and w.library_model.rows[0]['collapsed']
    w.collapse_bursts.setChecked(False)


def test_the_grid_fills_its_width_with_or_without_a_scroll_bar(app):
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QStandardItem, QStandardItemModel
    from lumen.library_view import GridView
    for count in (6, 15, 200):
        view = GridView()
        model = QStandardItemModel()
        for i in range(count):
            model.appendRow(QStandardItem(str(i)))
        view.setModel(model)
        view.fit_cells(QSize(212, 200))
        view.show()
        for width in (720, 1068, 1141, 1300):
            view.resize(width, 600)
            end = time.monotonic() + .1
            while time.monotonic() < end:
                app.processEvents()
            columns = len({view.visualRect(model.index(i, 0)).x() for i in range(count)})
            cell = view.gridSize().width()
            assert cell >= 212
            assert columns == count or view.viewport().width() - columns * cell < cell, (count, width, columns, cell)
        view.close()


def test_side_buttons_stack_when_their_labels_do_not_fit(app):
    from PySide6.QtWidgets import QBoxLayout, QPushButton
    from lumen.library_view import ButtonPair
    pair = ButtonPair([QPushButton('Importer un dossier…'), QPushButton('＋ Álbum inteligente')])
    pair.show()
    for width, direction in ((180, QBoxLayout.Direction.TopToBottom), (900, QBoxLayout.Direction.LeftToRight)):
        pair.resize(width, 40)
        app.processEvents()
        assert pair.box.direction() == direction
        assert all(b.width() >= b.sizeHint().width() for b in pair.buttons)
    pair.close()


def _tree_items(tree):
    items, stack = [], [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop()
        items.append(item)
        stack.extend(item.child(i) for i in range(item.childCount()))
    return items


def test_smart_albums_and_opening_photos_in_the_develop_view(app, library_window, tmp_path):
    w = library_window
    paths = [photo(tmp_path / 'set' / f'{n}.jpg', seed=i) for i, n in enumerate(('a', 'b', 'c'))]
    w.import_folder(str(tmp_path / 'set'))
    wait_until(app, lambda: len(w.library_model.rows) == 3)
    w.catalog.set([paths[1]], flag='pick')
    collection = w.library_db.add_collection('留用', 'flag:pick')
    w.refresh_sources()
    item = w.source_tree.find(['collection', collection])
    assert item is not None
    w.source_tree.setCurrentItem(item)
    assert w.library_source == ('collection', collection)
    assert [r['path'] for r in w.library_model.rows] == [paths[1]]
    # Opening from the grid: the photos join the 选片集 and the develop view opens the current one.
    w.source_tree.setCurrentItem(w.source_tree.find(['all']))
    w.select_library_path(paths[2])
    w.open_library_selection()
    assert w.mode == 'develop' and paths[2] in w.documents
    wait_until(app, lambda: w.source_path == paths[2] and w.rendered is not None and not w.loading)
    assert w.previews.misses + w.previews.hits >= 1
    w.set_mode('library')
    assert w.library_current_path() == paths[2]
    album = w.source_tree.find(['album'])
    w.source_tree.setCurrentItem(album)
    assert [r['path'] for r in w.library_model.rows] == [paths[2]]


def test_culling_moves_rates_compares_and_opens(app, library_window, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w = library_window
    paths = [photo(tmp_path / 'cull' / f'F{i}.jpg', seed=i, when=f'2026:10:07 12:00:0{i}.{i}0', blur=3 if i == 2 else 0)
             for i in range(4)]
    w.import_folder(str(tmp_path / 'cull'))
    wait_until(app, lambda: len(w.library_model.rows) == 4)
    wait_until(app, lambda: not w.library_worker.pending(), 60)
    w.refresh_library_grid()
    w.select_library_path(paths[0])
    w.start_culling()
    cull = w.cull_window
    assert cull is not None and cull.current_row()['path'] == paths[0]
    wait_until(app, lambda: cull.loader.image(paths[0]) is not None)
    QTest.keyClick(cull, Qt.Key.Key_Right)
    assert cull.current_row()['path'] == paths[1]
    QTest.keyClick(cull, Qt.Key.Key_P)
    assert w.catalog.get(paths[1])['flag'] == 'pick'
    QTest.keyClick(cull, Qt.Key.Key_A)                     # advance after rating
    QTest.keyClick(cull, Qt.Key.Key_X)
    assert w.catalog.get(paths[1])['flag'] == 'reject' and cull.current_row()['path'] == paths[2]
    QTest.keyClick(cull, Qt.Key.Key_A)
    QTest.keyClick(cull, Qt.Key.Key_Left)
    QTest.keyClick(cull, Qt.Key.Key_U)
    assert w.catalog.get(paths[1])['flag'] == '' and cull.current_row()['path'] == paths[1]
    QTest.keyClick(cull, Qt.Key.Key_Right)
    row = cull.rows[paths[2]]
    assert row['burst'] and row['blurry'] and '模糊' in __import__('lumen.library_view', fromlist=['hint_text']).hint_text(row)
    QTest.keyClick(cull, Qt.Key.Key_Left)
    QTest.keyClick(cull, Qt.Key.Key_C)                     # compare: the select and a candidate
    assert cull.compare and cull.right.isVisible()
    select, candidate = cull.index, cull.candidate
    assert (select, candidate) == (1, 2)
    QTest.keyClick(cull, Qt.Key.Key_Right)
    assert cull.index == select and cull.candidate == candidate + 1
    QTest.keyClick(cull, Qt.Key.Key_5)
    assert w.catalog.get(cull.paths[cull.candidate])['rating'] == 5
    QTest.keyClick(cull, Qt.Key.Key_Tab)
    QTest.keyClick(cull, Qt.Key.Key_1)
    assert w.catalog.get(cull.paths[select])['rating'] == 1
    QTest.keyClick(cull, Qt.Key.Key_Z)
    assert cull.left.zoomed and cull.right.zoomed           # zoom is shared
    wait_until(app, lambda: cull.loader.image(cull.paths[select], True) is not None)
    QTest.keyClick(cull, Qt.Key.Key_C)
    assert not cull.compare
    target = cull.current_row()['path']
    QTest.keyClick(cull, Qt.Key.Key_Return)
    wait_until(app, lambda: w.cull_window is None and w.source_path == target and not w.loading)
    assert w.mode == 'develop'


def test_the_filmstrip_uses_cached_thumbnails_and_prefetches_neighbours(app, library_window, tmp_path):
    w = library_window
    paths = [photo(tmp_path / 'strip' / f'S{i}.jpg', seed=i) for i in range(3)]
    w.import_paths(paths)
    wait_until(app, lambda: w.source_path == paths[0] and w.rendered is not None and not w.loading)
    wait_until(app, lambda: all(not w.film_item(p).icon().isNull() for p in paths))
    assert all(w.previews.has_thumbnail(p) for p in paths)
    assert w.library_db.count() == 3                         # the 选片集 joins the library
    w.previews._prefetch.submit(lambda: None).result()
    keys = [preview_cache.stamp(p)[0] for p in paths[1:]]
    wait_until(app, lambda: all(k in w.previews._memory for k in keys))


# ----------------------------------------------------------------------------- the speech model source

def test_the_speech_model_is_also_fetched_from_the_own_release(monkeypatch):
    from lumen import speech
    url = speech.UPSTREAM + 'org/repo/resolve/abc/model.int8.onnx'
    release = 'https://github.com/owner/repo/releases/download/speech-model-1/'
    monkeypatch.setattr(speech, 'language', lambda: 'en')
    assert speech.sources(url, ['https://hf-mirror.com/'], [release]) == [
        url, release + 'model.int8.onnx', 'https://hf-mirror.com/org/repo/resolve/abc/model.int8.onnx']
    monkeypatch.setattr(speech, 'language', lambda: 'zh_CN')
    assert speech.sources(url, ['https://hf-mirror.com/'], [release])[1:] == [release + 'model.int8.onnx', url]


def test_the_face_models_are_bundled_and_the_culling_manifest_is_not():
    from tools import bundle, compact_models
    data = json.loads((ROOT / 'assets/culling-assets.json').read_text(encoding='utf-8'))
    assert {Path(f['path']).name for f in data['files']} == {'face-yunet.onnx', 'face-landmarks.onnx'}
    assert sum(f['bytes'] for f in data['files']) < 6 * 2 ** 20
    assert 'culling-assets.json' in bundle.SKIP_ASSETS and 'face-yunet.onnx' in compact_models.KEEP_FLOAT32
    assert (ROOT / 'assets/YuNet-LICENSE.txt').is_file() and (ROOT / 'assets/MediaPipe-LICENSE.txt').is_file()


def test_a_damaged_library_or_cache_never_keeps_the_app_from_working(tmp_path):
    damaged = tmp_path / 'library.db'
    damaged.write_bytes(b'this is not a database' * 100)
    db = library_db.LibraryDB(damaged)
    assert db.count() == 0 and list(tmp_path.glob('library.damaged-*.db'))
    db.close()
    folder = tmp_path / 'cache'
    folder.mkdir()
    (folder / 'index.db').write_bytes(b'garbage' * 200)
    cache = preview_cache.PreviewCache(folder, limit=64)
    path = photo(tmp_path / 'ok.jpg')
    rgb, _ = cache.load(path)
    assert rgb.shape[2] == 3 and cache.total_bytes() >= 0
    cache.close()
    blocked = tmp_path / 'blocked'
    blocked.write_text('a file where the cache folder should be', encoding='utf-8')
    unusable = preview_cache.PreviewCache(blocked / 'previews', limit=64)
    rgb, _ = unusable.load(path)                     # falls back to decoding
    assert rgb.shape[2] == 3 and not unusable.has_thumbnail(path)
    assert unusable.thumbnail(path)                  # made, just not kept
    unusable.close()


def test_a_compact_model_changed_after_it_was_written_is_written_again(tmp_path):
    from tools import compact_models
    from lumen import onnx_graph
    graph = onnx_graph.Graph('m')
    x = graph.input('x', [1, 64])
    graph.output(graph.op('MatMul', x, graph.const(np.ones((64, 32), np.float32))), 'y', [1, 32])
    source, target = tmp_path / 'models', tmp_path / 'compact'
    source.mkdir()
    (source / 'net.onnx').write_bytes(graph.serialize())
    logged = []
    compact_models.compact_folder(source, target, logged.append)
    good = (target / 'net.onnx').read_bytes()
    (target / 'net.onnx').write_bytes(b'x')            # e.g. a test wrote into the build folder
    compact_models.compact_folder(source, target, logged.append)
    assert len(logged) == 2 and (target / 'net.onnx').read_bytes() == good
