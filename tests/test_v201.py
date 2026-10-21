"""2.0.1: library fixes from the code review — capture data that could not be read is tried again
(and then reported) instead of being marked read, a rescan reads changed photos and sidecars
again, libraries made by 2.0.0 are repaired, and XMP sidecars are written by one queue so
ratings and keywords never overwrite each other."""
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from lumen import catalog, library_db
from test_v200 import photo

GOOD = {'Make': 'TESTCAM', 'Model': 'Body 1', 'ImageSize': '360 240', 'DateTimeOriginal': '2026:10:05 08:30:12'}


@pytest.fixture
def db(tmp_path):
    database = library_db.LibraryDB(tmp_path / 'library.db')
    yield database
    database.close()


def exiftool_or_skip():
    try:
        catalog._exiftool()
    except RuntimeError:
        pytest.skip('ExifTool not installed')


def fake_tags(answers):
    """read_tags that answers per file name: a dict of tags, None (no record) or an exception."""
    def read(paths):
        result = {}
        for path in paths:
            answer = answers.get(Path(path).name, GOOD)
            if isinstance(answer, Exception):
                raise answer
            if answer is not None:
                result[str(Path(path).resolve())] = dict(answer)
        return result
    return read


# ----------------------------------------------------------------------------- reading capture data

def test_a_failed_read_waits_is_tried_again_and_then_reported(tmp_path, db, monkeypatch):
    path = photo(tmp_path / 'a.jpg')
    db.add([path])
    monkeypatch.setattr(library_db, 'read_tags', fake_tags({'a.jpg': RuntimeError('ExifTool timed out')}))
    assert db.index(db.unindexed()) == []
    row = db.record(path)
    assert row['indexed'] == 0 and row['index_attempts'] == 1 and 'timed out' in row['index_error']
    assert row['index_after'] > time.time() + library_db.INDEX_BACKOFF * .9
    assert db.unindexed() == []                                   # waits instead of failing in a loop
    assert db.index_all() == 0                                    # and the command line does not spin
    later = time.time() + 10 * library_db.INDEX_BACKOFF
    for attempt in range(2, library_db.INDEX_RETRIES + 1):
        assert [r['path'] for r in db.unindexed(now=later * attempt)] == [path]
        db.index(db.unindexed(now=later * attempt))
    row = db.record(path)
    assert row['indexed'] == -1 and row['index_attempts'] == library_db.INDEX_RETRIES
    assert db.unindexed(now=later * 100) == []                    # given up until asked again
    db.reindex([row['id']])
    monkeypatch.setattr(library_db, 'read_tags', fake_tags({}))
    assert db.index_all() == 1
    row = db.record(path)
    assert row['indexed'] == 1 and row['model'] == 'Body 1' and row['index_error'] is None and row['index_attempts'] == 0


def test_one_unreadable_file_does_not_fail_the_others(tmp_path, db, monkeypatch):
    good, empty, broken = (photo(tmp_path / f'{n}.jpg', seed=i) for i, n in enumerate(('good', 'empty', 'broken')))
    db.add([good, empty, broken])
    monkeypatch.setattr(library_db, 'read_tags', fake_tags({'empty.jpg': None, 'broken.jpg': {'Error': 'File format error'}}))
    assert db.index_all() == 1
    assert db.record(good)['indexed'] == 1
    assert db.record(empty)['indexed'] == 0 and 'no data' in db.record(empty)['index_error']
    assert db.record(broken)['index_error'] == 'File format error'
    # a photo with little metadata but a size is fine
    assert library_db._read_error({'ImageSize': '1 1'}) is None


def test_exiftool_errors_of_a_damaged_file_are_recorded(tmp_path, db):
    exiftool_or_skip()
    good = photo(tmp_path / 'good.jpg', when='2026:10:05 08:30:12')
    damaged = tmp_path / 'damaged.ARW'
    damaged.write_bytes(os.urandom(5000))
    db.add([good, str(damaged)])
    assert db.index_all() == 1
    row = db.record(str(damaged))
    assert row['indexed'] == 0 and row['index_attempts'] == 1 and row['index_error']
    assert db.record(good)['indexed'] == 1


def test_a_photo_missing_while_indexed_is_read_when_it_is_back(tmp_path, db, monkeypatch):
    monkeypatch.setattr(library_db, 'read_tags', fake_tags({}))
    path = photo(tmp_path / 'card' / 'a.jpg')
    db.add([path])
    away = tmp_path / 'a.jpg'
    os.replace(path, away)
    db.index_all()
    row = db.record(path)
    assert row['missing'] == 1 and row['indexed'] == 0            # 2.0.0 marked it read, with nothing
    os.replace(away, path)
    db.add([path])
    assert db.record(path)['missing'] == 0
    assert db.index_all() == 1 and db.record(path)['model'] == 'Body 1'


# ----------------------------------------------------------------------------- rescanning a folder

def test_a_rescan_reads_changed_photos_and_sidecars_again(tmp_path, db, monkeypatch):
    monkeypatch.setattr(library_db, 'read_tags', fake_tags({}))
    folder = tmp_path / 'trip'
    a, b, c, d = (photo(folder / f'{n}.jpg', seed=i) for i, n in enumerate('abcd'))
    db.import_folder(str(folder))
    db.index_all()
    for path in (a, b, c):
        db.set_analysis(db.id_of(path), dict(sharpness=50., faces=0, eyes_closed=0))
    assert all(db.record(p)['indexed'] == 1 for p in (a, b, c, d))
    # a: edited by another program; b: its XMP sidecar appears (Lightroom keywords); c: untouched;
    # d: could not be read before
    photo(a, seed=9, size=(400, 260))
    stamp = time.time() + 5
    os.utime(a, (stamp, stamp))
    Path(b).with_suffix('.xmp').write_text('<x:xmpmeta xmlns:x="adobe:ns:meta/"/>', encoding='utf-8')
    with db._lock:
        db._db().execute('UPDATE photos SET indexed=-1, index_attempts=3, index_error=? WHERE id=?', ('x', db.id_of(d)))
    folder_id = next(f['id'] for f in db.folders() if f['root'])
    events = []
    db.listeners.append(lambda kind, ids: events.append((kind, sorted(ids))))
    added, missing, refreshed = db.rescan(folder_id)
    ra, rb, rc, rd = (db.record(p) for p in (a, b, c, d))
    assert added == [] and missing == [] and sorted(refreshed) == sorted([ra['id'], rb['id'], rd['id']])
    assert ra['indexed'] == 0 and ra['analysis'] == 0 and ra['size'] == os.path.getsize(a)
    assert rb['indexed'] == 0 and rb['analysis'] == library_db.ANALYSIS     # only the sidecar changed
    assert rc['indexed'] == 1 and rc['analysis'] == library_db.ANALYSIS
    assert rd['indexed'] == 0 and rd['index_attempts'] == 0
    assert any(kind == 'photos' and set(ids) >= {ra['id'], rb['id'], rd['id']} for kind, ids in events)
    db.index_all()
    assert db.record(b)['sidecar_mtime'] == pytest.approx(Path(b).with_suffix('.xmp').stat().st_mtime)
    db.rescan(folder_id)
    assert db.record(b)['indexed'] == 1                           # unchanged since: not read again


# ----------------------------------------------------------------------------- libraries of 2.0.0

def test_a_library_made_by_2_0_0_is_upgraded_and_repaired(tmp_path):
    path = tmp_path / 'library.db'
    old = sqlite3.connect(path)
    old.executescript('''
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema', '1');
        CREATE TABLE folders(id INTEGER PRIMARY KEY, path TEXT NOT NULL, key TEXT NOT NULL UNIQUE, root INTEGER NOT NULL DEFAULT 0, added REAL);
        CREATE TABLE photos(id INTEGER PRIMARY KEY, path TEXT NOT NULL, key TEXT NOT NULL UNIQUE,
            folder_id INTEGER REFERENCES folders(id) ON DELETE CASCADE, name TEXT, ext TEXT, size INTEGER,
            mtime REAL, added REAL, indexed INTEGER NOT NULL DEFAULT 0, captured TEXT, captured_ts REAL,
            make TEXT, model TEXT, lens TEXT, iso REAL, aperture REAL, shutter REAL, focal REAL,
            width INTEGER, height INTEGER, orientation INTEGER, gps_lat REAL, gps_lon REAL, serial TEXT,
            analysis INTEGER NOT NULL DEFAULT 0, sharpness REAL, faces INTEGER, eyes_closed INTEGER,
            signature BLOB, face_data TEXT, missing INTEGER NOT NULL DEFAULT 0);
        INSERT INTO photos(path, key, name, indexed, make, width) VALUES ('/a.jpg', 'a', 'a.jpg', 1, 'Sony', 6000);
        INSERT INTO photos(path, key, name, indexed) VALUES ('/b.jpg', 'b', 'b.jpg', 1);
        INSERT INTO photos(path, key, name, indexed, make, missing) VALUES ('/c.jpg', 'c', 'c.jpg', 1, 'Sony', 1);
    ''')
    old.close()
    database = library_db.LibraryDB(path)
    try:
        db = database._db()
        columns = {r[1] for r in db.execute('PRAGMA table_info(photos)')}
        assert {name for name, _ in library_db.ADDED_COLUMNS} <= columns
        assert dict(db.execute('SELECT name, indexed FROM photos').fetchall()) == {'a.jpg': 1, 'b.jpg': 0, 'c.jpg': 0}
        assert db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0] == str(library_db.SCHEMA)
        db.execute("UPDATE photos SET indexed=1 WHERE name='b.jpg'")
    finally:
        database.close()
    again = library_db.LibraryDB(path)                            # repaired once, not every start
    try:
        assert again._db().execute("SELECT indexed FROM photos WHERE name='b.jpg'").fetchone()[0] == 1
    finally:
        again.close()


# ----------------------------------------------------------------------------- XMP sidecars

def test_sidecar_writes_never_overlap_and_the_newest_change_wins(tmp_path, monkeypatch):
    running, peak, batches = [0], [0], []
    lock = threading.Lock()

    def slow_write(entries):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(.05)
        batches.append([dict(e) for e in entries])
        with lock:
            running[0] -= 1
        return []
    monkeypatch.setattr(catalog, '_write_sidecars', slow_write)
    writer = catalog.SidecarWriter()
    photo_path = str(tmp_path / 'a.jpg')
    tickets = []

    def burst(i):
        tickets.append(writer.queue(ratings={photo_path: dict(rating=i % 6, flag='', label='')}))
        tickets.append(writer.queue(keywords={photo_path: [f'k{i}']}))
    threads = [threading.Thread(target=burst, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    last = writer.queue(ratings={photo_path: dict(rating=5, flag='', label='red')})
    assert writer.wait(last, 10) == [] and writer.flush(10)
    assert peak[0] == 1                                           # one ExifTool at a time
    final = [e for batch in batches for e in batch if e['photo'] == photo_path][-1]
    assert final['fields'] == dict(rating=5, flag='', label='red')
    assert final['keywords'] is not None                          # merged with the keywords queued with it
    assert sum(len(b) for b in batches) < 17                      # queued changes were merged per photo


def test_ratings_and_keywords_written_at_once_both_reach_the_sidecar(tmp_path):
    exiftool_or_skip()
    path = photo(tmp_path / 'a.jpg')
    catalog.write_xmp_later({path: dict(rating=2, flag='', label='')})
    library_db.write_keywords_xmp_later({path: ['海边', 'Sunset']})
    catalog.write_xmp_later({path: dict(rating=4, flag='', label='green')})
    assert catalog.SIDECARS.flush(60)
    tags = library_db.read_tags([path])[str(Path(path).resolve())]
    assert tags.get('Rating') == 4 and tags.get('Label') == 'Green'
    subject = tags.get('Subject')
    assert sorted(subject if isinstance(subject, list) else [subject]) == ['Sunset', '海边']
    written, errors = library_db.write_keywords_xmp({path: []})   # the waiting form still reports
    assert written == [path] and errors == []
    assert 'Subject' not in library_db.read_tags([path])[str(Path(path).resolve())]
    assert catalog.read_xmp([path])[path] == dict(rating=4, label='green')


# ----------------------------------------------------------------------------- large libraries

def test_searching_needs_no_path_lookups_and_checks_edits_only_when_asked(tmp_path, db, monkeypatch):
    ratings = catalog.Catalog(tmp_path / 'catalog.jsonl')
    paths = [photo(tmp_path / f'p{i:02d}.jpg', seed=i, size=(40, 30)) for i in range(30)]
    db.add(paths)
    ratings.set(paths[::3], rating=4)
    ratings.set(paths[1:2], flag='reject', label='red')
    calls = []
    real = catalog.key
    monkeypatch.setattr(catalog, 'key', lambda p: calls.append(p) or real(p))
    rows = db.search(ratings, 'rating>=3')
    assert sorted(r['path'] for r in rows) == sorted(paths[::3]) and calls == []      # 50 000 photos: 6.5 s -> 0.7 s
    row = db.search(ratings, 'flag:reject')[0]
    assert row['path'] == paths[1] and row['label'] == 'red'
    asked = []
    db.search(ratings, '', edited=lambda p: asked.append(p) or False)
    assert asked == []
    assert len(db.search(ratings, 'edited', edited=lambda p: asked.append(p) or True)) == 30 and len(asked) == 30
    rows = db.rows()
    library_db.with_ratings(rows, ratings)
    assert {r['path']: r['rating'] for r in rows} == {p: ratings.get(p)['rating'] for p in paths}
