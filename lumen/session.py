"""Headless editing session for the command line and the MCP server (1.6.0).

It holds what the window holds — a library of photos, the current photo, a recipe and an undo
history per photo — without any interface.  Photos open as in the editor: camera white balance,
camera develop reference and lens profile are set the first time, and a ``.lumen`` project with
the photo's name next to it is picked up, so work saved by the editor (or by an earlier session)
continues.  Nothing is written unless a command saves a project, exports or rates a photo.
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

from . import catalog, commands, engine, lens, model

log = logging.getLogger(__name__)


class Session(commands.Context):
    mode = 'headless'

    def __init__(self, library=None, backend=None):
        self.catalog = library or catalog.Catalog()
        self.documents = {}
        self.order = []
        self._current = None
        self._selected = []
        self._backend = backend
        self._lock = threading.RLock()

    # -- library

    def photos(self):
        return list(self.order)

    def current(self):
        return self._current

    def selected(self):
        return list(self._selected)

    def select(self, paths):
        self._selected = [p for p in paths if p in self.documents]

    def add(self, paths):
        added = []
        with self._lock:
            for path in paths:
                path = str(Path(path).resolve())
                if path not in self.documents:
                    self.documents[path] = dict(edits=model.recipe(), history=None, source=None, info=None,
                                                project='', initialized=False)
                    self.order.append(path)
                    added.append(path)
        if added:
            try:
                self.catalog.merge(catalog.read_xmp([p for p in added if not self.catalog.known(p)]))
            except Exception:
                log.warning('XMP read failed', exc_info=True)
        return added

    # -- photos

    def open(self, path):
        path = str(Path(path).resolve())
        project = ''
        if Path(path).suffix.lower() == '.lumen':
            project = path
            path, edits, _ = model.load_project(project, include_snapshots=True)
        else:
            edits = None
            sibling = Path(path).with_suffix('.lumen')
            if sibling.is_file():
                try:
                    photo, loaded = model.load_project(sibling)
                    if Path(photo).resolve() == Path(path):
                        project, edits = str(sibling), loaded
                except Exception:
                    log.warning('ignored project %s', sibling, exc_info=True)
        self.add([path])
        document = self.documents[path]
        self._load(path)
        if project:
            document.update(edits=edits, project=project, initialized=True)
            self._prepare(path, edits, new=False)
            document['history'] = model.History(edits)
        self._current = path
        return path

    def _load(self, path):
        document = self.documents[path]
        if document['source'] is None:
            source, info = engine.load_image(path)
            document.update(source=source, info=info)
            if not document['initialized']:
                edits = document['edits']
                edits['white_balance'] = copy.deepcopy(info.get('white_balance', edits['white_balance']))
                edits['develop'] = copy.deepcopy(info.get('develop', edits['develop']))
                self._prepare(path, edits, new=True)
                document['initialized'] = True
            if document['history'] is None:
                document['history'] = model.History(document['edits'])
        return document

    def _prepare(self, path, edits, new):
        from .lens_panel import auto_enable
        if new and auto_enable():
            edits.setdefault('lens', lens.defaults())['enabled'] = True
        edits['lens'] = lens.prepare(edits.get('lens'), self.documents[path]['info'].get('lens_match'))[0]

    def edits(self, path):
        document = self.documents.get(path)
        if document is None:
            raise commands.CommandError(f'not in the library: {path}')
        return copy.deepcopy(document['edits'])

    def initialized(self, path):
        return self.documents[path]['initialized']

    def apply(self, path, edits, label):
        document = self._load(path)
        document['edits'] = copy.deepcopy(edits)
        document['history'].push(edits)

    def step(self, path, delta):
        document = self._load(path)
        document['edits'] = document['history'].move(delta)
        return copy.deepcopy(document['edits'])

    def source(self, path):
        document = self._load(path)
        return document['source'], document['info']

    def save_project(self, path, target):
        model.save_project(target, path, self.documents[path]['edits'])
        self.documents[path]['project'] = target
        return target

    def backend(self):
        if self._backend is None:
            self._backend = engine.Backend('auto')
        return self._backend
