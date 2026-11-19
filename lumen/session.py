"""Headless editing session for the command line and the MCP server (1.6.0).

It holds what the window holds — a library of photos, the current photo, a recipe and an undo
history per photo — without any interface.  Photos open as in the editor: camera white balance,
camera develop reference and lens profile are set the first time, and a ``.lumen`` project with
the photo's name next to it is picked up, so work saved by the editor (or by an earlier session)
continues.  Nothing is written next to the photos unless a command saves a project, exports or
rates a photo.

2.3.1 (GitHub issue #13): a session made with ``keep`` (the MCP server, ``lumen-cli run``) also
writes every edit to a working copy in the per-user data folder (``working``) and picks it up
again the first time it opens the photo, so edits an agent applied survive a reconnect.
"""
from __future__ import annotations

import copy
import logging
import threading
from pathlib import Path

from . import catalog, commands, engine, lens, model, preview_cache, working

log = logging.getLogger(__name__)


class Session(commands.Context):
    mode = 'headless'

    def __init__(self, library=None, backend=None, keep=False):
        self.keep = keep
        self.catalog = library or catalog.Catalog()
        self.documents = {}
        self.order = []
        self._current = None
        self._selected = []
        self._backend = backend
        self._lock = threading.RLock()
        self._library_db = None

    @property
    def library_db(self):
        """The photo library catalog (2.0.0), opened on first use."""
        if self._library_db is None:
            from .library_db import LibraryDB
            self._library_db = LibraryDB()
        return self._library_db

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
                                                project='', initialized=False, restored=False)
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
        explicit = Path(path).suffix.lower() == '.lumen'
        if explicit:
            project = path
            path, edits, _ = model.load_project(project, include_snapshots=True)
        else:
            edits = None
            # 2.3.1: the project next to the photo is read when the photo is opened the first time.
            # Reading it on every open put its saved state over edits made since (issue #13).
            known = self.documents.get(path)
            sibling = Path(path).with_suffix('.lumen')
            if (known is None or not known['initialized']) and sibling.is_file():
                try:
                    photo, loaded = model.load_project(sibling)
                    if Path(photo).resolve() == Path(path):
                        project, edits = str(sibling), loaded
                except Exception:
                    log.warning('ignored project %s', sibling, exc_info=True)
        self.add([path])
        document = self.documents[path]
        if project:
            document['project'] = project
        self._load(path)
        if project and (explicit or not document['restored']):
            document.update(edits=edits, project=project, initialized=True, restored=False)
            self._prepare(path, edits, new=False)
            document['history'] = model.History(edits)
            if explicit and self.keep:
                working.save(path, edits)       # the project just opened is the state to continue from
        self._current = path
        return path

    def restored(self, path):
        """True when the photo's edits came from a working copy an earlier session left (2.3.1)."""
        document = self.documents.get(path)
        return bool(document and document['restored'])

    def persistence(self):
        if self.keep:
            return ('unsaved: kept as a working copy (per-user data folder) and restored when a session without '
                    'the editor window opens this photo again, also after a reconnect; save_project writes a '
                    '.lumen project next to the photo, export_photo writes the picture')
        return 'unsaved: in this process only; project.save writes a .lumen project'

    def _keep(self, path):
        if self.keep:
            working.save(path, self.documents[path]['edits'])

    def _load(self, path):
        document = self.documents[path]
        if document['source'] is None:
            source, info = preview_cache.load(path)      # 2.0.0: shared with the editor
            document.update(source=source, info=info)
            if self.keep and not document['initialized']:
                # 2.3.1: edits an agent applied in an earlier session (or in the editor window),
                # unless a project saved next to the photo since is newer.
                kept = working.load(path, newer_than=document['project'] or None)
                if kept is not None:
                    document.update(edits=kept, initialized=True, restored=True)
                    self._prepare(path, kept, new=False)
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
        match = self.documents[path]['info'].get('lens_match')
        if new and auto_enable() and lens.auto_enables(match):
            edits.setdefault('lens', lens.defaults())['enabled'] = True
        edits['lens'] = lens.prepare(edits.get('lens'), match)[0]

    def edits(self, path):
        document = self.documents.get(path)
        if document is None:
            raise commands.CommandError(f'not in the library: {path}')
        return copy.deepcopy(document['edits'])

    def initialized(self, path):
        return self.documents[path]['initialized']

    def apply(self, path, edits, label):
        fresh = path in self.documents and not self.documents[path]['initialized']
        document = self._load(path)
        edits = copy.deepcopy(edits)
        if fresh:
            # Edits computed before the photo was first opened (a batch preset, copied edits) get its
            # own white balance, develop reference and lens profile, as the window gives them (1.7.0).
            for key in ('white_balance', 'develop'):
                edits[key] = copy.deepcopy(document['edits'][key])
            edits['lens'] = lens.prepare(edits.get('lens'), document['info'].get('lens_match'))[0]
        document['edits'] = edits
        document['history'].push(edits)
        self._keep(path)

    def step(self, path, delta):
        document = self._load(path)
        document['edits'] = document['history'].move(delta)
        self._keep(path)
        return copy.deepcopy(document['edits'])

    def source(self, path):
        document = self._load(path)
        return document['source'], document['info']

    def save_project(self, path, target):
        model.save_project(target, path, self.documents[path]['edits'])
        self.documents[path]['project'] = target
        if self.keep and Path(target).resolve() == Path(path).with_suffix('.lumen').resolve():
            working.discard(path)       # the project next to the photo now holds these edits
        return target

    def backend(self):
        if self._backend is None:
            self._backend = engine.Backend('auto')
        return self._backend
