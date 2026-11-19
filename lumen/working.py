"""Working copies of edits agents made, kept between connections (2.3.1, GitHub issue #13).

An agent's edits lived only in the process that made them: the headless session of one
``lumen-cli mcp`` run, or the editor window.  When the MCP client reconnected (agents restart
their servers freely), or the window closed in the middle of a task, ``apply_edits`` had reported
success and the next ``get_edits`` read zeros.

Every recipe an agent applies is now also written here, one small project file per photo in the
per-user data folder (``working``; ``LUMEN_WORKING`` overrides it), and a session without the
window picks it up the first time it opens that photo, unless a project saved next to the photo
is newer.  Photos, their folders and the user's own projects are never touched; saving a project
next to the photo takes the working copy's place.  Copies not used for ``KEEP_DAYS`` are removed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

KEEP_DAYS = 30
_pruned = False


def folder():
    override = os.environ.get('LUMEN_WORKING')
    if override:
        return Path(override)
    from .host import data_folder
    return data_folder() / 'working'


def _file(path):
    resolved = str(Path(path).resolve())
    return folder() / (hashlib.sha1(os.path.normcase(resolved).encode('utf-8')).hexdigest() + '.lumen')


def save(path, edits):
    """Keep ``edits`` as the working copy of photo ``path``; True when it was written."""
    from . import model
    try:
        target = _file(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save_project(target, str(Path(path).resolve()), edits)
        _prune()
        return True
    except Exception:
        # Never fails an edit: the session still holds it.
        log.warning('working copy of %s not written', path, exc_info=True)
        return False


def load(path, newer_than=None):
    """The working copy of ``path`` as a validated recipe, or None: none kept, unreadable, kept for
    another photo, or older than the file ``newer_than`` (a project saved next to the photo)."""
    from . import model
    target = _file(path)
    try:
        written = target.stat().st_mtime
    except OSError:
        return None
    if newer_than is not None:
        try:
            if Path(newer_than).stat().st_mtime >= written:
                return None
        except OSError:
            pass
    try:
        source, edits = model.load_project(target)
        if os.path.normcase(str(Path(source).resolve())) != os.path.normcase(str(Path(path).resolve())):
            return None
        os.utime(target, (time.time(), written))     # read: counts as used, keeps its date
        return edits
    except Exception:
        log.warning('working copy %s unreadable', target, exc_info=True)
        return None


def discard(path):
    try:
        _file(path).unlink(missing_ok=True)
    except OSError:
        log.debug('working copy of %s not removed', path, exc_info=True)


def _prune():
    """Once per process: remove copies nobody opened for ``KEEP_DAYS``."""
    global _pruned
    if _pruned:
        return
    _pruned = True
    limit = time.time() - KEEP_DAYS * 86400
    try:
        for item in folder().glob('*.lumen'):
            status = item.stat()
            if max(status.st_atime, status.st_mtime) < limit:
                item.unlink(missing_ok=True)
    except OSError:
        log.debug('working copies not pruned', exc_info=True)
