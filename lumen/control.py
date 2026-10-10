"""Local control channel of the editor window (1.6.0).

While LUMEN RAW runs it listens on a local socket that only the same user can open (a named pipe
on Windows, a Unix socket elsewhere).  Requests are JSON lines ``{"id", "command", "args"}`` and
each gets one reply line ``{"id", "ok", "result" | "error"}``; commands are those of
``commands``.  ``lumen-cli`` and its MCP server use the channel when the editor is open, so an
agent's edits appear in the window and each is one step the user can undo.

Commands run one at a time on a worker thread; whatever touches the window runs on the GUI
thread through ``Bridge.call``.  ``Client`` is the standard-library side used by the CLI.
"""
from __future__ import annotations

import copy
import getpass
import json
import logging
import os
import queue
import re
import socket
import tempfile
import threading
import time
from pathlib import Path

from . import commands, engine, i18n, model

log = logging.getLogger(__name__)

SETTING = 'agent_control'


def channel_name():
    override = os.environ.get('LUMEN_CONTROL_NAME')   # tests run their own editor windows
    if override:
        return re.sub(r'[^A-Za-z0-9_.-]', '_', override)
    user = re.sub(r'[^A-Za-z0-9_.-]', '_', getpass.getuser() or 'user')
    return f'lumen-raw-control-{user}'


def enabled():
    return i18n.setting(SETTING, '1') == '1'


def _endpoint():
    if os.name == 'nt':
        return r'\\.\pipe' + '\\' + channel_name()
    return os.path.join(tempfile.gettempdir(), channel_name())


# --------------------------------------------------------------------------- client

class Unavailable(ConnectionError):
    """No editor window is listening."""


class Client:
    """Synchronous JSON-lines client of a running editor (standard library only)."""

    def __init__(self, connect_timeout=1.):
        self._buffer = b''
        self._ids = 0
        endpoint = _endpoint()
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                if os.name == 'nt':
                    self._pipe = open(endpoint, 'r+b', buffering=0)
                    self._socket = None
                else:
                    self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self._socket.connect(endpoint)
                    self._pipe = None
                return
            except FileNotFoundError:
                raise Unavailable('LUMEN RAW is not running')
            except (ConnectionRefusedError, PermissionError, OSError) as exc:
                # All pipe instances busy (Windows 231) or a stale socket file.
                if time.monotonic() > deadline:
                    raise Unavailable(f'LUMEN RAW is not reachable: {exc}')
                time.sleep(.05)

    def _send(self, data):
        if self._pipe is not None:
            self._pipe.write(data)
        else:
            self._socket.sendall(data)

    def _receive(self):
        chunk = self._pipe.read(65536) if self._pipe is not None else self._socket.recv(65536)
        if not chunk:
            raise ConnectionError('LUMEN RAW closed the connection')
        return chunk

    def call(self, command, args=None):
        self._ids += 1
        self._send((json.dumps(dict(id=self._ids, command=command, args=args or {}), ensure_ascii=False) + '\n').encode('utf-8'))
        while b'\n' not in self._buffer:
            self._buffer += self._receive()
        line, self._buffer = self._buffer.split(b'\n', 1)
        reply = json.loads(line.decode('utf-8'))
        if not reply.get('ok'):
            raise commands.CommandError(reply.get('error') or 'command failed')
        return reply.get('result', {})

    def close(self):
        try:
            (self._pipe or self._socket).close()
        except OSError:
            pass


def available():
    try:
        Client(connect_timeout=.2).close()
        return True
    except Unavailable:
        return False


# --------------------------------------------------------------------------- server side

def _qt():
    from PySide6.QtCore import QObject, Qt, Signal
    return QObject, Qt, Signal


QObject, Qt, Signal = _qt()


class Bridge(QObject):
    """Runs a function on the GUI thread and waits for its result."""
    invoke = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.invoke.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, box):
        function, done, out = box
        try:
            out['value'] = function()
        except BaseException as exc:
            out['error'] = exc
        finally:
            done.set()

    def call(self, function, timeout=300):
        if threading.current_thread() is threading.main_thread():
            return function()
        done, out = threading.Event(), {}
        self.invoke.emit((function, done, out))
        if not done.wait(timeout):
            raise commands.CommandError('the editor did not respond in time')
        if 'error' in out:
            raise out['error']
        return out.get('value')


class WindowContext(commands.Context):
    """Commands acting on the open editor window."""
    mode = 'gui'

    def __init__(self, window, bridge):
        self.window, self.bridge = window, bridge
        self.catalog = window.catalog
        self._backend = None
        self._sources = {}

    def _gui(self, function):
        return self.bridge.call(function)

    def photos(self):
        w = self.window
        return self._gui(lambda: [w.filmstrip.item(i).data(Qt.ItemDataRole.UserRole) for i in range(w.filmstrip.count())])

    def current(self):
        return self._gui(lambda: self.window.source_path if self.window.source is not None else None)

    def selected(self):
        return self._gui(self.window.selected_paths)

    def select(self, paths):
        def run():
            w = self.window
            w.filmstrip.clearSelection()
            for path in paths:
                item = w.film_item(path)
                if item is not None:
                    item.setSelected(True)
        self._gui(run)

    def add(self, paths):
        def run():
            before = set(self.window.documents)
            self.window.add_documents(paths)
            return [p for p in self.window.documents if p not in before]
        return self._gui(run)

    def open(self, path):
        from .scheduler import Activity as A
        w = self.window
        target = str(Path(path).resolve())
        if Path(target).suffix.lower() == '.lumen':
            target = model.load_project(target)[0]
        deadline = time.monotonic() + 30
        while self._gui(lambda: w.work.busy(A.LOADING, A.AI)):
            if time.monotonic() > deadline:
                raise commands.CommandError('the editor is busy; try again')
            time.sleep(.1)
        self._gui(lambda: w.open_path(path))
        deadline = time.monotonic() + 300
        while True:
            time.sleep(.08)
            state = self._gui(lambda: (w.source_path, w.source is not None, w.work.busy(A.LOADING)))
            if not state[2]:
                if state[0] == target and state[1]:
                    return target
                raise commands.CommandError(f'the editor could not open {path}')
            if time.monotonic() > deadline:
                raise commands.CommandError('opening the photo timed out')

    def edits(self, path):
        def run():
            w = self.window
            if path == w.source_path and w.source is not None:
                return copy.deepcopy(w.edits)
            document = w.documents.get(path)
            if document is None:
                raise commands.CommandError(f'not in the library: {path}')
            return copy.deepcopy(document['edits'])
        return self._gui(run)

    def initialized(self, path):
        w = self.window
        return self._gui(lambda: path == w.source_path or bool(w.documents.get(path, {}).get('initialized')))

    def apply(self, path, edits, label):
        def run():
            w = self.window
            if path == w.source_path and w.source is not None:
                w.commit()
                w.edits = copy.deepcopy(edits)
                w.current_mask = min(w.current_mask, len(w.edits['masks']) - 1)
                w.clear_preset_selection()
                w.refresh()
                w.changed()
                w.commit()
                w.stash_document()
            else:
                document = w.documents[path]
                history = document.setdefault('history', model.History(document['edits']))
                document['edits'] = copy.deepcopy(edits)
                history.push(document['edits'])
                item = w.film_item(path)
                if item is not None and not item.text().endswith(' •'):
                    item.setText(item.text() + ' •')
            w.agent_activity(label, path)
        self._gui(run)

    def step(self, path, delta):
        def run():
            w = self.window
            if path == w.source_path and w.source is not None:
                w.undo(delta)
                return copy.deepcopy(w.edits)
            document = w.documents[path]
            history = document.setdefault('history', model.History(document['edits']))
            document['edits'] = history.move(delta)
            return copy.deepcopy(document['edits'])
        return self._gui(run)

    def source(self, path):
        def run():
            w = self.window
            return (w.source, copy.deepcopy(w.info)) if path == w.source_path and w.source is not None else None
        shown = self._gui(run)
        if shown is not None:
            return shown
        if path not in self._sources:
            self._sources = {path: engine.load_image(path)}
        return self._sources[path]

    def save_project(self, path, target):
        def run():
            w = self.window
            if path == w.source_path and w.source is not None:
                model.save_project(target, path, w.edits, w.snapshots)
                w.project_path = target
                w.saved_edits = copy.deepcopy(w.edits)
                w.saved_snapshots = copy.deepcopy(w.snapshots)
                w.stash_document()
                w.state_label.setText(i18n.tr('编辑已保存'))
            else:
                document = w.documents[path]
                model.save_project(target, path, document['edits'], document['snapshots'])
                document.update(project=target, saved_edits=copy.deepcopy(document['edits']),
                                saved_snapshots=copy.deepcopy(document['snapshots']))
            return target
        return self._gui(run)

    def backend(self):
        if self._backend is None:
            cpu = self._gui(lambda: self.window.backend_combo.currentIndex() == 1)
            self._backend = engine.Backend('cpu' if cpu else 'auto')
        return self._backend

    def notify(self, text):
        self._gui(lambda: self.window.agent_activity(text))


class Server(QObject):
    """The listening side, owned by the main window; also runs the window's own commands
    (command palette) on the same worker thread."""
    reply = Signal(object, bytes)
    finished = Signal(object, object)

    def __init__(self, window):
        super().__init__(window)
        from PySide6.QtNetwork import QLocalServer
        self.window = window
        self.server = QLocalServer(self)
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.server.newConnection.connect(self._accept)
        self.bridge = Bridge(self)
        self.context = WindowContext(window, self.bridge)
        self.reply.connect(self._write, Qt.ConnectionType.QueuedConnection)
        self.finished.connect(lambda callback, outcome: callback(outcome), Qt.ConnectionType.QueuedConnection)
        self.queue = queue.Queue()
        self.buffers = {}
        self.worker = threading.Thread(target=self._work, name='lumen-control', daemon=True)
        self.worker.start()

    @property
    def listening(self):
        return self.server.isListening()

    def start(self):
        from PySide6.QtNetwork import QLocalServer
        name = channel_name()
        if not self.server.listen(name):
            QLocalServer.removeServer(name)   # a stale socket file left by a crash (macOS)
            if not self.server.listen(name):
                log.warning('control channel unavailable: %s', self.server.errorString())
                return False
        log.info('control channel listening as %s', name)
        return True

    def stop(self):
        self.server.close()

    def shutdown(self):
        self.stop()
        self.queue.put(None)

    def _accept(self):
        while self.server.hasPendingConnections():
            connection = self.server.nextPendingConnection()
            self.buffers[connection] = b''
            connection.readyRead.connect(lambda c=connection: self._read(c))
            connection.disconnected.connect(lambda c=connection: self._closed(c))

    def _closed(self, connection):
        self.buffers.pop(connection, None)
        connection.deleteLater()

    def _read(self, connection):
        data = self.buffers.get(connection, b'') + bytes(connection.readAll())
        while b'\n' in data:
            line, data = data.split(b'\n', 1)
            if line.strip():
                self.queue.put((connection, line))
        if len(data) > 64 * 2**20:
            data = b''
        self.buffers[connection] = data

    def submit(self, command, args, callback):
        """Run a command for the window itself; ``callback(result or CommandError)`` on the GUI thread."""
        self.queue.put(('local', command, args, callback))

    def _work(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            if item[0] == 'local':
                _, command, args, callback = item
                try:
                    outcome = commands.run(self.context, command, args)
                except commands.CommandError as exc:
                    outcome = exc
                except Exception as exc:
                    log.exception('command %s failed', command)
                    outcome = commands.CommandError(f'{type(exc).__name__}: {exc}')
                self.finished.emit(callback, outcome)
                continue
            connection, line = item
            self.reply.emit(connection, self.handle(line))

    def handle(self, line):
        ident = None
        try:
            request = json.loads(line.decode('utf-8'))
            ident = request.get('id')
            result = commands.run(self.context, request.get('command'), request.get('args') or {})
            reply = dict(id=ident, ok=True, result=result)
        except commands.CommandError as exc:
            reply = dict(id=ident, ok=False, error=str(exc))
        except Exception as exc:
            log.exception('control command failed')
            reply = dict(id=ident, ok=False, error=f'{type(exc).__name__}: {exc}')
        return json.dumps(reply, ensure_ascii=False).encode('utf-8')

    def _write(self, connection, data):
        import shiboken6
        if connection in self.buffers and shiboken6.isValid(connection):
            connection.write(data + b'\n')
            connection.flush()
