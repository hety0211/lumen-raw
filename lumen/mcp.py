"""MCP server: lets local agents (Claude Code, Codex, WorkBuddy, Claude Desktop …) drive LUMEN RAW (1.6.0).

``lumen-cli mcp`` speaks the Model Context Protocol over stdio (newline-delimited JSON-RPC 2.0).
It is a dual-era server: clients of the 2026-07-28 revision send ``server/discover`` and carry
the protocol version in each request's ``_meta``; older clients (2024-11-05 … 2025-11-25) open
with ``initialize``.  Only standard-library code: no SDK in the frozen app.

Each tool is a command of ``commands``.  When the editor window is open, calls go through its
control channel (``control``): the edits appear on screen and each is one undo step.  Otherwise
a headless ``session.Session`` in this process opens, edits, renders and exports photos.
stdout carries protocol messages only; logs go to stderr.
"""
from __future__ import annotations

import json
import logging
import sys
import threading

from . import __version__, commands, control

log = logging.getLogger(__name__)

MODERN = '2026-07-28'
LEGACY = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')
VERSIONS = (MODERN,) + LEGACY
META = 'io.modelcontextprotocol/'
SERVER = dict(name='lumen-raw', title='LUMEN RAW', version=__version__)

INSTRUCTIONS = '''LUMEN RAW is a RAW photo editor (landscape and travel). Typical flow:
1. list_photos (library, ratings) or add_photos (files/folders), then open_photo.
2. get_edits shows current values; apply_edits sets final values (not increments) as one undoable step.
   Local masks: masks=[{"region": "sky", "adjustments": {...}}] (sky/subject/person/background/foreground are found by local AI).
3. render_preview returns a JPEG to check the result; compare with original=true.
4. export_photo / export_photos write files; save_project keeps the edits (.lumen) without exporting.
5. rate_photos / flag_photos / label_photos organise the library (0–5 stars, pick/reject, colour labels).
lumen_status tells whether the editor window is open ("gui": the user sees each change and can undo it with Ctrl+Z)
or not ("headless": edits live in this server until saved or exported). Originals are never modified.'''


def tools():
    result = []
    for cmd in sorted(commands.REGISTRY.values(), key=lambda c: c.tool or ''):
        if not cmd.tool:
            continue
        result.append(dict(name=cmd.tool, title=cmd.tool.replace('_', ' ').capitalize(), description=cmd.summary,
                           inputSchema=cmd.schema(),
                           annotations=dict(readOnlyHint=cmd.read_only, destructiveHint=False,
                                            idempotentHint=cmd.read_only, openWorldHint=False)))
    result.append(dict(name='run_command', title='Run command',
                       description='Run any LUMEN RAW command by id (see list_commands), e.g. library.write_xmp.',
                       inputSchema=dict(type='object', properties=dict(command=dict(type='string'), args=dict(type='object')),
                                        required=['command'], additionalProperties=False),
                       annotations=dict(readOnlyHint=False, destructiveHint=False, openWorldHint=False)))
    return result


class Dispatcher:
    """Runs commands in the open editor when there is one, else in a headless session."""

    def __init__(self, headless=False):
        self.headless = headless
        self._session = None
        self._lock = threading.Lock()

    def session(self):
        if self._session is None:
            from .session import Session
            self._session = Session()
        return self._session

    def run(self, command, args):
        if not self.headless:
            try:
                client = control.Client(connect_timeout=.5)
            except control.Unavailable:
                client = None
            if client is not None:
                try:
                    return client.call(command, args)
                finally:
                    client.close()
        with self._lock:
            return commands.run(self.session(), command, args)


class Server:
    def __init__(self, dispatcher=None, output=None):
        self.dispatcher = dispatcher or Dispatcher()
        self.output = output or sys.stdout.buffer
        self.legacy = None          # negotiated legacy version after initialize
        self._write_lock = threading.Lock()

    # -- transport

    def send(self, message):
        data = (json.dumps(message, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
        with self._write_lock:
            self.output.write(data)
            self.output.flush()

    def serve(self, stream=None):
        stream = stream or sys.stdin.buffer
        for raw in stream:
            if not raw.strip():
                continue
            try:
                message = json.loads(raw.decode('utf-8'))
            except ValueError:
                self.send(dict(jsonrpc='2.0', id=None, error=dict(code=-32700, message='Parse error')))
                continue
            for reply in self.handle_batch(message):
                self.send(reply)

    def handle_batch(self, message):
        messages = message if isinstance(message, list) else [message]
        replies = [r for r in (self.handle(m) for m in messages) if r is not None]
        return replies

    # -- protocol

    def handle(self, message):
        if not isinstance(message, dict) or message.get('jsonrpc') != '2.0' or 'method' not in message:
            if isinstance(message, dict) and ('result' in message or 'error' in message):
                return None          # a response to a request we never send
            return dict(jsonrpc='2.0', id=message.get('id') if isinstance(message, dict) else None,
                        error=dict(code=-32600, message='Invalid Request'))
        method, ident = message['method'], message.get('id')
        params = message.get('params') or {}
        if ident is None:            # notification
            return None
        try:
            result = self.dispatch(method, params)
        except _RpcError as exc:
            return dict(jsonrpc='2.0', id=ident, error=exc.error)
        except Exception as exc:
            log.exception('MCP %s failed', method)
            return dict(jsonrpc='2.0', id=ident, error=dict(code=-32603, message=f'{type(exc).__name__}: {exc}'))
        return dict(jsonrpc='2.0', id=ident, result=result)

    def dispatch(self, method, params):
        meta = params.get('_meta') or {} if isinstance(params, dict) else {}
        requested = meta.get(META + 'protocolVersion')
        modern = requested is not None
        if modern and requested not in VERSIONS:
            raise _RpcError(-32022, 'Unsupported protocol version', dict(supported=list(VERSIONS), requested=requested))
        if method == 'initialize':
            wanted = params.get('protocolVersion')
            self.legacy = wanted if wanted in LEGACY else LEGACY[0]
            return dict(protocolVersion=self.legacy, capabilities=dict(tools=dict(listChanged=False)),
                        serverInfo=SERVER, instructions=INSTRUCTIONS)
        if method == 'server/discover':
            return self._modern(dict(supportedVersions=list(VERSIONS), capabilities=dict(tools={}),
                                     instructions=INSTRUCTIONS, ttlMs=3600000, cacheScope='private'))
        if method == 'ping':
            return {}
        if method == 'tools/list':
            result = dict(tools=tools())
            return self._modern(dict(result, ttlMs=3600000, cacheScope='private')) if modern else result
        if method == 'tools/call':
            result = self.call(params.get('name'), params.get('arguments') or {})
            return self._modern(result) if modern else result
        if method in ('resources/list', 'prompts/list'):
            key = method.split('/')[0]
            return self._modern({key: [], 'ttlMs': 3600000, 'cacheScope': 'private'}) if modern else {key: []}
        if method == 'resources/templates/list':
            return {'resourceTemplates': []}
        raise _RpcError(-32601, f'Method not found: {method}')

    @staticmethod
    def _modern(result):
        return dict(result, resultType='complete', _meta={META + 'serverInfo': SERVER})

    def call(self, name, arguments):
        if name == 'run_command':
            command, args = arguments.get('command'), arguments.get('args') or {}
        else:
            command = next((c.id for c in commands.REGISTRY.values() if c.tool == name), None)
            args = arguments
        if command is None:
            raise _RpcError(-32602, f'Unknown tool: {name}')
        try:
            result = self.dispatcher.run(command, args)
        except commands.CommandError as exc:
            return dict(content=[dict(type='text', text=str(exc))], isError=True)
        except Exception as exc:
            log.exception('tool %s failed', name)
            return dict(content=[dict(type='text', text=f'{type(exc).__name__}: {exc}')], isError=True)
        content = []
        if command == 'render.preview' and result.get('image'):
            content.append(dict(type='image', data=result['image'], mimeType=result.get('mime', 'image/jpeg')))
            result = {k: v for k, v in result.items() if k != 'image'}
        content.append(dict(type='text', text=json.dumps(result, ensure_ascii=False)))
        return dict(content=content, structuredContent=result, isError=False)


class _RpcError(Exception):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.error = dict(code=code, message=message, **({'data': data} if data is not None else {}))


def serve(headless=False):
    """Run the stdio server until stdin closes."""
    protocol_out = sys.stdout.buffer
    # Anything printed by libraries must not corrupt the protocol stream.
    sys.stdout = sys.stderr
    Server(Dispatcher(headless), protocol_out).serve(sys.stdin.buffer)
    return 0
