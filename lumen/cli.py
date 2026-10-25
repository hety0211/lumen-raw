"""``lumen-cli``: LUMEN RAW from the command line and as an MCP server (1.6.0).

    lumen-cli render INPUT OUTPUT [--size 2560] [--quality 92] [--preset NAME] [--lut FILE.cube] [--set exposure=0.3 …]
    lumen-cli import-presets FILE               Lightroom .xmp / .lrtemplate, .cube, .lumenpreset or a .zip pack
    lumen-cli info PHOTO
    lumen-cli list [QUERY]                      e.g. "rating>=3 -flag:reject"
    lumen-cli rate PHOTO… 0-5 | flag PHOTO… pick|reject|none | label PHOTO… red|…|none
    lumen-cli xmp read|write PHOTO…
    lumen-cli import-lightroom CATALOG.lrcat
    lumen-cli commands                          every command id and its parameters
    lumen-cli run COMMAND [JSON] [--headless]   any command, in the open editor when there is one
    lumen-cli mcp [--headless]                  MCP server over stdio (Claude Code, Codex, WorkBuddy …)
    lumen-cli mcp-config [claude-code|codex|json|dsh] [--install]   (--install: DeepSeek Harness only)
    lumen-cli models [--download]               the AI models this version needs (2.1.1)

INPUT is a photo or a ``.lumen`` project.  Results are JSON on stdout; errors go to stderr with
exit status 1.  The editor is used through its control channel when it is open (``run``,
``mcp``); ``render`` and the library commands always work without it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import __version__


def executable():
    """The command an MCP client should start: lumen-cli.exe next to the frozen app, or this source tree."""
    if getattr(sys, 'frozen', False):
        folder = Path(sys.executable).resolve().parent
        name = 'lumen-cli.exe' if sys.platform == 'win32' else 'lumen-cli'
        return [str(folder / name)]
    root = Path(__file__).resolve().parents[1]
    return [sys.executable, str(root / 'main.py'), '--cli']


def mcp_config(kind='json'):
    command = executable()
    program, arguments = command[0], command[1:] + ['mcp']
    quoted = ' '.join(f'"{part}"' if ' ' in part or '\\' in part else part for part in command + ['mcp'])
    if kind == 'claude-code':
        return f'claude mcp add lumen-raw -- {quoted}'
    if kind == 'codex':
        return (f'codex mcp add lumen-raw -- {quoted}\n\n# or in ~/.codex/config.toml:\n'
                f'[mcp_servers.lumen-raw]\ncommand = {json.dumps(program)}\nargs = {json.dumps(arguments)}\n'
                f'startup_timeout_sec = 60\ntool_timeout_sec = 600')
    if kind == 'dsh':
        return (f'# DeepSeek Harness: merge into {dsh_patch_file()} (every profile reads it),\n'
                f'# then restart DSH; the tools appear as mcp__lumen-raw__<tool>.\n' + dsh_entry())
    return json.dumps(dict(mcpServers={'lumen-raw': dict(type='stdio', command=program, args=arguments)}),
                      indent=2, ensure_ascii=False)


# 2.1.1: DeepSeek Harness (DSH) loads MCP servers as @deepseek-ai/dsh-mcp-client plugin entries of a
# Cordis patch file; $DSH_HOME/cordis.patch.yml (normally ~/.dsh) applies to every profile, the
# command-line ones and the Desktop app's.
DSH_BEGIN = '# >>> LUMEN RAW (MCP server) - written by LUMEN RAW; edit or delete this block freely'
DSH_END = '# <<< LUMEN RAW (MCP server)'


def _yaml(text):
    """A YAML single-quoted scalar: backslashes stay literal, quotes are doubled."""
    return "'" + str(text).replace("'", "''") + "'"


def dsh_entry():
    command = executable()
    program, arguments = command[0], command[1:] + ['mcp']
    return '\n'.join([
        '- insert:',
        '    - id: mcp-lumen-raw',
        "      name: '@deepseek-ai/dsh-mcp-client'",
        '      config:',
        '        serverName: lumen-raw',
        '        transport: stdio',
        f'        command: {_yaml(program)}',
        f'        args: [{", ".join(_yaml(a) for a in arguments)}]',
        '        env: {}',
        # Exports of large photos and batches take longer than DSH's default 60 s per call.
        '        toolCallTimeoutMs: 600000',
    ])


def dsh_patch_file():
    home = os.environ.get('DSH_HOME')
    return (Path(home).expanduser() if home else Path.home() / '.dsh') / 'cordis.patch.yml'


def install_dsh(path=None):
    """Add (or refresh) the LUMEN RAW entry in DSH's home patch file; returns ``(path, action)``
    with action ``created`` / ``added`` / ``updated`` / ``present`` (an entry the user wrote).
    The previous file is kept once as ``cordis.patch.yml.lumen-backup``."""
    from .i18n import tr
    path = Path(path or dsh_patch_file())
    block = f'{DSH_BEGIN}\n{dsh_entry()}\n{DSH_END}\n'
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        text = ''
    marked = re.compile(re.escape(DSH_BEGIN.split(' - ')[0]) + r'.*?' + re.escape(DSH_END) + r'[^\n]*\n?', re.S)
    body = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if marked.search(text):
        result, action = marked.sub(lambda _: block, text, count=1), 'updated'
    elif re.search(r'serverName:\s*[\'"]?lumen-raw\b', text):
        return path, 'present'
    elif not body or body == ['[]']:
        kept = '\n'.join(line for line in text.splitlines() if line.strip() != '[]').strip()
        result, action = (kept + '\n\n' if kept else '') + block, 'created' if not text.strip() else 'added'
    elif body[0].startswith('-'):
        result, action = text.rstrip('\n') + '\n\n' + block, 'added'
    else:
        raise ValueError(tr('{path} 不是 YAML 列表，无法自动合并；请复制配置手动合并。', path=path))
    if result == text:
        return path, action
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(path.name + '.lumen-backup')
    if text and not backup.exists():
        backup.write_text(text, encoding='utf-8')
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(result.encode('utf-8'))
    temporary.replace(path)
    return path, action


def models(download=False):
    """``lumen-cli models``: where the AI models are and which are missing; downloads them."""
    from . import ai_models, updates
    ai_models.prepare(keep=updates.keep_models())
    if download and ai_models.missing():
        shown = [-1]

        def progress(done, total):
            percent = int(done * 100 / max(total, 1))
            if percent != shown[0]:
                shown[0] = percent
                sys.stderr.write(f'\r{done / 1e6:.0f} / {total / 1e6:.0f} MB')
                sys.stderr.flush()
        try:
            ai_models.download(progress=progress)
        except Exception as exc:
            sys.stderr.write(f'\n{exc}\n')
            _dump(ai_models.status())
            return 1
        sys.stderr.write('\n')
    _dump(ai_models.status())
    return 0 if not ai_models.missing() else 2


def _dump(result):
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


def _value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _context(headless):
    """The open editor (through its control channel) unless ``headless``; otherwise a session."""
    from . import commands, control
    if not headless:
        try:
            client = control.Client(connect_timeout=.5)

            class Remote:
                mode = 'gui'

                def run(self, command, args=None):
                    return client.call(command, args or {})
            return Remote()
        except control.Unavailable:
            pass
    from .session import Session
    session = Session()

    class Local:
        mode = 'headless'

        def run(self, command, args=None):
            return commands.run(session, command, args or {})
    return Local()


def render(args):
    from . import commands
    from .session import Session
    session = Session()
    run = lambda command, **kw: commands.run(session, command, kw)
    run('photo.open', path=args.input)
    if args.process:
        run('edit.process', version=args.process)
    if args.preset:
        run('edit.preset', name=args.preset, amount=args.amount)
    changes = {}
    for item in args.set or []:
        key, _, value = item.partition('=')
        if not _:
            raise SystemExit(f'--set expects key=value, got {item!r}')
        target = changes.setdefault('adjustments', {})
        if '.' in key:            # e.g. hsl.blue.saturation=20, effects.vignette=-20
            parts = key.split('.')
            target = changes
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            key = parts[-1]
        target[key] = _value(value)
    if args.lut:
        changes['lut'] = dict(path=args.lut, amount=args.lut_amount)
    if changes:
        run('edit.apply', changes=changes)
    limit = dict(max_kb=args.max_kb) if args.max_kb else {}
    result = run('photo.export', output=args.output, quality=args.quality, long_edge=args.size, overwrite=args.overwrite,
                 metadata=not args.no_metadata, location=not args.no_location, space=args.space, **limit)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from . import logs
    logs.configure(filename='lumen-cli.log')
    parser = argparse.ArgumentParser(prog='lumen-cli', description=f'LUMEN RAW {__version__} command line and MCP server')
    parser.add_argument('--version', action='version', version=f'LUMEN RAW {__version__}')
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('render', help='render a photo or .lumen project to JPEG / PNG / TIFF / DNG')
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--size', type=int, help='long edge in pixels (default: full size)')
    p.add_argument('--quality', type=int, default=92)
    p.add_argument('--preset', help='preset name (built-in or imported, see list_presets)')
    p.add_argument('--lut', help='.cube LUT to apply')
    p.add_argument('--lut-amount', type=float, default=100, help='LUT strength 0–100')
    p.add_argument('--amount', type=float, default=100, help='preset strength 0–100')
    p.add_argument('--set', action='append', metavar='KEY=VALUE',
                   help='edit value, e.g. exposure=0.3, highlights=-40, hsl.blue.saturation=20 (repeatable)')
    p.add_argument('--process', type=int, choices=(1, 2), help='process version')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-metadata', action='store_true', help="don't copy the original's EXIF / IPTC / XMP")
    p.add_argument('--no-location', action='store_true', help='copy metadata without the GPS position')
    p.add_argument('--space', choices=('srgb', 'p3', 'adobe-rgb'), default='srgb', help='output colour space')
    p.add_argument('--max-kb', type=int, help='JPEG: largest file size in KB (lowers the quality)')
    p = sub.add_parser('info', help='size, format and shooting data of a photo')
    p.add_argument('photo')
    p = sub.add_parser('list', help='library photos (editor open) or rated photos in the catalog')
    p.add_argument('query', nargs='*')
    for name, choices, help_text in (('rate', None, 'set star ratings (0–5)'), ('flag', ('pick', 'reject', 'none'), 'pick / reject flags'),
                                      ('label', ('red', 'yellow', 'green', 'blue', 'purple', 'none'), 'colour labels')):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('photos', nargs='+')
        p.add_argument('value', choices=choices) if choices else p.add_argument('value', type=int, choices=range(6))
    p = sub.add_parser('xmp', help='read ratings from XMP, or write them to XMP sidecars')
    p.add_argument('direction', choices=('read', 'write'))
    p.add_argument('photos', nargs='+')
    p = sub.add_parser('import-lightroom', help='ratings, flags and labels from a Lightroom Classic catalog')
    p.add_argument('catalog')
    p = sub.add_parser('import-presets', help='Lightroom .xmp / .lrtemplate presets, .cube LUTs or a .zip pack')
    p.add_argument('file')
    sub.add_parser('commands', help='every command id with its parameters')
    p = sub.add_parser('run', help='run any command (in the open editor when there is one)')
    p.add_argument('command')
    p.add_argument('args', nargs='?', default='{}', help='JSON object')
    p.add_argument('--headless', action='store_true', help='never use the open editor')
    p = sub.add_parser('mcp', help='MCP server on stdio')
    p.add_argument('--headless', action='store_true', help='never use the open editor')
    p = sub.add_parser('mcp-config', help='how to add this server to an agent')
    p.add_argument('kind', nargs='?', default='json', choices=('claude-code', 'codex', 'json', 'dsh'))
    p.add_argument('--install', action='store_true', help='DeepSeek Harness: write the entry into its patch file')
    p = sub.add_parser('models', help='the AI models this version needs; --download fetches missing ones')
    p.add_argument('--download', action='store_true')
    args = parser.parse_args(argv)

    if args.action == 'mcp':
        from . import mcp
        return mcp.serve(args.headless)
    if args.action == 'mcp-config':
        if args.install:
            if args.kind != 'dsh':
                parser.error('--install is only available for dsh')
            path, action = install_dsh()
            _dump(dict(file=str(path), action=action))
            return 0
        sys.stdout.write(mcp_config(args.kind) + '\n')
        return 0
    if args.action == 'models':
        return models(args.download)
    from . import catalog, commands
    try:
        if args.action == 'render':
            result = render(args)
        elif args.action == 'info':
            from .session import Session
            result = commands.run(Session(), 'photo.info', dict(path=str(Path(args.photo).resolve())))
        elif args.action == 'list':
            query = ' '.join(args.query)
            context = _context(False)
            if context.mode == 'gui':
                result = context.run('library.list', dict(query=query))
            else:
                library = catalog.Catalog()
                terms = catalog.parse_query(query)
                photos = [dict(path=p, name=Path(p).name, **m) for p, m in library.items()
                          if any(m.values()) and catalog.matches(terms, p, m)]
                result = dict(count=len(photos), photos=photos, source='catalog')
        elif args.action in ('rate', 'flag', 'label'):
            field = dict(rate='rating', flag='flag', label='label')[args.action]
            paths = [str(Path(p).resolve()) for p in args.photos]
            context = _context(False)
            if context.mode == 'gui':
                result = context.run(f'library.{args.action}', {field: args.value, 'paths': paths})
            else:
                result = dict(changed=catalog.Catalog().set(paths, **{field: args.value}))
        elif args.action == 'xmp':
            library = catalog.Catalog()
            paths = [str(Path(p).resolve()) for p in args.photos]
            if args.direction == 'read':
                result = dict(changed=library.merge(catalog.read_xmp(paths), overwrite=True))
            else:
                written, errors = catalog.write_xmp({p: library.get(p) for p in paths})
                result = dict(written=[str(catalog.sidecar(p)) for p in written], errors=errors)
        elif args.action == 'import-lightroom':
            from .session import Session
            result = commands.run(Session(), 'library.import_lightroom', dict(catalog=args.catalog))
        elif args.action == 'import-presets':
            context = _context(False)
            result = context.run('preset.import', dict(path=str(Path(args.file).resolve())))
        elif args.action == 'commands':
            result = dict(commands=commands.describe())
        else:
            payload = _value(args.args)
            if not isinstance(payload, dict):
                raise commands.CommandError('arguments must be a JSON object')
            context = _context(args.headless)
            result = dict(context.run(args.command, payload), mode=context.mode)
    except (commands.CommandError, ValueError, OSError) as exc:
        sys.stderr.write(f'lumen-cli: {exc}\n')
        return 1
    if isinstance(result, dict) and 'image' in result:
        result = {k: v for k, v in result.items() if k != 'image'}
    _dump(result)
    return 0
