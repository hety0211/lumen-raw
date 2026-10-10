"""Smoke test of a frozen ``lumen-cli`` (1.6.0): version, render, and an MCP session over stdio.

Usage: python tools/check_cli.py <lumen-cli executable> <sample photo> <output folder>

Runs with its own catalog and control-channel name, so it never touches the user's library or
talks to an editor window that happens to be open.  Writes report.json into the output folder.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main(executable, sample, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LUMEN_CATALOG=str(folder / 'catalog.jsonl'), LUMEN_CONTROL_NAME=f'lumen-cli-check-{os.getpid()}')
    report = {}

    def run(*args, stdin=None, timeout=300):
        started = time.perf_counter()
        result = subprocess.run([executable, *args], input=stdin, capture_output=True, timeout=timeout, env=env)
        report.setdefault('runs', []).append(dict(args=list(args)[:3], code=result.returncode,
                                                   seconds=round(time.perf_counter() - started, 2),
                                                   stderr=result.stderr.decode('utf-8', 'replace')[-400:]))
        if result.returncode:
            raise SystemExit(f'{args[:2]} failed ({result.returncode}): {result.stderr.decode("utf-8", "replace")[-800:]}')
        return result.stdout.decode('utf-8')

    report['version'] = run('--version').strip()
    output = folder / 'render.jpg'
    if output.exists():
        output.unlink()
    rendered = json.loads(run('render', sample, str(output), '--size', '1200', '--set', 'exposure=0.2', '--set', 'highlights=-30'))
    from PIL import Image
    with Image.open(output) as image:
        report['render'] = dict(output=rendered['output'], size=image.size)
    if max(report['render']['size']) != 1200:
        raise SystemExit(f'render size {report["render"]["size"]}')
    meta = {'_meta': {'io.modelcontextprotocol/protocolVersion': '2026-07-28'}}
    messages = [dict(jsonrpc='2.0', id=1, method='initialize', params=dict(protocolVersion='2025-11-25', capabilities={},
                                                                        clientInfo=dict(name='check', version='1'))),
                dict(jsonrpc='2.0', method='notifications/initialized'),
                dict(jsonrpc='2.0', id=2, method='tools/list'),
                dict(jsonrpc='2.0', id=3, method='server/discover', params=meta),
                dict(jsonrpc='2.0', id=4, method='tools/call', params=dict(name='open_photo', arguments=dict(path=sample), **meta)),
                dict(jsonrpc='2.0', id=5, method='tools/call', params=dict(
                    name='apply_edits', arguments=dict(changes=dict(adjustments=dict(exposure=.3, vibrance=15)))))
                , dict(jsonrpc='2.0', id=6, method='tools/call', params=dict(name='render_preview', arguments=dict(size=512)))]
    stdin = ''.join(json.dumps(m) + '\n' for m in messages).encode('utf-8')
    replies = {}
    for line in run('mcp', '--headless', stdin=stdin).splitlines():
        reply = json.loads(line)
        replies[reply.get('id')] = reply
    tools = [t['name'] for t in replies[2]['result']['tools']]
    ok = (replies[1]['result']['protocolVersion'] == '2025-11-25' and 'apply_edits' in tools
          and '2026-07-28' in replies[3]['result']['supportedVersions']
          and not replies[4]['result']['isError'] and not replies[5]['result']['isError']
          and replies[6]['result']['content'][0]['type'] == 'image')
    report['mcp'] = dict(tools=len(tools), ok=ok, applied=replies[5]['result']['structuredContent'].get('applied'))
    report['ok'] = ok
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:4]))
