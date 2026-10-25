"""Check the program of the update installer (2.1.1), which carries no AI models.

    python tools/check_update_package.py <update installer extracted with /PORTABLE=1> <photo> <output folder>

Without models it must report them missing; a model a 2.1.0 installation left (the installer
moves them to ``models/import``) is adopted; ``lumen-cli models --download`` fetches the rest
(``build/model-pack`` served on this machine, ``LUMEN_MODEL_RELEASE``); then the editor passes
the smoke test.  The run has its own LOCALAPPDATA, never the user's.
"""
from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def run(command, environment, timeout=900):
    result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def main(program, photo, output):
    program, output = Path(program).resolve(), Path(output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    cli = program / 'lumen-cli.exe'
    if (program / 'models').exists() and any((program / 'models').iterdir()):
        raise SystemExit('the update package must not contain models')
    # A 2.1.0 installation's model under its plain name, as the update installer moves it.
    adopt = 'face-yunet.onnx'
    (program / 'models' / 'import').mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / 'build' / 'compact-models' / adopt, program / 'models' / 'import' / adopt)
    server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(PROJECT / 'build' / 'model-pack')))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    environment = dict(os.environ, LOCALAPPDATA=str(output / 'appdata'), LUMEN_NO_UPDATE_CHECK='1',
                       LUMEN_MODEL_RELEASE=f'http://127.0.0.1:{server.server_address[1]}/', PYTHONIOENCODING='utf-8')
    report = {}
    try:
        code, out, err = run([str(cli), 'models'], environment)
        before = json.loads(out)
        report['before'] = before
        if code != 2 or not before['packaged'] or adopt in before['missing'] or len(before['missing']) != 14:
            raise SystemExit(f'unexpected model status before the download (exit {code}): {out}{err}')
        code, out, err = run([str(cli), 'models', '--download'], environment)
        after = json.loads(out)
        report['after'] = after
        if code != 0 or after['missing'] or Path(after['download_folder']) != program / 'models':
            raise SystemExit(f'model download failed (exit {code}): {out}{err}')
        smoke = output / 'smoke'
        code, out, err = run([str(program / 'LumenRAW.exe'), '--smoke-test', str(photo), str(smoke)], environment)
        result = json.loads((smoke / 'report.json').read_text(encoding='utf-8'))
        report['smoke'] = dict(exit=code, ok=result.get('ok'), models=len(result.get('models', {})),
                               updates=result.get('updates'), error=result.get('error'))
        if code != 0 or not result.get('ok') or len(result.get('models', {})) != 15:
            raise SystemExit(f'smoke test of the updated program failed (exit {code}): {result.get("error")}')
    finally:
        server.shutdown()
        server.server_close()
    report['update_package_models_ok'] = True
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
