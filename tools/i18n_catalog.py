"""Interface translation catalogs (1.5.1).

    python tools/i18n_catalog.py            check every catalog: missing, unused, placeholders
    python tools/i18n_catalog.py --source   print the source strings (JSON list) in code order

Source strings are the Simplified Chinese literals passed to ``tr``, ``tr_in`` and ``N_``
in ``lumen/*.py``; ``lumen/locales/<code>.json`` maps them to each language.
"""
from __future__ import annotations

import ast
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lumen.i18n import CODES, LOCALES, SOURCE  # noqa: E402


def sources():
    found = {}
    for path in sorted((ROOT / 'lumen').glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else \
                node.func.attr if isinstance(node.func, ast.Attribute) else ''
            position = 1 if name == 'tr_in' else 0 if name in ('tr', 'N_') else None
            if position is None or len(node.args) <= position:
                continue
            arg = node.args[position]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value:
                found.setdefault(arg.value, f'{path.name}:{node.lineno}')
    return found


def fields(text):
    try:
        return sorted({f for _, f, _, _ in string.Formatter().parse(text) if f})
    except ValueError:
        return None


def check():
    found = sources()
    problems = 0
    for code in CODES:
        if code == SOURCE:
            continue
        path = LOCALES / f'{code}.json'
        catalog = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        missing = [s for s in found if s not in catalog]
        unused = [s for s in catalog if s not in found]
        wrong = [s for s in found if s in catalog and fields(s) != fields(catalog[s])]
        empty = [s for s in found if s in catalog and not catalog[s].strip() and s.strip()]
        edges = [s for s in found if s in catalog and catalog[s] and
                 (s.startswith('\n') != catalog[s].startswith('\n') or s.endswith('\n') != catalog[s].endswith('\n'))]
        print(f'{code}: {len(catalog)} entries, {len(missing)} missing, {len(unused)} unused, '
              f'{len(wrong)} placeholder mismatches, {len(empty)} empty, {len(edges)} line-break edges')
        for s in missing[:20]:
            print('   missing', found[s], repr(s[:80]))
        for s in wrong[:20]:
            print('   fields', found[s], repr(s[:60]), '->', repr(catalog[s][:60]))
        for s in edges[:20]:
            print('   edge', found[s], repr(s[:60]), '->', repr(catalog[s][:60]))
        problems += len(missing) + len(wrong) + len(empty)
    return problems


if __name__ == '__main__':
    if '--source' in sys.argv:
        print(json.dumps(list(sources()), ensure_ascii=False, indent=1))
    else:
        sys.exit(1 if check() else 0)
