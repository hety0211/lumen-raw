"""Interface language (1.5.1).

Source strings in the code are Simplified Chinese and pass through ``tr``; the catalogs in
``lumen/locales/<code>.json`` map them to the other languages.  The language is chosen once
per process: ``LUMEN_LANGUAGE``, else ``settings.ini`` in the data folder (written by the
language menu and by the installer), else the system interface language.  A change takes
effect after a restart, so module-level strings may be translated at import time.
"""
from __future__ import annotations

import configparser
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

#: (code, native name) in menu order.
LANGUAGES = (('zh_CN', '简体中文'), ('zh_TW', '繁體中文'), ('en', 'English'), ('ja', '日本語'),
             ('ko', '한국어'), ('de', 'Deutsch'), ('fr', 'Français'), ('es', 'Español'), ('ru', 'Русский'))
SOURCE = 'zh_CN'
CODES = tuple(code for code, _ in LANGUAGES)
#: Installer (Inno Setup) language names that differ from the codes; older installs used these.
INSTALLER_NAMES = {'zh_CN': 'zhcn', 'zh_TW': 'zhtw'}
_ALIASES = {'zhcn': 'zh_CN', 'zhtw': 'zh_TW', 'zh': 'zh_CN', 'zh_hans': 'zh_CN', 'zh_sg': 'zh_CN',
            'zh_hant': 'zh_TW', 'zh_hk': 'zh_TW', 'zh_mo': 'zh_TW'}
LOCALES = Path(__file__).resolve().parent / 'locales'
#: Interface fonts per language; the bundled Noto Sans SC covers Latin, Cyrillic and Chinese.
#: Latin and Cyrillic text uses the system UI font (Noto Sans SC draws a centred CJK ellipsis).
FONTS = {'zh_TW': ['Microsoft JhengHei UI', 'PingFang TC', 'Noto Sans SC'],
         'ja': ['Yu Gothic UI', 'Meiryo UI', 'Hiragino Sans', 'Noto Sans SC'],
         'ko': ['Malgun Gothic', 'Apple SD Gothic Neo', 'Noto Sans SC'],
         **{code: ['Segoe UI', 'Helvetica Neue'] for code in ('en', 'de', 'fr', 'es', 'ru')}}

_language = None
_catalog = None


def normalize(code):
    """Language code from a setting, an installer name or a system locale, or None."""
    if not code:
        return None
    text = str(code).strip().replace('-', '_')
    lowered = text.lower()
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    parts = lowered.split('_')
    if parts[0] == 'zh':
        script = {'hant', 'tw', 'hk', 'mo'} & set(parts[1:])
        return 'zh_TW' if script else 'zh_CN'
    return parts[0] if parts[0] in CODES else None


def settings_path():
    # Same folder as host.data_folder(); host translates strings at import, so it cannot be used here.
    import sys
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'LUMEN RAW' / 'settings.ini'
    base = os.environ.get('LOCALAPPDATA') or str(Path.home() / '.local' / 'state')
    return Path(base) / 'LUMEN RAW' / 'settings.ini'


def _read_settings(path=None):
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(path or settings_path(), encoding='utf-8-sig')
    except (OSError, configparser.Error, UnicodeDecodeError) as exc:
        log.warning('settings.ini unreadable: %s', exc)
    return parser


def setting(key, default=None, path=None):
    """A value of ``[General]`` in settings.ini (shared with the installer)."""
    parser = _read_settings(path)
    return parser.get('General', key, fallback=default)


def save_setting(key, value, path=None):
    path = Path(path or settings_path())
    parser = _read_settings(path)
    if not parser.has_section('General'):
        parser.add_section('General')
    parser.set('General', key, str(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        parser.write(handle, space_around_delimiters=False)  # as the installer writes it


def system_language():
    try:
        from PySide6.QtCore import QLocale
        for name in QLocale.system().uiLanguages():
            code = normalize(name)
            if code:
                return code
    except Exception:
        pass
    if os.name == 'nt':
        try:
            import ctypes
            import locale
            code = normalize(locale.windows_locale.get(ctypes.windll.kernel32.GetUserDefaultUILanguage()))
            if code:
                return code
        except Exception:
            pass
    return 'en'


def language():
    """Code of the interface language of this process."""
    global _language
    if _language is None:
        _language = (normalize(os.environ.get('LUMEN_LANGUAGE')) or normalize(setting('language'))
                     or system_language())
    return _language


def language_name(code=None):
    return dict(LANGUAGES).get(code or language(), code or language())


def _load(code):
    try:
        return json.loads((LOCALES / f'{code}.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        log.warning('no catalog for %s: %s', code, exc)
        return {}


def catalog():
    """Translations of the current language; missing entries fall back to English (Traditional
    Chinese falls back to the source text, which its readers can read)."""
    global _catalog
    if _catalog is None:
        code = language()
        if code == SOURCE:
            _catalog = {}
        elif code in ('en', 'zh_TW'):
            _catalog = _load(code)
        else:
            _catalog = {**_load('en'), **_load(code)}
    return _catalog


def tr(text, **values):
    """``text`` (Simplified Chinese source) in the interface language; ``{name}`` fields are
    filled from ``values``."""
    translated = catalog().get(text, text) if text else text
    return translated.format(**values) if values else translated


def tr_in(code, text, **values):
    """``text`` in another language (the language menu asks to restart in the chosen one)."""
    code = normalize(code) or SOURCE
    if code == SOURCE:
        translated = text
    else:
        table = _load(code)
        if code not in ('en', 'zh_TW') and text not in table:
            table = _load('en')
        translated = table.get(text, text)
    return translated.format(**values) if values else translated


def N_(text):
    """Marks a source string kept as data (labels in recipes and tables); ``tr`` translates it
    where it is shown."""
    return text


def select(code):
    """Make ``code`` the interface language of this process (tests and diagnostics)."""
    global _language, _catalog
    _language, _catalog = normalize(code) or SOURCE, None


def save_language(code):
    """Remember ``code`` for the next start; also pre-selects it in the next installer run."""
    code = normalize(code)
    if code is None:
        raise ValueError(code)
    save_setting('language', code)
    if os.name == 'nt':
        try:
            import winreg
            key = r'Software\Microsoft\Windows\CurrentVersion\Uninstall\{F1CA8FF7-EB54-4B53-81E3-4CC183C8C1B9}_is1'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_SET_VALUE) as handle:
                winreg.SetValueEx(handle, 'Inno Setup: Language', 0, winreg.REG_SZ, INSTALLER_NAMES.get(code, code))
        except OSError:
            pass  # not installed (source tree or portable copy)


def fonts():
    return FONTS.get(language(), []) + ['Noto Sans SC', 'Microsoft YaHei UI', 'Segoe UI']
