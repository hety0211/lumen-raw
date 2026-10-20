"""Interface themes (1.8.0): a dark and a light look with amber accents.

Dark is near-black around the photograph; light keeps a neutral mid-grey canvas so the photo is
still judged against grey.  The choice is ``settings.ini`` ``appearance`` = ``system`` (follow
the operating system), ``dark`` or ``light`` (View → Appearance) and applies at once.

Widgets that paint themselves read their colours through ``color(name)`` while painting, so a
switch only needs a repaint; everything else is styled by the stylesheet built here.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette

from . import i18n

PREFERENCES = ('system', 'dark', 'light')

DARK = dict(
    window='#08080a', panel='#0f0f12', topbar='#08080a', card='#18181c', border='#1f1f24', border_hi='#3a3a42',
    text='#e4e1da', dim='#7c7972', head='#f6f2e8', disabled='#4d4b47', disabled_bg='#111114',
    input='#0b0b0d', input_border='#25252b', spin='#0f0f12', button='#16161a', button_border='#232329',
    button_hover='#1f1f25', checked='#33270f', checked_border='#f0a530', checked_text='#ffd890',
    accent='#f0a530', accent_text='#1a1204', accent_hover='#ffbb4d', accent_soft='#1d1608', accent_line='#6b4d16',
    groove='#26262c', fill='#a8742a', handle='#f0a530', selection='#2a2110', selection_text='#ffe0a8',
    list='#0b0b0d', scroll='#34343c', status='#08080a', splitter='#050506',
    canvas='#000000', plot='#0b0b0d', grid='#1f1f24', diagonal='#3a3a42', ink='#ffffff', bubble='#000000c8',
    hist_r='#ea7d88', hist_g='#8ed7ac', hist_b='#86adf3', warn='#e4ce8e', error='#b65a4c',
)

LIGHT = dict(
    window='#e8e6e1', panel='#f4f3f0', topbar='#fbfaf8', card='#ffffff', border='#dad7d0', border_hi='#aca79d',
    text='#25231f', dim='#716c64', head='#121110', disabled='#a7a39b', disabled_bg='#ecebe7',
    input='#ffffff', input_border='#d3cfc7', spin='#f4f3f0', button='#ffffff', button_border='#d3cfc7',
    button_hover='#f3efe7', checked='#fbe6c2', checked_border='#d4860f', checked_text='#5c3500',
    accent='#e8962a', accent_text='#1f1302', accent_hover='#f2a845', accent_soft='#fdf0dc', accent_line='#e7c48c',
    groove='#d6d2ca', fill='#c58628', handle='#d4860f', selection='#fbe6c2', selection_text='#4a2b00',
    list='#eceae5', scroll='#c4bfb5', status='#fbfaf8', splitter='#dad7d0',
    canvas='#8f8e8b', plot='#f0eeea', grid='#dcd9d2', diagonal='#bdb8ae', ink='#000000', bubble='#ffffffe0',
    hist_r='#d9505d', hist_g='#3f9f63', hist_b='#4b7fd6', warn='#a4740f', error='#c0503c',
)

PALETTES = {'dark': DARK, 'light': LIGHT}

TEMPLATE = '''
QWidget { background: @panel; color: @text; font-family: @fonts; font-size: 12px; }
QMainWindow, QWidget#library { background: @window; }
QMenuBar { background: @topbar; color: @text; }
QMenuBar::item { padding: 4px 9px; background: transparent; }
QMenuBar::item:selected { background: @button_hover; }
QMenu { background: @card; border: 1px solid @border; padding: 4px; }
QMenu::item { padding: 6px 24px 6px 22px; border-radius: 6px; }
QMenu::item:selected { background: @selection; color: @selection_text; }
QMenu::item:disabled { color: @disabled; }
QMenu::separator { height: 1px; background: @border; margin: 4px 6px; }
QWidget#topbar { background: @topbar; border-bottom: 1px solid @border; }
QLabel#brand { color: @head; font-size: 24px; font-weight: 300; letter-spacing: 10px; background: transparent; }
QLabel#subtle { color: @dim; font-size: 11px; background: transparent; }
QLabel#section { color: @head; font-size: 13px; font-weight: 600; padding: 7px 0; background: transparent; }
QLabel#navigator, QLabel#preview { background: @canvas; color: @dim; border: 1px solid @border; border-radius: 10px; }
QLabel#badge { color: @accent; background: @accent_soft; border: 1px solid @accent_line; border-radius: 6px; padding: 3px 8px; font-size: 10px; }
QPushButton { background: @button; color: @text; border: 1px solid @button_border; border-radius: 14px; padding: 7px 12px; }
QPushButton:hover { background: @button_hover; border-color: @border_hi; }
QPushButton:checked, QPushButton:pressed { background: @checked; border-color: @checked_border; color: @checked_text; }
QPushButton#primary { background: @accent; color: @accent_text; border: 1px solid @accent; padding: 8px 18px; font-weight: 600; }
QPushButton#primary:hover { background: @accent_hover; border-color: @accent_hover; }
QPushButton:disabled { background: @disabled_bg; color: @disabled; border-color: @border; }
QPushButton#primary:disabled { background: @disabled_bg; color: @disabled; border: 1px solid @border; }
QPushButton#swatch { border: 2px solid @border_hi; border-radius: 12px; padding: 0; }
QPushButton#swatch:hover { border-color: @head; }
QToolButton { color: @text; }
QTabWidget::pane { border: 0; border-top: 1px solid @border; }
QTabBar::tab { color: @dim; background: @panel; padding: 12px 9px; font-size: 11px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: @head; border-bottom: 2px solid @accent; }
QSlider::groove:horizontal { background: @groove; height: 2px; border-radius: 1px; }
QSlider::sub-page:horizontal { background: @fill; border-radius: 1px; }
QSlider::handle:horizontal { background: @handle; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:disabled { background: @disabled; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox, QPlainTextEdit, QTextEdit { background: @input; border: 1px solid @input_border; border-radius: 6px; padding: 4px; color: @text; selection-background-color: @selection; selection-color: @selection_text; }
QDoubleSpinBox { border: 0; background: @spin; }
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus { border-color: @accent_line; }
QComboBox { padding: 6px; }
QComboBox QAbstractItemView { background: @card; selection-background-color: @selection; selection-color: @selection_text; }
QCheckBox, QRadioButton { spacing: 7px; padding: 4px 0; background: transparent; }
QCheckBox::indicator { width: 13px; height: 13px; background: @input; border: 1px solid @border_hi; border-radius: 3px; }
QCheckBox::indicator:checked { background: @accent; border-color: @accent; }
QCheckBox:disabled, QRadioButton:disabled { color: @disabled; }
QCheckBox::indicator:disabled { background: @disabled_bg; border-color: @border; }
QCheckBox::indicator:checked:disabled { background: @disabled; border-color: @disabled; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: @panel; width: 6px; }
QScrollBar::handle:vertical { background: @scroll; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: @panel; height: 6px; }
QScrollBar::handle:horizontal { background: @scroll; border-radius: 3px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QListWidget { background: @list; border: 1px solid @border; border-radius: 10px; }
QListWidget::item { padding: 8px; }
QListWidget::item:selected { background: @selection; color: @selection_text; }
QListWidget#presetList { border: 0; background: @list; }
QListWidget#presetList::item { background: @card; border: 1px solid @border; border-radius: 10px; padding: 5px; }
QListWidget#presetList::item:hover { border-color: @border_hi; }
QListWidget#presetList::item:selected { border-color: @accent; background: @selection; }
QProgressBar { background: @input; border: 0; border-radius: 2px; }
QProgressBar::chunk { background: @accent; border-radius: 2px; }
QToolTip { color: @text; background: @card; border: 1px solid @border_hi; padding: 5px; }
QStatusBar { background: @status; color: @dim; }
QStatusBar QLabel { background: transparent; }
QSplitter::handle { background: @splitter; }
QWidget#tabStrip { background: transparent; }
QToolButton#tabButton { color: @dim; background: transparent; padding: 10px 6px; font-size: 11px; border: 0; border-bottom: 2px solid transparent; }
QToolButton#tabButton:hover { color: @text; }
QToolButton#tabButton:checked { color: @head; border-bottom: 2px solid @accent; }
QLineEdit[invalid="true"] { border-color: @error; }
QPushButton#modeButton { border-radius: 8px; padding: 6px 15px; background: transparent; border: 1px solid @border; }
QPushButton#modeButton:hover { background: @button_hover; border-color: @border_hi; }
QPushButton#modeButton:checked { background: @accent_soft; border-color: @accent_line; color: @accent; font-weight: 600; }
QListView#libraryGrid { background: @canvas; border: 0; }
QTreeWidget#sourceTree { background: transparent; border: 0; }
QTreeWidget#sourceTree::item { padding: 3px 2px; border-radius: 5px; }
QTreeWidget#sourceTree::item:selected { background: @selection; color: @selection_text; }
QToolButton#keywordChip { background: @card; border: 1px solid @border_hi; border-radius: 10px; padding: 2px 8px; color: @text; }
QToolButton#keywordChip:hover { border-color: @accent_line; color: @accent; }
QLabel#infoTitle { color: @head; font-size: 13px; font-weight: 600; background: transparent; }
QLabel#hints { color: @warn; background: transparent; }
'''

_mode = 'dark'


def preference():
    value = i18n.setting('appearance', 'system')
    return value if value in PREFERENCES else 'system'


def save_preference(value):
    i18n.save_setting('appearance', value)


def system_mode():
    """``dark`` or ``light`` as the operating system asks; dark when it does not say."""
    app = QGuiApplication.instance()
    scheme = app.styleHints().colorScheme() if app is not None else Qt.ColorScheme.Unknown
    return 'light' if scheme == Qt.ColorScheme.Light else 'dark'


def mode():
    """The look in use: ``dark`` or ``light``."""
    return _mode


def color(name, alpha=None):
    """A QColor of the current theme (``#rrggbbaa`` values carry their alpha)."""
    value = PALETTES[_mode][name]
    result = QColor(value[:7])
    if len(value) == 9:
        result.setAlpha(int(value[7:], 16))
    if alpha is not None:
        result.setAlpha(alpha)
    return result


def stylesheet(look='dark'):
    values = PALETTES[look]
    sheet = TEMPLATE.replace('@fonts', ', '.join(f'"{f}"' for f in i18n.fonts()))
    for key in sorted(values, key=len, reverse=True):
        sheet = sheet.replace('@' + key, values[key][:7])
    return sheet


def palette(look):
    """The Qt palette of a look, for the parts the stylesheet does not reach (message boxes,
    item views, native-looking dialogs)."""
    v = PALETTES[look]
    p = QPalette()
    roles = {QPalette.ColorRole.Window: v['panel'], QPalette.ColorRole.WindowText: v['text'],
             QPalette.ColorRole.Base: v['input'], QPalette.ColorRole.AlternateBase: v['card'],
             QPalette.ColorRole.Text: v['text'], QPalette.ColorRole.Button: v['button'],
             QPalette.ColorRole.ButtonText: v['text'], QPalette.ColorRole.BrightText: v['head'],
             QPalette.ColorRole.Highlight: v['accent'], QPalette.ColorRole.HighlightedText: v['accent_text'],
             QPalette.ColorRole.ToolTipBase: v['card'], QPalette.ColorRole.ToolTipText: v['text'],
             QPalette.ColorRole.PlaceholderText: v['dim'], QPalette.ColorRole.Link: v['accent'],
             QPalette.ColorRole.Light: v['border_hi'], QPalette.ColorRole.Midlight: v['border'],
             QPalette.ColorRole.Mid: v['border'], QPalette.ColorRole.Dark: v['window'],
             QPalette.ColorRole.Shadow: v['window']}
    for role, value in roles.items():
        p.setColor(role, QColor(value[:7]))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(v['disabled']))
    return p


def apply(app, preference_value=None):
    """Style ``app`` for a preference (default: the saved one) and repaint every widget."""
    global _mode
    value = preference_value or preference()
    hints = app.styleHints()
    if hasattr(hints, 'setColorScheme'):
        # Qt 6.8+: also the window frames (Windows 11 title bars) follow the chosen look.
        hints.setColorScheme({'dark': Qt.ColorScheme.Dark, 'light': Qt.ColorScheme.Light}.get(value, Qt.ColorScheme.Unknown))
    _mode = system_mode() if value == 'system' else value
    app.setPalette(palette(_mode))
    app.setStyleSheet(stylesheet(_mode))
    for widget in app.allWidgets():
        widget.update()
    return _mode


def follow_system(app):
    """Re-apply the theme when the operating system switches between light and dark."""
    def changed(*_):
        if preference() == 'system':
            look = system_mode()
            if look != _mode:
                apply(app, 'system')
    app.styleHints().colorSchemeChanged.connect(changed)
