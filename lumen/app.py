from __future__ import annotations
import copy
import sys
import time
import threading
import traceback
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QEvent, QObject, QRunnable, QThreadPool, QTimer, Signal, QPointF
from PySide6.QtGui import QAction, QKeySequence, QFont, QFontDatabase, QIcon, QPainter, QColor, QPen, QPolygonF
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QScrollArea, QComboBox, QCheckBox, QListWidget,
    QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QSpinBox,
    QFrame, QSplitter, QProgressDialog, QStatusBar, QSizePolicy)
from . import engine, model, compute, host, i18n, __version__
from .scheduler import Activity as A, Job, JobScheduler, JobSignals, WorkState, WorkStateAccess
from .widgets import AdjustSlider, Canvas, CurveEditor, Histogram, TabStrip
from .studio import StudioMixin
from .revision import RevisionMixin
from .resolution import ResolutionMixin
from .library import LibraryMixin
from .auto_masks import AutoMaskMixin
from .workflow import WorkflowMixin
from .nl_panel import NaturalLanguageMixin
from .lens_panel import LensMixin
from .rating import RatingMixin
from .agent_dialog import AgentMixin
from . import geometry, develop, watermark
from . import performance
from .exposure_curve import ExposureCurve
from .watermark_dialog import WatermarkEditor
from .i18n import tr

STYLE = '''
QWidget { background: #1b2028; color: #dbe0e9; font-family: FAMILIES; font-size: 12px; }
QMainWindow { background: #101318; }
QLabel#brand { color: #e9eaff; font-size: 23px; font-weight: 700; letter-spacing: 3px; }
QLabel#subtle { color: #8390a2; font-size: 11px; }
QLabel#section { color: #b8c4d8; font-size: 13px; font-weight: 600; padding: 5px 0; }
QPushButton { background: #292f3a; border: 1px solid #363e4b; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #353d4c; border-color: #637395; }
QPushButton:pressed, QPushButton:checked { background: #465279; border-color: #9dafff; color: #fff; }
QPushButton:disabled { color: #5e6878; background: #20252d; border-color: #2b323d; }
QPushButton#primary { background: #b3c2ff; border: 0; color: #171d32; font-weight: 600; }
QPushButton#primary:hover { background: #c9d4ff; }
QPushButton#primary:disabled { background: #47516b; color: #8491ab; }
QTabWidget::pane { border: 0; border-top: 1px solid #323a47; }
QTabBar::tab { color: #8f9aae; background: #1b2028; padding: 12px 11px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #c9d3ff; border-bottom: 2px solid #b3c2ff; }
QSlider::groove:horizontal { height: 3px; background: #3a4351; border-radius: 1px; }
QSlider::sub-page:horizontal { background: #808fae; }
QSlider::handle:horizontal { background: #c5cffa; width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox { background: #161b22; border: 1px solid #333d4a; border-radius: 4px; padding: 4px; }
QDoubleSpinBox { border: 0; color: #c3cbe0; }
QComboBox { padding: 6px; }
QComboBox QAbstractItemView { background: #222a36; selection-background-color: #4b587e; }
QCheckBox { spacing: 7px; padding: 4px 0; }
QCheckBox::indicator { width: 13px; height: 13px; background: #171c24; border: 1px solid #56647b; border-radius: 3px; }
QCheckBox::indicator:checked { background: #aab9f1; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #1b2028; width: 7px; }
QScrollBar::handle:vertical { background: #485366; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QListWidget { background: #141921; border: 1px solid #303947; border-radius: 5px; }
QListWidget::item { padding: 8px; }
QListWidget::item:selected { background: #35405a; color: #e0e7ff; }
QToolTip { color: #e2e7f0; background: #30394a; border: 1px solid #54627c; padding: 5px; }
QStatusBar { background: #151a22; color: #8794a7; }
QStatusBar QLabel { background: transparent; }
QSplitter::handle { background: #11161d; width: 2px; }
'''

# 1.5.1: Japanese, Korean and Traditional Chinese put a native font first.
STYLE = STYLE.replace('FAMILIES', ', '.join(f'"{f}"' for f in i18n.fonts()))

# Quiet neutral surfaces keep the photograph dominant; sage accents mark active tools.
STYLE += '''
QWidget { background: #202422; color: #d9ded8; font-size: 12px; }
QMainWindow, QWidget#library { background: #191d1b; }
QWidget#topbar { background: #171b19; border-bottom: 1px solid #353b35; }
QLabel#brand { color: #e5eadc; font-size: 24px; letter-spacing: 4px; background: transparent; }
QLabel#subtle { color: #879389; font-size: 11px; background: transparent; }
QLabel#section { color: #cbd5c7; font-size: 13px; font-weight: 600; padding: 7px 0; background: transparent; }
QLabel#navigator { background: #111513; color: #67766a; border: 1px solid #343c34; border-radius: 7px; }
QLabel#badge { color: #c6d4b7; background: #303a2c; border: 1px solid #4d5c43; border-radius: 4px; padding: 3px 8px; font-size: 10px; }
QPushButton { background: #2a302b; color: #cbd4c7; border: 1px solid #3a443b; border-radius: 6px; padding: 7px 11px; }
QPushButton:hover { background: #354034; border-color: #697b61; }
QPushButton:checked, QPushButton:pressed { background: #47573d; border-color: #a0b18f; color: #f1f5e9; }
QPushButton#primary { background: #c1d1af; color: #202a1c; border: 1px solid #c1d1af; padding: 8px 18px; }
QPushButton#primary:hover { background: #d5e0c7; }
QPushButton:disabled { background: #232822; color: #64705f; border-color: #30372e; }
QTabWidget::pane { border-top: 1px solid #3b453b; }
QTabBar::tab { color: #8e9b8c; background: #202422; padding: 12px 9px; font-size: 11px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #d7e3ca; border-bottom: 2px solid #b9cea0; }
QSlider::groove:horizontal { background: #414a40; height: 3px; }
QSlider::sub-page:horizontal { background: #88967b; }
QSlider::handle:horizontal { background: #c9d8b9; width: 10px; margin: -4px 0; border-radius: 5px; border: 1px solid #c9d8b9; }
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox { background: #181d19; border-color: #394538; color: #cad7c4; }
QDoubleSpinBox { border: 0; background: #1c221d; }
QComboBox QAbstractItemView { background: #252e26; selection-background-color: #46543b; }
QCheckBox::indicator { background: #171d18; border-color: #62735b; }
QCheckBox::indicator:checked { background: #b1c498; border-color: #b1c498; }
QScrollBar:vertical { background: #202422; width: 6px; }
QScrollBar::handle:vertical { background: #4b5947; }
QListWidget { background: #181d19; border-color: #343e33; }
QListWidget::item:selected { background: #394632; color: #e2eada; }
QListWidget#presetList { border: 0; background: #191e1a; }
QListWidget#presetList::item { background: #242c23; border: 1px solid #354132; border-radius: 6px; padding: 5px; }
QListWidget#presetList::item:hover { border-color: #8c9d7c; }
QListWidget#presetList::item:selected { border-color: #b5c99c; background: #39472f; }
QStatusBar { background: #151a16; color: #8c9984; }
QStatusBar QLabel { background: transparent; }
QSplitter::handle { background: #101610; }
QToolTip { color: #e0e9d7; background: #333e2e; border-color: #60704f; }
QPushButton#primary:disabled { background: #394233; color: #788570; border: 1px solid #4e5b44; }
QWidget#tabStrip { background: transparent; }
QToolButton#tabButton { color: #8e9b8c; background: transparent; padding: 10px 6px; font-size: 11px; border: 0; border-bottom: 2px solid transparent; }
QToolButton#tabButton:hover { color: #c3cfba; }
QToolButton#tabButton:checked { color: #d7e3ca; border-bottom: 2px solid #b9cea0; }
'''


class ModeBadge(QLabel):
    """Draw the bolt as a vector so it works without an emoji font."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_gpu = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor('#384b34' if self.is_gpu else '#39362b'))
        painter.setPen(QPen(QColor('#82a178' if self.is_gpu else '#756b50'), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)
        left, _, right = self.text().partition('⚡')
        metrics = painter.fontMetrics()
        bolt_width = 11
        width = metrics.horizontalAdvance(left) + bolt_width + metrics.horizontalAdvance(right)
        x = (self.width() - width) / 2
        baseline = (self.height() + metrics.ascent() - metrics.descent()) / 2
        painter.setPen(QColor('#d8ebc2' if self.is_gpu else '#d2c7aa'))
        painter.drawText(QPointF(x, baseline), left)
        bx = x + metrics.horizontalAdvance(left)
        by = (self.height() - 14) / 2
        points = [(6, 0), (1, 8), (5, 8), (3, 14), (10, 5), (6, 5)]
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#d7e8a4' if self.is_gpu else '#e4ce8e'))
        painter.drawPolygon(QPolygonF([QPointF(bx + px, by + py) for px, py in points]))
        painter.setPen(QColor('#d8ebc2' if self.is_gpu else '#d2c7aa'))
        painter.drawText(QPointF(bx + bolt_width, baseline), right)


class ComputeStatusBar(QStatusBar):
    """Three balanced cells keep the device badge at the real window center."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeGripEnabled(False)
        self._message = ''
        panel = QWidget(self)
        row = QHBoxLayout(panel)
        row.setContentsMargins(9, 1, 9, 1)
        row.setSpacing(5)
        self.message_label = QLabel()
        self.message_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.mode_label = ModeBadge()
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setFixedWidth(178)
        self.state_label = QLabel()
        self.state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.state_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        row.addWidget(self.message_label, 1)
        row.addWidget(self.mode_label)
        row.addWidget(self.state_label, 1)
        self.addPermanentWidget(panel, 1)
        self.mode_label.setText(tr('CPU⚡（{threads} 线程）', threads=performance.THREADS))

    def showMessage(self, message, timeout=0):
        self._message = str(message)
        self.message_label.setToolTip(self._message)
        self._elide()
        if timeout:
            QTimer.singleShot(timeout, lambda: self.clearMessage() if self._message == message else None)

    def currentMessage(self):
        return self._message

    def clearMessage(self):
        self.showMessage('')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._elide)

    def _elide(self):
        self.message_label.setText(self.message_label.fontMetrics().elidedText(
            self._message, Qt.ElideRight, max(20, self.message_label.width() - 8)))

    def refresh_mode(self):
        provider, device, detail, warning = compute.state.snapshot()
        gpu = provider != 'CPUExecutionProvider'
        self.mode_label.is_gpu = gpu
        self.mode_label.setText('GPU⚡' if gpu else tr('CPU⚡（{threads} 线程）', threads=performance.THREADS))
        self.mode_label.setToolTip((tr('当前设备：') + device + '\n' if gpu else '') + detail +
                                    ('\n' + warning if warning else ''))
        self.mode_label.update()


def note(text):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setObjectName('subtle')
    return label


def heading(text):
    label = QLabel(text)
    label.setObjectName('section')
    return label


from .enhance_dialog import ExportDialog


class MainWindow(WorkStateAccess, WorkflowMixin, AgentMixin, NaturalLanguageMixin, LensMixin, RatingMixin, LibraryMixin, ResolutionMixin, AutoMaskMixin, RevisionMixin, StudioMixin, QMainWindow):
    export_progress = Signal(int, str)

    def __init__(self):
        super().__init__()
        # Work state and the job queue exist before any mixin touches a legacy flag.
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(performance.MAX_THREADS)
        self.work = WorkState(self)
        self.scheduler = JobScheduler(self.pool, self)
        self.scheduler.drained.connect(lambda: QTimer.singleShot(0, self.close))
        self.render_cache = engine.RenderCache()
        font =Path(__file__).resolve().parents[1] / 'assets' / 'NotoSansSC.ttf'
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
        self.setWindowTitle(tr('LUMEN RAW {version} · 多品牌 RAW 工作室', version=__version__))
        self.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1]/'assets/lumen.ico')))
        self.resize(1600, 1040)
        self.setMinimumSize(1180, 780)
        self.source = None
        self.source_path = ''
        self.info = {}
        self.edits = model.recipe()
        self.saved_edits = copy.deepcopy(self.edits)
        self.history = model.History(self.edits)
        self.project_path = ''
        self.rendered = None
        self.generation = 0
        self.export_progress.connect(self.update_export_progress)
        self.current_mask = -1
        self.refreshing = False
        self.comparing = False
        self.backend = engine.Backend()
        self.setStatusBar(ComputeStatusBar(self))
        self.compute_timer = QTimer(self)
        self.compute_timer.setInterval(250)
        self.compute_timer.timeout.connect(self.statusBar().refresh_mode)
        self.compute_timer.start()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(180)
        self.timer.timeout.connect(self.render)
        self.history_timer = QTimer(self)
        self.history_timer.setSingleShot(True)
        self.history_timer.setInterval(550)
        self.history_timer.timeout.connect(self.commit)
        self.controls = {}
        self.local_controls = {}
        self.init_studio()
        self.init_resolution()
        self.init_library()
        self.init_rating()
        self.init_natural_language()
        self.build_ui()
        self.shortcuts()
        # 1.6.0: local control channel for lumen-cli and MCP agents.
        self.init_agents()
        self.refresh()

    def button(self, text, callback, primary=False):
        b = QPushButton(text)
        b.clicked.connect(callback)
        if primary:
            b.setObjectName('primary')
        return b

    def build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        top = QWidget()
        top.setObjectName('topbar')
        top.setFixedHeight(76)
        bar = QHBoxLayout(top)
        bar.setContentsMargins(22, 0, 20, 0)
        brand = QLabel('LUMEN')
        brand.setObjectName('brand')
        bar.addWidget(brand)
        bar.addWidget(note(tr('  风光与旅行工作室\n  LANDSCAPE & TRAVEL')))
        badge = QLabel(f'STUDIO {__version__}')
        badge.setObjectName('badge')
        badge.setFixedHeight(25)
        bar.addSpacing(18)
        bar.addWidget(badge)
        bar.addStretch()
        self.open_button = self.button(tr('打开原片'), self.open_file)
        bar.addWidget(self.open_button)
        self.save_button = self.button(tr('保存工程'), self.save)
        bar.addWidget(self.save_button)
        self.export_button = self.button(tr('导出成片  ↗'), self.export, True)
        bar.addWidget(self.export_button)
        outer.addWidget(top)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.library = self.build_library()
        splitter.addWidget(self.library)
        left = QWidget()
        l = QVBoxLayout(left)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(16, 10, 16, 10)
        toolbar.addWidget(self.button(tr('撤销'), lambda: self.undo(-1)))
        toolbar.addWidget(self.button(tr('重做'), lambda: self.undo(1)))
        toolbar.addStretch()
        self.split_check = QCheckBox(tr('前后对比'))
        self.split_check.toggled.connect(self.update_display)
        toolbar.addWidget(self.split_check)
        self.compare = QPushButton(tr('按住看原片'))
        self.compare.pressed.connect(lambda: self.show_original(True))
        self.compare.released.connect(lambda: self.show_original(False))
        toolbar.addWidget(self.compare)
        self.final_view = QCheckBox(tr('成片预览'))
        self.final_view.toggled.connect(self.update_display)
        toolbar.addWidget(self.final_view)
        l.addLayout(toolbar)
        self.canvas = Canvas()
        self.canvas.opened.connect(self.open_file)
        self.canvas.dropped.connect(self.open_path)
        self.canvas.files_dropped.connect(self.import_paths)
        self.canvas.viewport_changed.connect(self.viewport_changed)
        self.canvas.geometry_changed.connect(self.geometry_changed)
        self.canvas.stroke_finished.connect(self.stroke_finished)
        self.canvas.sampled.connect(self.pick_wb)
        l.addWidget(self.canvas, 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(18, 12, 18, 12)
        self.file_label = note(tr('尚未打开原片'))
        footer.addWidget(self.file_label, 1)
        self.zoom_label = note(tr('适应窗口'))
        footer.addWidget(self.zoom_label)
        footer.addWidget(self.button(tr('适应'), self.canvas.fit))
        footer.addWidget(self.button('100%', self.canvas.actual_size))
        self.canvas.zoom_changed.connect(self.zoom_label.setText)
        l.addLayout(footer)
        splitter.addWidget(left)
        right = QWidget()
        right.setMinimumWidth(385)
        right.setMaximumWidth(440)
        r = QVBoxLayout(right)
        r.setContentsMargins(12, 10, 12, 10)
        r.setSpacing(5)
        heading_row = QHBoxLayout()
        heading_row.addWidget(heading(tr('显影  /  DEVELOP')), 1)
        self.auto_button = self.button(tr('自动'), self.automatic)
        self.auto_button.setToolTip(tr('根据原片亮度分布设置曝光、暗部、亮部与对比度'))
        heading_row.addWidget(self.auto_button)
        r.addLayout(heading_row)
        self.histogram = Histogram()
        r.addWidget(self.histogram)
        warning_row = QHBoxLayout()
        self.shadow_warning = QCheckBox(tr('暗部溢出'))
        self.highlight_warning = QCheckBox(tr('高光溢出'))
        self.shadow_warning.toggled.connect(self.update_display)
        self.highlight_warning.toggled.connect(self.update_display)
        warning_row.addWidget(self.shadow_warning)
        warning_row.addStretch()
        warning_row.addWidget(self.highlight_warning)
        r.addLayout(warning_row)
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.tab_changed)
        self.tab_strip = TabStrip(self.tabs)
        r.addWidget(self.tab_strip)
        r.addWidget(self.tabs, 1)
        self.build_basic()
        self.build_color()
        self.build_watermark()
        self.build_details()
        self.build_masks()
        self.build_crop()
        self.build_grading()
        self.build_effects()
        self.build_retouch()
        self.tab_strip.rebuild()
        backend_row = QHBoxLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.addItems([host.ACCELERATION, tr('CPU 模式')])
        self.backend_combo.currentIndexChanged.connect(self.backend_changed)
        backend_row.addWidget(self.backend_combo, 1)
        backend_row.addWidget(self.button(tr('重置'), self.reset_edits))
        r.addLayout(backend_row)
        self.backend_label = note(self.backend.name)
        r.addWidget(self.backend_label)
        r.addWidget(note(tr('运算线程上限 32 · 本机使用 {threads} · 最大 4 亿像素', threads=performance.THREADS)))
        splitter.addWidget(right)
        splitter.setSizes([225, 870, 405])
        outer.addWidget(splitter, 1)
        outer.addWidget(self.build_filmstrip())
        self.setCentralWidget(central)
        self.statusBar().showMessage(tr('准备就绪 · 原片始终保留 · {shortcut} 打开', shortcut=host.keys("Ctrl+O")))
        self.state_label = self.statusBar().state_label
        self.state_label.setText('16-bit RAW  /  sRGB')
        self.statusBar().refresh_mode()
        self.build_menus()

    def panel(self, title):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(5, 12, 9, 12)
        layout.setSpacing(7)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        return layout

    def add_adjustment(self, layout, label, key, low=-100, high=100, step=1, local=False):
        control = AdjustSlider(label, low, high, step)
        control.changed.connect(lambda value, k=key, loc=local: self.adjust(k, value, loc))
        control.committed.connect(self.commit)
        (self.local_controls if local else self.controls)[key] = control
        layout.addWidget(control)

    def build_basic(self):
        l = self.panel(tr('光影'))
        l.addWidget(heading(tr('曝光与动态范围')))
        self.exposure_curve=ExposureCurve()
        self.exposure_curve.changed.connect(self.exposure_curve_changed)
        self.exposure_curve.committed.connect(self.commit)
        l.addWidget(self.exposure_curve)
        l.addWidget(note(tr('曲线任意位置拖动，局部平滑调整并联动黑色／暗部／亮部／白色 · Shift 拖动整体曝光 · 双击还原')))
        for pair in ((('exposure',tr('曝光 EV')),('contrast',tr('对比度'))),(('shadows',tr('暗部')),('highlights',tr('亮部'))),(('blacks',tr('黑色')),('whites',tr('白色')))):
            row=QHBoxLayout()
            for key,title in pair:
                column=QVBoxLayout()
                self.add_adjustment(column,title,key,-5 if key=='exposure' else -100,5 if key=='exposure' else 100,.05 if key=='exposure' else 1)
                row.addLayout(column,1)
            l.addLayout(row)
        self.build_develop_profile(l)
        l.addWidget(heading(tr('白平衡')))
        row = QHBoxLayout()
        self.wb_button = QPushButton(tr('吸管取样'))
        self.wb_button.setCheckable(True)
        self.wb_button.toggled.connect(self.toggle_wb)
        row.addWidget(self.wb_button)
        row.addWidget(self.button(tr('还原相机白平衡'), self.reset_wb))
        l.addLayout(row)
        self.build_camera_wb(l)
        self.add_adjustment(l, tr('冷暖微调'), 'temperature')
        self.add_adjustment(l, tr('色调 · 绿 / 洋红'), 'tint')
        l.addWidget(note(tr('以相机白平衡为起点，相对调整色温与色调。亮部滑块可压低已记录的高光，无法重建完全过曝的信息。')))
        l.addStretch()

    def build_color(self):
        l = self.panel(tr('色彩'))
        l.addWidget(heading(tr('全局色彩')))
        self.mono_check = QCheckBox(tr('黑白处理'))
        self.mono_check.toggled.connect(self.monochrome_changed)
        l.addWidget(self.mono_check)
        self.add_adjustment(l, tr('饱和度'), 'saturation')
        self.add_adjustment(l, tr('自然饱和度'), 'vibrance')
        l.addWidget(heading(tr('色彩混合器')))
        swatches = QHBoxLayout()
        self.swatch_buttons = []
        for index, (title, _, color) in enumerate(model.COLORS):
            b = QPushButton('')
            b.setFixedSize(24, 24)
            b.setToolTip(title)
            b.setStyleSheet(f'QPushButton {{background: {color}; border: 2px solid #343d32; border-radius: 12px; padding: 0;}} QPushButton:hover {{border-color: white;}}')
            b.clicked.connect(lambda checked=False, i=index: self.color_select.setCurrentIndex(i))
            swatches.addWidget(b)
            self.swatch_buttons.append(b)
        l.addLayout(swatches)
        self.color_select = QComboBox()
        for title, _, _ in model.COLORS:
            self.color_select.addItem(title)
        self.color_select.currentIndexChanged.connect(self.refresh_hsl)
        l.addWidget(self.color_select)
        self.hsl_controls = []
        for index, title in enumerate([tr('色相'), tr('饱和度'), tr('明度')]):
            c = AdjustSlider(title)
            c.changed.connect(lambda value, i=index: self.change_hsl(i, value))
            c.committed.connect(self.commit)
            l.addWidget(c)
            self.hsl_controls.append(c)
        l.addWidget(note(tr('8 个相互平滑过渡的色彩区间，可独立调整色相、饱和度与明度。')))
        self.build_curves(l)
        l.addStretch()

    def build_curves(self,l):
        l.addWidget(heading(tr('RGB 与通道曲线')))
        self.channel = QComboBox()
        self.channel.addItems(['RGB', 'R', 'G', 'B'])
        self.channel.currentTextChanged.connect(self.refresh_curve)
        l.addWidget(self.channel)
        self.curve = CurveEditor()
        self.curve.changed.connect(self.curve_changed)
        self.curve.committed.connect(self.commit)
        l.addWidget(self.curve)
        values = QHBoxLayout()
        self.curve_input = QSpinBox()
        self.curve_output = QSpinBox()
        for title, control in [(tr('输入'), self.curve_input), (tr('输出'), self.curve_output)]:
            control.setRange(0, 255)
            control.setKeyboardTracking(False)
            values.addWidget(QLabel(title))
            values.addWidget(control)
        self.curve.selected.connect(self.curve_selection)
        self.curve_input.valueChanged.connect(lambda v: self.curve.report(v / 255))
        self.curve_output.valueChanged.connect(lambda v: self.curve.set_output(self.curve_input.value(), v))
        l.addLayout(values)
        self.curve_mode = QComboBox()
        self.curve_mode.addItems([tr('平滑曲线'), tr('分段直线（兼容旧工程）')])
        self.curve_mode.currentIndexChanged.connect(self.curve_mode_changed)
        l.addWidget(self.curve_mode)
        l.addWidget(note(tr('在任意亮度位置按下，向上或向下拖动。\n输入锁定当前亮度，输出决定调整后的亮度。\n支持 0–255 全部亮度值；右键删除中间节点。')))
        l.addWidget(self.button(tr('重置当前通道'), self.reset_curve))

    def build_watermark(self):
        l=self.panel(tr('水印'))
        l.addWidget(heading(tr('水印与边框')))
        self.watermark_editor=WatermarkEditor(self,compact=True)
        self.watermark_editor.changed.connect(self.watermark_changed)
        l.addWidget(self.watermark_editor)
        l.addWidget(self.button(tr('打开大图水印预览…'),self.configure_watermark))
        l.addStretch()

    def watermark_changed(self,settings):
        if self.refreshing or self.source is None:return
        self.edits['watermark']=settings
        self.changed();self.commit()

    def exposure_curve_changed(self,values,points):
        if self.refreshing or self.source is None:return
        for key in ('exposure','blacks','shadows','highlights','whites'):
            self.edits['adjustments'][key]=values[key]
            self.controls[key].setValue(values[key])
        self.edits['tone_curve']=points
        self.clear_preset_selection();self.changed()

    def build_details(self):
        l = self.panel(tr('细节'))
        l.addWidget(heading(tr('质感')))
        self.add_adjustment(l, tr('去薄雾'), 'dehaze')
        self.add_adjustment(l, tr('清晰度'), 'clarity')
        self.add_adjustment(l, tr('纹理'), 'texture')
        self.add_adjustment(l, tr('锐化'), 'sharpness', 0, 100)
        l.addWidget(self.button(tr('AI 超分辨率…'), lambda: self.open_ai('super'), True))
        l.addWidget(self.button(tr('AI 去杂色…'), lambda: self.open_ai('denoise')))
        l.addWidget(heading(tr('降噪')))
        self.add_adjustment(l, tr('明度降噪'), 'denoise', 0, 100)
        self.add_adjustment(l, tr('彩色杂点'), 'color_noise', 0, 100)
        l.addWidget(note(tr('滑块使用传统降噪；AI 去杂色在独立窗口中处理并生成副本。\n点击 100% 检查原图锐化与降噪效果。')))
        l.addStretch()

    def build_masks(self):
        l = self.panel(tr('蒙版'))
        l.addWidget(heading(tr('局部调整')))
        row = QHBoxLayout()
        for title, kind in [(tr('画笔'), 'brush'), (tr('渐变'), 'linear'), (tr('径向'), 'radial'), (tr('亮度'), 'luminance')]:
            row.addWidget(self.button(title, lambda checked=False, k=kind: self.add_mask(k)))
        l.addLayout(row)
        self.build_auto_masks(l)
        self.mask_list = QListWidget()
        self.mask_list.setFixedHeight(105)
        self.mask_list.currentRowChanged.connect(self.select_mask)
        l.addWidget(self.mask_list)
        row = QHBoxLayout()
        self.overlay_check = QCheckBox(tr('显示范围'))
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self.update_overlay)
        self.enabled_check = QCheckBox(tr('启用'))
        self.enabled_check.setChecked(True)
        self.enabled_check.toggled.connect(lambda v: self.mask_property('enabled', v))
        self.invert_check = QCheckBox(tr('反选'))
        self.invert_check.toggled.connect(lambda v: self.mask_property('invert', v))
        row.addWidget(self.overlay_check)
        row.addWidget(self.enabled_check)
        row.addWidget(self.invert_check)
        l.addLayout(row)
        row = QHBoxLayout()
        self.erase_check = QCheckBox(tr('橡皮擦'))
        self.erase_check.toggled.connect(lambda v: setattr(self.canvas, 'erase', v))
        row.addWidget(self.erase_check)
        row.addStretch()
        row.addWidget(self.button(tr('删除蒙版'), self.delete_mask))
        l.addLayout(row)
        self.brush_size = AdjustSlider(tr('画笔大小'), 1, 40)
        self.brush_size.setValue(9)
        self.brush_size.default_value = 9
        self.brush_size.changed.connect(lambda v: setattr(self.canvas, 'brush_radius', v / 200))
        l.addWidget(self.brush_size)
        self.opacity = AdjustSlider(tr('不透明度'), 0, 100)
        self.opacity.default_value = 100
        self.opacity.changed.connect(lambda v: self.mask_property('opacity', v))
        l.addWidget(self.opacity)
        self.feather = AdjustSlider(tr('羽化'), 0, 100)
        self.feather.default_value = 50
        self.feather.changed.connect(lambda v: self.mask_property('feather', v))
        l.addWidget(self.feather)
        self.range_box = QWidget()
        range_layout = QVBoxLayout(self.range_box)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.range_controls = {}
        for key, title in [('low', tr('亮度下限')), ('high', tr('亮度上限')), ('falloff', tr('过渡范围'))]:
            c = AdjustSlider(title, 0, 100)
            c.changed.connect(lambda value, k=key: self.range_value(k, value))
            c.committed.connect(self.commit)
            self.range_controls[key] = c
            range_layout.addWidget(c)
        range_layout.addWidget(note(tr('按原片的显示亮度选择范围，适合天空、高光或阴影。')))
        l.addWidget(self.range_box)
        for key, title in [('exposure', tr('局部曝光 EV')), ('shadows', tr('局部暗部')), ('highlights', tr('局部亮部')), ('temperature', tr('局部色温')), ('saturation', tr('局部饱和度')), ('clarity', tr('局部清晰度')), ('sharpness', tr('局部锐化'))]:
            self.add_adjustment(l, title, key, -5 if key == 'exposure' else (0 if key == 'sharpness' else -100), 5 if key == 'exposure' else 100, .05 if key == 'exposure' else 1, True)
        l.addWidget(note(tr('画笔：按住左键涂抹。渐变／径向：拖出范围，可重新拖动绘制。\n渐变从起点 0% 过渡至终点 100%；径向框内生效。蒙版按列表顺序叠加。')))
        l.addStretch()

    def build_crop(self):
        l = self.panel(tr('裁切·镜头'))
        l.addWidget(heading(tr('重新构图')))
        self.crop_ratio = QComboBox()
        self.crop_ratio.addItems([tr('自由比例'), tr('原片比例'), '1 : 1', '3 : 2', '4 : 3', '16 : 9', '2 : 3', '9 : 16'])
        self.crop_ratio.currentIndexChanged.connect(self.ratio_changed)
        l.addWidget(self.crop_ratio)
        l.addWidget(note(tr('在画面上拖动绘制裁切框。切换比例后重新拖动。\n裁切与蒙版始终保存在原片坐标中。')))
        l.addWidget(self.button(tr('确认裁切（Enter）'), self.confirm_crop, True))
        l.addWidget(self.button(tr('重新裁切'), lambda: self.final_view.setChecked(False)))
        l.addWidget(self.button(tr('清除裁切'), self.clear_crop))
        l.addWidget(self.button(tr('顺时针旋转 90°'), self.rotate))
        self.straighten = AdjustSlider(tr('水平校正  °'), -15, 15, .1)
        self.straighten.changed.connect(self.straighten_changed)
        self.straighten.committed.connect(self.commit)
        l.addWidget(self.straighten)
        l.addWidget(note(tr('校正时自动放大填满裁切框，保留当前输出比例。')))
        l.addWidget(note(tr('按 Enter 确认，只显示保留部分；重新裁切可调整范围。\n原片数据始终保留，蒙版和修复可在裁切后继续绘制。')))
        self.build_lens(l)
        l.addStretch()

    def shortcuts(self):
        for shortcut, fn in [('Ctrl+O', self.open_file), ('Ctrl+S', self.save), ('Ctrl+E', self.export),
                             ('Ctrl+Z', lambda: self.undo(-1)), ('Ctrl+Shift+Z', lambda: self.undo(1)),
                             ('Ctrl+0', self.canvas.fit), ('J', self.toggle_clipping),
                             ('Return', self.confirm_crop), ('Enter', self.confirm_crop),
                             ('Y', lambda: self.split_check.setChecked(not self.split_check.isChecked())),
                             ('Esc', lambda: self.wb_button.setChecked(False)),
                             ('Ctrl+K', self.open_palette)]:
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(fn)
            self.addAction(action)
        self.rating_shortcuts()

    def open_palette(self):
        from .palette import CommandPalette
        CommandPalette(self).exec()

    def job(self, fn, success, fail=None, priority=0):
        """Queue background image work; see scheduler.JobScheduler."""
        return self.scheduler.submit(fn, success, fail or self.error, priority)

    def error(self, text):
        self.statusBar().showMessage(tr('操作未完成'))
        QMessageBox.warning(self, 'LUMEN RAW', text)

    def may_discard(self):
        if self.source is None or (self.edits == self.saved_edits and self.snapshots == self.saved_snapshots):
            return True
        answer = QMessageBox.question(self, tr('保存编辑'), tr('当前编辑尚未保存为工程。是否保存？'),
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def open_file(self):
        if self.work.busy(A.AI):return
        paths, _ = QFileDialog.getOpenFileNames(self, tr('导入照片、工程或选片集'), '',
            engine.PHOTO_FILTER)
        if paths:
            self.import_paths(paths)

    def import_paths(self, paths):
        if not self.work.can_start(A.LOADING):
            self.statusBar().showMessage(tr('请等待当前读取或导出完成。'))
            return
        paths = [str(Path(p).resolve()) for p in paths if Path(p).is_file()]
        if not paths:return
        if Path(paths[0]).suffix.lower()=='.lumenalbum':
            return self.open_album(paths[0])
        photos = [p for p in paths if Path(p).suffix.lower() not in ('.lumen','.lumenalbum')]
        self.add_documents(photos)
        self.open_path(paths[0])

    def open_path(self, path):
        if self.work.busy(A.AI):return
        if Path(path).suffix.lower()=='.lumenalbum':
            return self.open_album(path)
        if not self.work.can_start(A.LOADING):
            self.statusBar().showMessage(tr('请等待当前读取／导出完成。'))
            return
        self.stash_document()
        explicit_project = Path(path).suffix.lower()=='.lumen'
        document = self.documents.get(str(Path(path).resolve())) if not explicit_project else None
        edits = copy.deepcopy(document['edits']) if document else model.recipe()
        snapshots = copy.deepcopy(document['snapshots']) if document else []
        project = document['project'] if document else ''
        try:
            if Path(path).suffix.lower() == '.lumen':
                project = path
                path, edits, snapshots = model.load_project(path, include_snapshots=True)
                if not Path(path).exists():
                    QMessageBox.information(self, tr('重新定位原片'), tr('工程引用的原片已移动，请选择对应的原片。'))
                    path, _ = QFileDialog.getOpenFileName(self, tr('定位工程原片'))
                    if not path:
                        return
        except Exception as exc:
            self.error(str(exc))
            return
        self.work.begin(A.LOADING)
        self.cancel_detail()
        self.open_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.timer.stop()
        self.history_timer.stop()
        self.generation += 1
        self.statusBar().showMessage(tr('正在解码原片…'))
        def loaded(result):
            self.source, self.info = result
            self.source_path = str(Path(path).resolve())
            if not explicit_project and (not document or not document['initialized']):
                for state in [edits] + ([document['saved_edits']] if document else []):
                    state['white_balance'] = copy.deepcopy(self.info.get('white_balance', state['white_balance']))
                    state['develop'] = copy.deepcopy(self.info.get('develop', state['develop']))
                if document and 'history' in document:
                    for state in document['history'].items:
                        state['white_balance'] = copy.deepcopy(edits['white_balance'])
                        state['develop'] = copy.deepcopy(edits['develop'])
            # 1.5.1: solve the lens profile for this photo in every state it may return to.
            self.init_lens_states([edits] + (self.lens_states_for_document(document) if document else []),
                                  not explicit_project and (not document or not document['initialized']))
            self.reset_resolution()
            self.canvas.image = None
            self.edits = edits
            self.clear_clone_source()
            self.project_path = project
            self.saved_edits = copy.deepcopy(edits)
            self.snapshots = snapshots
            self.saved_snapshots = copy.deepcopy(snapshots)
            self.clear_preset_selection()
            self.history = model.History(edits)
            self.current_mask = -1
            self.rendered = None
            self.work.end(A.LOADING)
            self.open_button.setEnabled(True)
            self.final_view.setChecked(False)
            self.split_check.setChecked(False)
            self.wb_button.setChecked(False)
            self.canvas.fit()
            self.file_label.setText(f'{self.info["name"]}   ·   {self.info["width"]} × {self.info["height"]}   ·   {self.info["format"]}')
            self.file_label.setToolTip(self.info['note'])
            if document is None:
                self.add_documents([self.source_path])
                self.documents[self.source_path]['saved_edits'] = copy.deepcopy(edits)
            self.loaded_document(self.source_path, explicit_project)
            self.refresh()
            self.update_thumbnails()
            self.changed()
        def failed(text):
            self.work.end(A.LOADING)
            self.open_button.setEnabled(True)
            self.refresh()
            self.error(tr('无法打开文件：\n') + text)
        self.job(lambda: engine.load_image(path), loaded, failed)

    def changed(self):
        if self.refreshing or self.source is None:
            return
        self.generation += 1
        self.cancel_stale_detail()
        self.timer.start()
        self.history_timer.start()
        self.detail_timer.start(220)
        dirty =self.edits != self.saved_edits or self.snapshots != self.saved_snapshots
        self.state_label.setText(tr('未保存编辑') if dirty else tr('编辑已保存'))

    def commit(self):
        self.history_timer.stop()
        self.history.push(self.edits)

    def render(self):
        if self.source is None or self.work.busy(A.LOADING, A.AI):
            return
        if self.work.active(A.RENDER):
            self.work.pending_render = True
            return
        self.work.begin(A.RENDER)
        self.work.pending_render = False
        source, edits, token, backend = self.source, copy.deepcopy(self.edits), self.generation, self.backend
        cache = self.render_cache
        # The comparison original is developed here once per photo / develop setting, not on the GUI thread.
        original_key = (self.document_token, engine._key(edits.get('develop', {}), engine.process_version(edits)))
        original = self._original[0] != original_key
        self.statusBar().showMessage(tr('正在更新预览…'))
        started = time.perf_counter()
        def work():
            result = engine.process(source, edits, backend, apply_crop=False, detail_scale=detail_scale, cache=cache)
            before = engine.develop_view(source, edits) if original else None
            return result, before
        def finish(output):
            self.work.end(A.RENDER)
            result, before = output
            if before is not None and source is self.source and self._original[0] != original_key:
                self._original = (original_key, before)
            if token == self.generation:
                self.rendered = result
                self.histogram.set_image(engine.crop_rotate(result, self.edits))
                self.update_display()
                self.update_navigator()
                self.backend_label.setText(self.backend.name)
                self.backend_label.setToolTip(self.backend.warning)
                self.statusBar().showMessage(tr('预览已更新 · {ms:.0f} ms · 放大时自动读取原图细节', ms=(time.perf_counter() - started) * 1000))
            if self.work.pending_render or token != self.generation:
                self.timer.start()
        def failed(text):
            self.work.end(A.RENDER)
            if token == self.generation:
                self.error(tr('预览处理失败：\n') + text)
            elif self.work.pending_render:
                self.timer.start()
        detail_scale=max(.1,source.shape[1]/self.info.get('width',source.shape[1]))
        self.job(work, finish, failed)

    def update_display(self, *_):
        """Compose the preview-size display; original-resolution detail arrives as tiles (1.4.0)."""
        if self.source is None:
            return
        source, edited = self.source, self.rendered
        if edited is None:return
        original = self.preview_original()
        image = original if self.comparing else edited
        final = self.final_view.isChecked()
        size = (self.info.get('width',source.shape[1]),self.info.get('height',source.shape[0]))
        if final:
            matrix,display_size,brush_scale = geometry.frame(self.edits,size)
            inverse = np.linalg.inv(matrix)
            self.canvas.to_source = lambda p,m=inverse:geometry.point(m,p)
            self.canvas.to_display = lambda p,m=matrix:geometry.point(m,p)
            self.canvas.reference_size = display_size
            self.canvas.brush_scale = brush_scale
        else:
            self.canvas.to_source = self.canvas.to_display = None
            self.canvas.reference_size = size
            self.canvas.brush_scale = 1.
        display = engine.crop_rotate(image,self.edits) if final else image
        self.canvas.set_image(display)
        split = self.split_check.isChecked() and not self.comparing
        self.canvas.set_before((engine.crop_rotate(original,self.edits) if final else original) if split else None, split)
        self.canvas.set_clipping(display,self.shadow_warning.isChecked() and not self.comparing,
                                 self.highlight_warning.isChecked() and not self.comparing)
        self.canvas.crop = None if final else self.edits['crop']
        self.refresh_detail()
        self.update_tool()
        self.update_overlay()

        if self.tabs.currentIndex()==2:self.watermark_editor.update_preview()

    def show_original(self, state):
        self.comparing = state
        self.update_display()

    def adjust(self, key, value, local=False):
        if self.refreshing:
            return
        if local:
            mask = self.selected_mask()
            if mask is None:
                return
            mask['adjustments'][key] = value
        else:
            self.edits['adjustments'][key] = value
            self.clear_preset_selection()
            self.exposure_curve.set_values(self.edits['adjustments'],self.edits.get('tone_curve'))
        self.changed()

    def change_hsl(self, index, value):
        if not self.refreshing:
            self.clear_preset_selection()
            self.edits['hsl'][self.color_select.currentIndex()][index] = value
            self.changed()

    def refresh_hsl(self, *_):
        for c, v in zip(self.hsl_controls, self.edits['hsl'][self.color_select.currentIndex()]):
            c.setValue(v)

    def curve_changed(self, points):
        self.edits['curves'][self.channel.currentText()] = points
        self.clear_preset_selection()
        self.changed()

    def refresh_curve(self, *_):
        channel = self.channel.currentText()
        self.curve.set_points(self.edits['curves'][channel], channel, self.edits['curve_mode'])
        self.curve_mode.blockSignals(True)
        self.curve_mode.setCurrentIndex(0 if self.edits['curve_mode'] == 'smooth' else 1)
        self.curve_mode.blockSignals(False)
        self.curve.report(self.curve_input.value() / 255)

    def curve_selection(self, x, y):
        for control, value in [(self.curve_input, x), (self.curve_output, y)]:
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)

    def curve_mode_changed(self, index):
        self.edits['curve_mode'] = 'smooth' if index == 0 else 'linear'
        self.refresh_curve()
        self.clear_preset_selection()
        self.changed()
        self.commit()

    def reset_curve(self):
        self.edits['curves'][self.channel.currentText()] = [[0., 0.], [1., 1.]]
        self.refresh_curve()
        self.clear_preset_selection()
        self.changed()
        self.commit()

    def selected_mask(self):
        return self.edits['masks'][self.current_mask] if 0 <= self.current_mask < len(self.edits['masks']) else None

    def add_mask(self, kind):
        if self.source is None:
            return
        if len(self.edits['masks']) >= 32:
            return self.error(tr('最多支持 32 个蒙版。'))
        self.edits['masks'].append(model.new_mask(kind, len(self.edits['masks']) + 1))
        self.current_mask = len(self.edits['masks']) - 1
        self.split_check.setChecked(False)
        self.refresh()
        self.changed()
        self.commit()

    def select_mask(self, index):
        if self.refreshing:
            return
        self.current_mask = index
        self.refresh_mask_controls()
        self.update_tool()
        self.update_overlay()

    def delete_mask(self):
        if self.selected_mask() is not None:
            self.edits['masks'].pop(self.current_mask)
            self.current_mask = min(self.current_mask, len(self.edits['masks']) - 1)
            self.refresh()
            self.changed()
            self.commit()

    def mask_property(self, key, value):
        if self.refreshing or self.selected_mask() is None:
            return
        self.selected_mask()[key] = value
        self.update_overlay()
        self.changed()

    def refresh_mask_controls(self):
        old = self.refreshing
        self.refreshing = True
        mask = self.selected_mask()
        for c in [self.enabled_check, self.invert_check, self.opacity, self.feather, *self.local_controls.values()]:
            c.setEnabled(mask is not None)
        self.brush_size.setEnabled(mask is not None and (mask['kind']=='brush' or 'raster' in mask))
        self.erase_check.setEnabled(mask is not None and (mask['kind']=='brush' or 'raster' in mask))
        is_range = mask is not None and mask['kind'] == 'luminance'
        self.range_box.setVisible(is_range)
        self.feather.setVisible(not is_range)
        if is_range:
            self.range_controls['low'].setValue(mask['luminance_range'][0])
            self.range_controls['high'].setValue(mask['luminance_range'][1])
            self.range_controls['falloff'].setValue(mask['range_falloff'])
        if mask:
            self.enabled_check.setChecked(mask['enabled'])
            self.invert_check.setChecked(mask['invert'])
            self.opacity.setValue(mask['opacity'])
            self.feather.setValue(mask['feather'])
            for key, control in self.local_controls.items():
                control.setValue(mask['adjustments'][key])
        self.refreshing = old

    def update_overlay(self, *_):
        mask = self.selected_mask()
        if self.source is not None and mask and self.tabs.currentIndex() == 4 and self.overlay_check.isChecked() and not self.comparing and not self.split_check.isChecked():
            reference = self.preview_reference() if mask['kind'] == 'luminance' else None
            alpha = engine.mask_alpha(mask,self.source.shape,reference)
            self.canvas.set_overlay(engine.crop_rotate(alpha,self.edits) if self.final_view.isChecked() else alpha)
        else:
            self.canvas.set_overlay(None)

    def update_tool(self, *_):
        if self.comparing or self.split_check.isChecked():
            self.canvas.tool = 'view'
        elif hasattr(self,'color_region_button') and self.color_region_button.isChecked() and self.tabs.currentIndex()==4:
            self.canvas.tool = 'color'
        elif self.wb_sampling:
            self.canvas.tool = 'sample'
        elif self.tabs.currentIndex() == 8:
            self.canvas.tool = 'heal' if self.retouch_tool.currentIndex() == 0 else 'clone'
            self.canvas.brush_radius = self.retouch_controls['size'].spin.value() / 200
        elif self.tabs.currentIndex() == 5:
            self.canvas.tool = 'view' if self.final_view.isChecked() else 'crop'
        elif self.tabs.currentIndex() == 4 and self.selected_mask():
            mask = self.selected_mask()
            self.canvas.brush_radius = self.brush_size.spin.value() / 200
            self.canvas.tool = mask['kind'] if mask['kind'] in ('brush','linear','radial') else ('brush' if self.refine_mask_check.isChecked() and 'raster' in mask else 'view')
            self.canvas.mask_geometry = [mask['start'], mask['end']]
        else:
            self.canvas.tool = 'view'
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if self.canvas.tool in ('sample','color') else Qt.CursorShape.ArrowCursor)
        self.canvas.update()

    def tab_changed(self, *_):
        if not hasattr(self, 'overlay_check'):
            return
        if self.tabs.currentIndex()==5:
            self.final_view.setChecked(False)
        if self.tabs.currentIndex() in (4,5,8):
            self.split_check.setChecked(False)
        if self.wb_sampling and self.tabs.currentIndex() != 0:
            self.wb_button.setChecked(False)
        if self.tabs.currentIndex()==2:self.watermark_editor.update_preview()
        self.update_tool()
        self.update_overlay()

    def geometry_changed(self, geometry):
        a, b = geometry
        if self.canvas.tool == 'crop':
            crop = [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
            if crop[2] - crop[0] < .002 or crop[3] - crop[1] < .002:
                return
            self.edits['crop'] = crop
        elif self.selected_mask():
            self.selected_mask().update(start=a, end=b)
        self.update_display()
        self.changed()
        self.commit()

    def stroke_finished(self, stroke):
        if stroke.get('kind') in ('heal', 'clone'):
            self.add_retouch(stroke)
            return
        mask = self.selected_mask()
        if mask and (mask['kind']=='brush' or 'raster' in mask):
            mask['strokes'].append(stroke)
            self.update_overlay()
            self.changed()
            self.commit()

    def ratio_changed(self, index):
        self.canvas.ratio = [None, self.source.shape[1] / self.source.shape[0] if self.source is not None else 1.5,
                             1., 1.5, 4 / 3, 16 / 9, 2 / 3, 9 / 16][index]

    def confirm_crop(self):
        if self.source is None or self.tabs.currentIndex()!=5:return
        self.final_view.setChecked(True)
        self.canvas.fit()
        self.commit()
        self.statusBar().showMessage(tr('裁切已确认 · 仅显示保留部分，可继续调色或绘制蒙版'))

    def clear_crop(self):
        self.edits['crop'] = None
        self.final_view.setChecked(False)
        self.canvas.fit()
        self.update_display()
        self.changed()
        self.commit()

    def rotate(self):
        self.edits['rotation'] = (self.edits['rotation'] + 1) % 4
        self.final_view.setChecked(True)
        self.update_display()
        self.changed()
        self.commit()

    def backend_changed(self, index):
        self.backend = engine.Backend('cpu' if index else 'auto')
        if index:
            compute.state.report('CPUExecutionProvider', detail=tr('{threads} 线程 · 手动 CPU 模式', threads=performance.THREADS))
        self.statusBar().refresh_mode()
        self.backend_label.setText(self.backend.name)
        self.backend_label.setToolTip(self.backend.warning)
        self.changed()

    def reset_edits(self):
        if self.source is not None:
            self.commit()
            self.edits = model.recipe()
            self.edits['white_balance'] = copy.deepcopy(self.info.get('white_balance', self.edits['white_balance']))
            self.edits['develop'] = copy.deepcopy(self.info.get('develop',self.edits['develop']))
            self.init_lens_states([self.edits], True)
            self.clear_clone_source()
            self.clear_preset_selection()
            self.current_mask = -1
            self.refresh()
            self.changed()
            self.commit()

    def undo(self, delta):
        if self.source is None:
            return
        if delta < 0:
            self.commit()
        self.edits = self.history.move(delta)
        self.clear_preset_selection()
        self.current_mask = min(self.current_mask, len(self.edits['masks']) - 1)
        self.refresh()
        self.changed()
        self.history_timer.stop()

    def refresh(self):
        self.refreshing = True
        for key, control in self.controls.items():
            control.setValue(self.edits['adjustments'][key])
        self.refresh_hsl()
        self.refresh_curve()
        self.exposure_curve.set_values(self.edits['adjustments'],self.edits.get('tone_curve'))
        self.watermark_editor.set_settings(self.edits['watermark'])
        self.mask_list.clear()
        self.mask_list.addItems([m['name'] for m in self.edits['masks']])
        self.mask_list.setCurrentRow(self.current_mask)
        self.refresh_mask_controls()
        self.refresh_studio()
        self.refresh_lens()
        self.refresh_camera_wb()
        self.develop_combo.blockSignals(True)
        self.develop_combo.setCurrentIndex(0 if self.edits['develop']['mode']=='camera' else 1)
        self.develop_combo.blockSignals(False)
        self.develop_hint.setText(tr(self.edits['develop']['source']))  # stored in the recipe as source text
        self.refresh_process()
        self.update_library_status()
        self.refresh_retouch()
        self.save_button.setEnabled(self.source is not None and not self.work.busy(A.LOADING))
        self.export_button.setEnabled(self.source is not None and not self.work.busy(A.EXPORTING, A.LOADING))
        self.refresh_access()
        self.compare.setEnabled(self.source is not None)
        self.refreshing = False
        self.ratio_changed(self.crop_ratio.currentIndex())
        self.update_display()

    def save(self):
        if self.source is None or self.loading:
            return False
        path = self.project_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, tr('保存无损编辑工程'), str(Path(self.source_path).with_suffix('.lumen')), tr('Lumen 工程 (*.lumen)'))
        if not path:
            return False
        if Path(path).suffix.lower() != '.lumen':
            path += '.lumen'
        try:
            model.save_project(path, self.source_path, self.edits, self.snapshots)
            self.project_path = path
            self.saved_edits = copy.deepcopy(self.edits)
            self.saved_snapshots = copy.deepcopy(self.snapshots)
            self.state_label.setText(tr('编辑已保存'))
            self.statusBar().showMessage(tr('工程已保存 · 原片未修改'))
            return True
        except Exception as exc:
            self.error(str(exc))
            return False

    def export(self, checked=False, enhance=False):
        if enhance:return self.open_ai('super')
        if self.work.busy(A.AI):return
        if self.work.busy(A.SELECTION):
            self.statusBar().showMessage(tr('正在生成蒙版，请完成后再导出。'))
            return
        if self.source is None or not self.work.can_start(A.EXPORTING):
            return
        dialog = ExportDialog(self, self.source_path, enhance)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        ext = ['.jpg', '.png', '.tif', '.dng'][dialog.format.currentIndex()]
        path, _ = QFileDialog.getSaveFileName(self, tr('导出成片'), str(Path(self.source_path).with_name(Path(self.source_path).stem + '-Lumen' + ext)), tr('图片 (*{ext})', ext=ext))
        if not path:
            return
        if Path(path).suffix.lower() != ext:
            path += ext
        if Path(path).resolve() == Path(self.source_path).resolve():
            return self.error(tr('请选择不同的文件名，以保留原片。'))
        self.start_export(path, [1, 2, 4][dialog.scale.currentIndex()], dialog.model_choice(), dialog.quality.value())

    def start_export(self, path, scale=1, model_path='', quality=95):
        if self.work.busy(A.AI):return
        if self.work.busy(A.SELECTION):
            self.statusBar().showMessage(tr('正在生成蒙版，请完成后再导出。'))
            return
        if not self.work.can_start(A.EXPORTING):
            return
        if Path(path).resolve() == Path(self.source_path).resolve():
            return self.error(tr('请选择不同的文件名，以保留原片。'))
        self.work.begin(A.EXPORTING)
        self.export_cancel = threading.Event()
        self.export_dialog = QProgressDialog(tr('正在全尺寸解码…'), tr('取消导出'), 0, 100, self)
        self.export_dialog.setWindowTitle(tr('增强与导出'))
        self.export_dialog.setAutoClose(False)
        self.export_dialog.setAutoReset(False)
        self.export_dialog.setMinimumDuration(0)
        self.export_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.export_dialog.canceled.connect(self.export_cancel.set)
        self.export_dialog.show()
        self.export_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.statusBar().showMessage(tr('正在全尺寸导出… 大图或超分可能需要较长时间。'))
        edits, source_path = copy.deepcopy(self.edits), self.source_path
        photo = copy.deepcopy(self.info.get('photo', {}))
        use_cuda = self.backend_combo.currentIndex() == 0
        def work():
            self.export_progress.emit(2, tr('正在全尺寸解码…'))
            source, _ = engine.load_image(source_path, preview_limit=None)
            if self.export_cancel.is_set():
                raise InterruptedError(tr('已取消导出'))
            self.export_progress.emit(15, tr('正在应用调色、修复与蒙版…'))
            h, w = source.shape[:2]
            crop = edits['crop'] or [0, 0, 1, 1]
            pixels = h * w * (crop[2] - crop[0]) * (crop[3] - crop[1]) * scale * scale
            if pixels > engine.MAX_EXPORT_PIXELS:
                raise ValueError(tr('输出超过 4 亿像素，请先裁切或降低超分倍数。'))
            result = engine.process(source, edits, engine.Backend('auto' if use_cuda else 'cpu'))
            del source
            self.export_progress.emit(30, tr('正在增强…'))
            def progress(done, total):
                self.export_progress.emit(30 + round(65 * done / total), tr('正在增强 · 分块 {done} / {total}', done=done, total=total))
            result, backend = engine.super_resolve(result, scale, model_path or None, use_cuda, progress, self.export_cancel)
            if self.export_cancel.is_set():
                raise InterruptedError(tr('已取消导出'))
            self.export_progress.emit(98, tr('正在写入成片…'))
            result = watermark.apply(result, edits['watermark'], photo)
            engine.export_image(path, result, quality, photo=photo)
            return result.shape, backend
        def success(result):
            self.export_dialog.reset()
            self.work.end(A.EXPORTING)
            self.open_button.setEnabled(True)
            self.export_button.setEnabled(True)
            shape, backend = result
            self.statusBar().showMessage(tr('导出完成 · {width} × {height} · {backend} · {path}', width=shape[1], height=shape[0], backend=backend, path=path))
            QMessageBox.information(self, tr('导出完成'), tr('已保存：\n{path}\n\n{width} × {height} 像素\n{backend}', path=path, width=shape[1], height=shape[0], backend=backend))
        def failed(text):
            self.export_dialog.reset()
            self.work.end(A.EXPORTING)
            self.open_button.setEnabled(True)
            self.export_button.setEnabled(True)
            if self.export_cancel.is_set():
                self.statusBar().showMessage(tr('导出已取消，未写入成片。'))
            else:
                self.error(tr('导出失败：\n') + text)
        self.job(work, success, failed)

    def update_export_progress(self, value, text):
        if self.work.active(A.EXPORTING) and not self.export_cancel.is_set():
            self.export_dialog.setValue(value)
            self.export_dialog.setLabelText(text)

    def closeEvent(self, event):
        if self.closing:
            if self.scheduler.active is None:event.accept()
            else:event.ignore()
            return
        if self.work.busy():
            self.statusBar().showMessage(tr('正在处理图像，请等待完成，或在 AI 窗口取消后关闭。'))
            event.ignore()
            return
        if not self.confirm_close_library():
            event.ignore()
            return
        self.timer.stop()
        self.history_timer.stop()
        self.detail_timer.stop()
        self.cancel_detail()
        self.thumbnail_queue.clear()
        self.nl_recorder.cancel()
        self.nl_request+=1
        self.control.shutdown()
        self.closing=True
        if not self.scheduler.shutdown():
            self.statusBar().showMessage(tr('正在完成当前预览任务后关闭…'))
            event.ignore()
        else:event.accept()


class FileOpenEvents(QObject):
    """macOS delivers Finder “Open With”, Dock drops and ``open -a`` as QFileOpenEvent, not argv."""
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.paths = []
        # Several files opened together arrive as separate events; import them as one batch.
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self.flush)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.FileOpen and event.file():
            self.paths.append(event.file())
            self.timer.start()
            return True
        return False

    def flush(self):
        paths, self.paths = self.paths, []
        if paths:
            self.window.import_paths(paths)


def main():
    from . import logs
    logs.configure()
    logs.describe_system()
    app = QApplication(sys.argv)
    from . import ai_worker
    app.aboutToQuit.connect(ai_worker.shutdown)
    app.setApplicationName('LUMEN RAW')
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    font = QFont(i18n.fonts()[0], 9)
    font.setFamilies(i18n.fonts())
    app.setFont(font)
    window = MainWindow()
    if sys.platform == 'darwin':
        app.installEventFilter(FileOpenEvents(window))
    window.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        QTimer.singleShot(100, lambda: window.open_path(sys.argv[1]))
    code = app.exec()
    if getattr(window, 'restart_requested', False):
        # 1.5.1: the language menu restarts the editor in the chosen language.
        from PySide6.QtCore import QProcess
        arguments = [] if getattr(sys, 'frozen', False) else [str(Path(sys.argv[0]).resolve())]
        QProcess.startDetached(sys.executable, arguments)
    return code
