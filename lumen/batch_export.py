"""Batch export of the photos selected in the filmstrip (1.4.1).

Every photo is exported with its own edits, crop and watermark, one at a time in
the background; a failed photo is reported and the batch continues.  Photos that
were never opened get their camera white balance and develop reference first,
exactly as opening them in the editor would.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import cv2
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout)

from . import engine, lens, model, watermark
from .i18n import tr

log = logging.getLogger(__name__)

FORMATS = (('JPEG · 8-bit sRGB', '.jpg'), ('PNG · 8-bit sRGB', '.png'), ('TIFF · 16-bit sRGB', '.tif'),
           (tr('DNG · 16-bit 线性成片'), '.dng'))
#: Output sizes: None keeps the full size, otherwise the long edge in pixels (never enlarged).
SIZES = ((tr('原始尺寸'), None), (tr('长边 4000 像素'), 4000), (tr('长边 2560 像素'), 2560), (tr('长边 1600 像素'), 1600))


class BatchExportDialog(QDialog):
    def __init__(self, parent, count, folder):
        super().__init__(parent)
        self.setWindowTitle(tr('批量导出'))
        self.setMinimumWidth(520)
        root = QVBoxLayout(self)
        title = QLabel(tr('批量导出  /  {count} 张照片', count=count))
        title.setObjectName('section')
        root.addWidget(title)
        text = QLabel(tr('每张照片使用各自的调色、裁切、蒙版、修复与水印，从全尺寸原片逐张生成；原片不会被修改。\n'
                      '同名文件不会被覆盖，会自动加编号。AI 超分辨率请在单张导出中使用。'))
        text.setWordWrap(True)
        text.setObjectName('subtle')
        root.addWidget(text)
        form = QFormLayout()
        self.format = QComboBox()
        self.format.addItems([name for name, _ in FORMATS])
        self.size = QComboBox()
        self.size.addItems([name for name, _ in SIZES])
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(95)
        self.folder = QLineEdit(str(folder))
        browse = QPushButton(tr('选择…'))
        browse.clicked.connect(self.choose_folder)
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        for name, widget in ((tr('输出格式'), self.format), (tr('输出尺寸'), self.size), (tr('JPEG 质量'), self.quality)):
            form.addRow(name, widget)
        form.addRow(tr('保存到'), row)
        root.addLayout(form)
        self.status = QLabel('')
        self.status.setObjectName('subtle')
        root.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('导出 {count} 张', count=count))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr('取消'))
        buttons.accepted.connect(self.confirm)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.format.currentIndexChanged.connect(lambda i: self.quality.setEnabled(FORMATS[i][1] == '.jpg'))

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr('选择导出目录'), self.folder.text())
        if folder:
            self.folder.setText(folder)

    def confirm(self):
        if not Path(self.folder.text()).is_dir():
            self.status.setText(tr('请选择已存在的导出目录。'))
            return
        self.accept()

    def options(self):
        return dict(extension=FORMATS[self.format.currentIndex()][1], long_edge=SIZES[self.size.currentIndex()][1],
                    quality=self.quality.value(), folder=self.folder.text())


def target_path(folder, source, extension, taken):
    """``<name>-Lumen<ext>`` in ``folder``; numbered instead of overwriting anything."""
    stem = Path(source).stem + '-Lumen'
    path = Path(folder) / (stem + extension)
    number = 2
    while path.exists() or str(path).lower() in taken or path.resolve() == Path(source).resolve():
        path = Path(folder) / f'{stem}-{number}{extension}'
        number += 1
    taken.add(str(path).lower())
    return path


def fit_long_edge(rgb, long_edge):
    h, w = rgb.shape[:2]
    if not long_edge or max(h, w) <= long_edge:
        return rgb
    scale = long_edge / max(h, w)
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def run(items, options, backend, progress, cancel):
    """Worker: export ``items`` = [(path, edits, initialized)]; returns ``[(path, target or None, error)]``."""
    results, taken = [], set()
    total = len(items)
    for index, (path, edits, initialized) in enumerate(items):
        if cancel.is_set():
            break
        name = Path(path).name
        progress(round(100 * index / total), tr('正在导出 {v} / {total} · {name}', v=index + 1, total=total, name=name))
        try:
            source, info = engine.load_image(path, None, develop_reference=not initialized)
            edits = copy.deepcopy(edits)
            if not initialized:
                # As MainWindow.open_path does for a photo opened the first time.
                edits['white_balance'] = copy.deepcopy(info.get('white_balance', edits['white_balance']))
                edits['develop'] = copy.deepcopy(info.get('develop', edits['develop']))
            edits = model.validate(edits)
            # 1.5.1: the lens profile of this photo (synchronised recipes carry another photo's).
            edits['lens'] = lens.prepare(edits['lens'], info.get('lens_match'))[0]
            if cancel.is_set():
                break
            result = engine.process(source, edits, backend)
            del source
            result = fit_long_edge(result, options['long_edge'])
            photo = info.get('photo', {})
            result = watermark.apply(result, edits['watermark'], photo)
            target = target_path(options['folder'], path, options['extension'], taken)
            engine.export_image(target, result, options['quality'], photo=photo)
            results.append((path, str(target), ''))
        except Exception as exc:
            log.exception('batch export failed for %s', path)
            results.append((path, None, str(exc) or type(exc).__name__))
    progress(100, tr('批量导出完成'))
    return results
