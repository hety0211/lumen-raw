"""Render the built-in watermark logos: assets/logos/src/<key>.svg → assets/logos/<key>.png.

    python tools/build_logos.py

The SVGs are text logos from Wikimedia Commons (public domain there; trademarks of their owners,
see assets/logos/README.txt and logos.json).  Each is rendered with Qt SVG at 512 pixels high,
cropped to its visible shape (Ricoh's "imagine. change." tagline is left out) and stored as an
RGBA PNG, so exports need neither Qt SVG nor fonts at run time.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'assets' / 'logos'
HEIGHT = 512


def render(svg, height=HEIGHT):
    renderer = QSvgRenderer(str(svg))
    if not renderer.isValid():
        raise ValueError(f'invalid SVG: {svg}')
    box = renderer.viewBoxF()
    width = max(1, round(height * box.width() / box.height()))
    image = QImage(width, height, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, width, height))
    painter.end()
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(height, image.bytesPerLine() // 4, 4)
    return rows[:, :width].copy()


def first_block(rgba):
    """Rows of the first band of visible pixels (drops a tagline set below a gap)."""
    filled = rgba[..., 3].max(axis=1) > 8
    start = int(np.argmax(filled))
    gap = np.flatnonzero(~filled[start:])
    return rgba[:start + gap[0]] if len(gap) else rgba


def crop(rgba, pad=4):
    alpha = rgba[..., 3] > 8
    ys, xs = np.flatnonzero(alpha.any(axis=1)), np.flatnonzero(alpha.any(axis=0))
    rgba = rgba[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    return cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))


def main():
    QGuiApplication(sys.argv[:1])
    manifest = json.loads((FOLDER / 'logos.json').read_text(encoding='utf-8'))
    for key in manifest:
        rgba = render(FOLDER / 'src' / f'{key}.svg')
        if key == 'ricoh':
            rgba = first_block(rgba)
        rgba = crop(rgba)
        if rgba.shape[0] != HEIGHT:          # cropping removed margins: back to the common height
            width = max(1, round(rgba.shape[1] * HEIGHT / rgba.shape[0]))
            rgba = cv2.resize(rgba, (width, HEIGHT), interpolation=cv2.INTER_AREA if rgba.shape[0] > HEIGHT
                              else cv2.INTER_CUBIC)
        ok, data = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        (FOLDER / f'{key}.png').write_bytes(data.tobytes())
        print(f'{key:11s} {rgba.shape[1]:5d} x {rgba.shape[0]}  {len(data) // 1024} KB')


if __name__ == '__main__':
    main()
