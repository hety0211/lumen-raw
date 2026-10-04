"""Read-only camera metadata and temperature corrections relative to as-shot WB."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from .i18n import tr


def exiftool():
    """Command prefix for the bundled ExifTool, or ``None`` when it is not installed.

    Windows uses the standalone ``exiftool.exe``.  macOS runs the pure-Perl
    Image-ExifTool distribution in ``assets/exiftool/unix`` with the system Perl.
    """
    folder = Path(__file__).resolve().parents[1] / 'assets' / 'exiftool'
    if os.name == 'nt':
        exe = folder / 'exiftool.exe'
        return [str(exe)] if exe.exists() else None
    script = folder / 'unix' / 'exiftool'
    perl = '/usr/bin/perl' if Path('/usr/bin/perl').exists() else shutil.which('perl')
    return [perl, str(script)] if perl and script.exists() else None


def metadata(path):
    command = exiftool()
    if command is None:
        return {}, tr('未安装元数据读取器')
    try:
        env = dict(os.environ, LC_ALL='C', LANG='C', LC_CTYPE='C')
        # -config disables per-user Perl config execution. Absolute filename and --
        # keep filenames from being interpreted as options. This never writes tags.
        result = subprocess.run([*command, '-config', '', '-j', '-charset', 'filename=UTF8',
            '-ColorTemperature#', '-WhiteBalance#', '-Make', '-Model', '-LensMake', '-LensModel', '-Lens', '-LensID',
            '-DateTimeOriginal', '-CreateDate', '-FNumber#', '-ExposureTime#', '-ISO#', '-FocalLength#',
            # 1.5.1: lens profile matching (crop factor, focus distance).
            '-FocalLengthIn35mmFormat#', '-ScaleFactor35efl#', '-FocusDistance#', '-FocusDistance2#',
            '-ApproximateFocusDistance#', '-SubjectDistance#', '--', str(Path(path).resolve())],
            capture_output=True, timeout=15, env=env,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            return {}, tr('元数据读取失败；使用相机白平衡增益')
        return json.loads(result.stdout.decode('utf-8'))[0], ''
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}, tr('元数据不可用；使用相机白平衡增益')


def from_metadata(tags, estimated=None):
    value = tags.get('ColorTemperature')
    if isinstance(value, (int, float)) and 1500 <= value <= 25000:
        return dict(camera_kelvin=float(value), kelvin=float(value), estimated=False)
    value = estimated if estimated is not None and 1500 <= estimated <= 25000 else None
    return dict(camera_kelvin=value, kelvin=value, estimated=value is not None)


def estimate(raw):
    """Approximate CCT from LibRaw's XYZ-to-camera calibration and as-shot gains.

    This is an estimate, never a substitute for a recorded camera Kelvin tag.
    McCamy's approximation is restricted to its useful photographic interval.
    """
    try:
        gains = np.asarray(raw.camera_whitebalance[:3], dtype=float)
        if np.min(gains) <= 0:
            return None
        xyz = np.linalg.solve(np.asarray(raw.rgb_xyz_matrix[:3], dtype=float), 1 / gains)
        if np.min(xyz) <= 0:
            return None
        x, y = (xyz / xyz.sum())[:2]
        if not (.22 < x < .55 and .23 < y < .48):
            return None
        n = (x - .3320) / (.1858 - y)
        t = 449*n**3 + 3525*n**2 + 6823.3*n + 5520.33
        return float(round(t / 10) * 10) if 2500 <= t <= 10000 else None
    except (ValueError, np.linalg.LinAlgError, AttributeError):
        return None


def illuminant_rgb(kelvin):
    # Planckian chromaticity approximation, normalized to a neutral green channel.
    t = float(np.clip(kelvin, 1667, 25000))
    x = (-.2661239e9/t**3 - .2343580e6/t**2 + .8776956e3/t + .179910) if t <= 4000 else (-3.0258469e9/t**3 + 2.1070379e6/t**2 + .2226347e3/t + .240390)
    if t <= 2222:
        y = -1.1063814*x**3 - 1.34811020*x**2 + 2.18555832*x - .20219683
    elif t <= 4000:
        y = -.9549476*x**3 - 1.37418593*x**2 + 2.09137015*x - .16748867
    else:
        y = 3.0817580*x**3 - 5.87338670*x**2 + 3.75112997*x - .37001483
    rgb = np.array([[3.2406,-1.5372,-.4986],[-.9689,1.8758,.0415],[.0557,-.2040,1.0570]]) @ np.array([x/y, 1., (1-x-y)/y])
    rgb = np.maximum(rgb, .025)
    return rgb / rgb[1]


def gains(settings):
    a, b = settings.get('camera_kelvin'), settings.get('kelvin')
    if a is None or b is None or a == b:
        return np.ones(3, np.float32)
    return np.clip(illuminant_rgb(a) / illuminant_rgb(b), .125, 8).astype(np.float32)
