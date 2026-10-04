"""Offline Chinese / English speech recognition with SenseVoice-Small (1.5.0).

The int8 ONNX export of FunAudioLLM SenseVoice-Small (sherpa-onnx packaging) runs on
ONNX Runtime's CPU provider. The front end reproduces the Kaldi 80-bin log-mel fbank
that the model was trained with (25 ms Hamming window, 10 ms shift, pre-emphasis
0.97, edges snipped), stacks 7 frames every 6 (LFR) and applies the CMVN stored in
the model's metadata. CTC output is decoded greedily.
"""
from __future__ import annotations
import re
import threading
import wave
from pathlib import Path
import numpy as np
from .i18n import tr

RATE = 16000
MODEL_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'models' / 'sensevoice'
MODEL = MODEL_DIR / 'model.int8.onnx'
TOKENS = MODEL_DIR / 'tokens.txt'
MAX_SECONDS = 60

_lock = threading.Lock()
_recognizer = None


def available():
    return MODEL.is_file() and TOKENS.is_file()


def _mel_banks(bins=80, fft=512, rate=RATE, low=20., high=0.):
    """Kaldi MelBanks: triangles on the 1127·ln(1+f/700) scale, Nyquist bin dropped."""
    high = rate / 2 + high if high <= 0 else high
    mel = lambda f: 1127. * np.log(1. + np.asarray(f, np.float64) / 700.)
    low_mel, high_mel = mel(low), mel(high)
    delta = (high_mel - low_mel) / (bins + 1)
    freqs = mel(np.arange(fft // 2) * rate / fft)
    left = low_mel + np.arange(bins)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    up = (freqs - left) / (center - left)
    down = (right - freqs) / (right - center)
    weights = np.maximum(0., np.minimum(up, down))
    weights[(freqs <= left) | (freqs >= right)] = 0.
    return weights.astype(np.float32)


_BANKS = None


def fbank(samples):
    """80-dim log-mel fbank of int16-range samples, shape (frames, 80)."""
    global _BANKS
    if _BANKS is None:
        _BANKS = _mel_banks()
    x = np.asarray(samples, np.float64)
    length, shift = 400, 160
    count = 1 + (len(x) - length) // shift if len(x) >= length else 0
    if count <= 0:
        return np.zeros((0, 80), np.float32)
    index = np.arange(length)[None, :] + shift * np.arange(count)[:, None]
    frames = x[index]
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= .97 * frames[:, :-1].copy()
    frames[:, 0] -= .97 * frames[:, 0]
    frames *= .54 - .46 * np.cos(2 * np.pi * np.arange(length) / (length - 1))
    power = np.abs(np.fft.rfft(frames, 512, axis=1)[:, :256]) ** 2
    energies = power.astype(np.float32) @ _BANKS.T
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


def lfr(features, window=7, shift=6):
    count = (len(features) - window) // shift + 1
    if count <= 0:
        return np.zeros((0, features.shape[1] * window), np.float32)
    index = np.arange(window)[None, :] + shift * np.arange(count)[:, None]
    return features[index].reshape(count, -1)


class Recognizer:
    def __init__(self, model=MODEL, tokens=TOKENS, threads=4):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.window = int(meta['lfr_window_size'])
        self.shift = int(meta['lfr_window_shift'])
        self.neg_mean = np.array(meta['neg_mean'].split(','), np.float32)
        self.inv_stddev = np.array(meta['inv_stddev'].split(','), np.float32)
        self.languages = {key[5:]: int(value) for key, value in meta.items() if key.startswith('lang_')}
        self.with_itn = int(meta['with_itn'])
        self.tokens = {}
        for line in Path(tokens).read_text(encoding='utf-8').splitlines():
            piece, _, number = line.rpartition(' ')
            if piece:
                self.tokens[int(number)] = piece

    def features(self, samples):
        stacked = lfr(fbank(samples), self.window, self.shift)
        return (stacked + self.neg_mean) * self.inv_stddev

    def transcribe(self, samples, language='auto'):
        """Samples are float in [-1, 1] or int16, mono, 16 kHz. Returns text."""
        samples = np.asarray(samples)
        if samples.dtype.kind == 'f':
            samples = samples.astype(np.float64) * 32768.
        samples = samples[:RATE * MAX_SECONDS]
        x = self.features(samples)
        if len(x) == 0:
            return ''
        logits = self.session.run(None, {
            'x': x[None].astype(np.float32),
            'x_length': np.array([len(x)], np.int32),
            'language': np.array([self.languages.get(language, self.languages['auto'])], np.int32),
            'text_norm': np.array([self.with_itn], np.int32),
        })[0][0]
        ids = logits[4:].argmax(axis=1)
        text, previous = [], -1
        for token in ids:
            token = int(token)
            if token != previous and token != 0:
                piece = self.tokens.get(token, '')
                if not (piece.startswith('<|') and piece.endswith('|>')):
                    text.append(piece)
            previous = token
        return ''.join(text).replace('\u2581', ' ').strip()


def meaningful(text):
    """Chinese or English content; noise can decode to a stray Korean or Japanese syllable."""
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', text or ''))


def recognizer():
    """Shared, lazily created recognizer (about 1 s to load on first use)."""
    global _recognizer
    with _lock:
        if _recognizer is None:
            if not available():
                raise FileNotFoundError(tr('未找到语音识别模型（assets/models/sensevoice）。'))
            _recognizer = Recognizer()
        return _recognizer


def transcribe(samples, language='auto'):
    return recognizer().transcribe(samples, language)


def read_wav(path):
    """Mono 16-bit PCM WAV at 16 kHz -> int16 samples (used by tests and tools)."""
    with wave.open(str(path), 'rb') as stream:
        if stream.getsampwidth() != 2 or stream.getframerate() != RATE:
            raise ValueError(tr('需要 16 kHz、16 位 PCM WAV。'))
        data = np.frombuffer(stream.readframes(stream.getnframes()), np.int16)
        return data.reshape(-1, stream.getnchannels())[:, 0]
