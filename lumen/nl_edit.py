"""Natural-language editing (1.5.0): describe a look, a language model returns slider values.

The model is told the fixed set of adjustable parameters (``PARAMS``, ``HSL_COLORS``,
grading, effects, an RGB curve and region masks) with their ranges and the photo's
current values. It answers with one JSON object of *final* values; ``plan`` clamps
and applies them to a copy of the recipe, so a reply can never produce an invalid
project. Requests use only the standard library and go either to a local runner
(Ollama, LM Studio, llama.cpp, any OpenAI-compatible server) or to a cloud API the
user has a key for.
"""
from __future__ import annotations
import base64
import copy
import io
import ipaddress
import json
import math
import os
import re
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
from . import model
from .i18n import tr, N_

# --------------------------------------------------------------------------- providers

#: id: (label, api, default base URL, suggested model, local, JSON mode, thinking style).
#: Base URLs, model names and thinking parameters follow each provider's documentation
#: as of 2026-10; LM Studio's behaviour was checked against a running server.
PROVIDERS = {
    'ollama': ('Ollama', 'ollama', 'http://127.0.0.1:11434', '', True, 'json', 'ollama'),
    'lmstudio': ('LM Studio', 'openai', 'http://127.0.0.1:1234/v1', '', True, None, 'lmstudio'),
    'llamacpp': (N_('llama.cpp（llama-server）'), 'openai', 'http://127.0.0.1:8080/v1', '', True, 'json_object', 'llamacpp'),
    'local': (N_('其他本地 OpenAI 兼容服务（vLLM、Jan、LocalAI…）'), 'openai', 'http://127.0.0.1:8000/v1', '', True, None, 'template'),
    'openai': ('OpenAI', 'openai', 'https://api.openai.com/v1', 'gpt-6-luna', False, 'json_object', 'openai'),
    'anthropic': ('Anthropic Claude', 'anthropic', 'https://api.anthropic.com', 'claude-opus-5-5', False, None, 'anthropic'),
    'gemini': ('Google Gemini', 'openai', 'https://generativelanguage.googleapis.com/v1beta/openai', 'gemini-3.8-flash', False, 'json_object', 'gemini'),
    'deepseek': (N_('DeepSeek 深度求索'), 'openai', 'https://api.deepseek.com', 'deepseek-flash', False, 'json_object', 'deepseek'),
    'dashscope': (N_('阿里云百炼 · 通义千问'), 'openai', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'qwen3.7-flash', False, 'json_object', 'qwen'),
    'moonshot': (N_('月之暗面 Kimi'), 'openai', 'https://api.moonshot.cn/v1', 'kimi-k2.6', False, 'json_object', 'kimi'),
    'zhipu': (N_('智谱 GLM'), 'openai', 'https://open.bigmodel.cn/api/paas/v4', 'glm-5.3-flash', False, 'json_object', 'switch'),
    'siliconflow': (N_('硅基流动 SiliconFlow'), 'openai', 'https://api.siliconflow.cn/v1', 'deepseek-ai/DeepSeek-V4-Flash', False, 'json_object', 'budget'),
    'volcengine': (N_('火山方舟 · 豆包'), 'openai', 'https://ark.cn-beijing.volces.com/api/v3', 'doubao-seed-2-0-lite-260428', False, 'json_object', 'ark'),
    'openrouter': ('OpenRouter', 'openai', 'https://openrouter.ai/api/v1', '', False, 'json_object', 'openrouter'),
    'custom': (N_('自定义 OpenAI 兼容接口'), 'openai', '', '', False, None, 'openai'),
}
LOCAL_PROVIDERS = [k for k, v in PROVIDERS.items() if v[4]]
CLOUD_PROVIDERS = [k for k, v in PROVIDERS.items() if not v[4]]
#: Defaults written by earlier 1.5.0 builds; still-untouched values move to the current ones.
LEGACY = {'deepseek': ('https://api.deepseek.com/v1', 'deepseek-chat'), 'openai': (None, 'gpt-4.1-mini'),
          'anthropic': (None, 'claude-sonnet-5-5'), 'gemini': (None, 'gemini-2.5-flash'), 'dashscope': (None, 'qwen-plus'),
          'moonshot': (None, 'moonshot-v1-8k'), 'zhipu': (None, 'glm-4-flash'),
          'siliconflow': (None, 'Qwen/Qwen2.5-7B-Instruct'), 'volcengine': (None, '')}

#: Thinking depth chosen in the settings; mapped per provider by ``thinking_request``.
REASONING = {'default': N_('默认（由模型决定）'), 'off': N_('关闭（最快、最省）'), 'low': N_('低'), 'medium': N_('中'),
             'high': N_('高'), 'max': N_('最高')}
_BUDGET = {'low': 1024, 'medium': 4096, 'high': 16384, 'max': 32768}
THINKING_NOTES = {
    'ollama': N_('Ollama：关闭 = think:false；低 / 中 / 高 = think 档位（如 gpt-oss），模型不支持档位时改用默认。'),
    'lmstudio': N_('LM Studio：关闭 = reasoning_effort none（qwen3 实测可关闭思考）；低 / 中 / 高 = reasoning_effort。'),
    'llamacpp': N_('llama.cpp：关闭 = reasoning_effort none + enable_thinking false；其余 = reasoning_effort。'),
    'template': N_('关闭 = chat_template_kwargs.enable_thinking=false；其余 = reasoning_effort。'),
    'openai': N_('reasoning_effort（关闭 = none；模型不支持时自动改用默认）。'),
    'anthropic': N_('output_config.effort。当前 Claude 无法关闭思考，“关闭”按 low 处理；Opus 5.5 默认 medium。'),
    'gemini': N_('reasoning_effort（关闭 = minimal，“最高”按 high）。'),
    'deepseek': N_('thinking.type + reasoning_effort。DeepSeek 默认开启思考（high），只有 low / high / max 三档，“中”按 high。'),
    'qwen': N_('enable_thinking + thinking_budget（低 1024 / 中 4096 / 高 16384 / 最高 32768 token）。开启思考时百炼不支持 JSON 模式，改由提示词约束。'),
    'budget': N_('enable_thinking + thinking_budget（低 1024 / 中 4096 / 高 16384 / 最高 32768 token）。'),
    'kimi': N_('kimi-k3 始终思考，用 reasoning_effort（low / high / max）；kimi-k2.x 用 thinking 开关。'),
    'switch': N_('智谱 thinking.type 开关：GLM 没有档位，低 / 中 / 高 / 最高都为开启。'),
    'ark': N_('thinking.type + reasoning_effort（低 / 中 / 高，“最高”按 high）。'),
    'openrouter': N_('reasoning.effort，由 OpenRouter 转换为各家参数。'),
}


def provider_label(key):
    return tr(PROVIDERS.get(key, PROVIDERS['custom'])[0])


def default_settings():
    return dict(provider='ollama', attach_image=False, timeout=180, voice_language='auto', auto_send=True, reasoning='default',
                providers={key: dict(base_url=v[2], model=v[3], key='') for key, v in PROVIDERS.items()})


def thinking_request(style, level, model=''):
    """Request fields that set the thinking depth for one provider; {} keeps the model's default."""
    if level not in REASONING or level == 'default':
        return {}
    off, capped = level == 'off', 'high' if level == 'max' else level
    if style == 'openai':
        return {'reasoning_effort': 'none' if off else level}
    if style == 'lmstudio':
        return {'reasoning_effort': 'none' if off else capped}
    if style == 'llamacpp':
        return {'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False}} if off else {'reasoning_effort': capped}
    if style == 'template':
        return {'chat_template_kwargs': {'enable_thinking': False}} if off else {'reasoning_effort': capped}
    if style == 'anthropic':
        return {'output_config': {'effort': 'low' if off else level}}
    if style == 'gemini':
        return {'reasoning_effort': 'minimal' if off else capped}
    if style == 'deepseek':
        if off:
            return {'thinking': {'type': 'disabled'}}
        return {'thinking': {'type': 'enabled'}, 'reasoning_effort': {'low': 'low', 'medium': 'high', 'high': 'high', 'max': 'max'}[level]}
    if style in ('qwen', 'budget'):
        return {'enable_thinking': False} if off else {'enable_thinking': True, 'thinking_budget': _BUDGET[level]}
    if style == 'kimi':
        if model.lower().startswith('kimi-k3'):
            return {'reasoning_effort': {'off': 'low', 'low': 'low', 'medium': 'high', 'high': 'high', 'max': 'max'}[level]}
        return {'thinking': {'type': 'disabled' if off else 'enabled'}}
    if style == 'switch':
        return {'thinking': {'type': 'disabled' if off else 'enabled'}}
    if style == 'ark':
        return {'thinking': {'type': 'disabled'}} if off else {'thinking': {'type': 'enabled'}, 'reasoning_effort': capped}
    if style == 'openrouter':
        return {'reasoning': {'effort': 'none' if off else level}}
    if style == 'ollama':
        return {'think': False if off else capped}
    return {}


def normalize_base(api, url):
    """Accept a pasted endpoint as well as a base URL (…/chat/completions, …/v1/messages, …/api/chat)."""
    url = url.strip().rstrip('/')
    for suffix in {'ollama': ('/api/chat', '/api', '/v1'), 'anthropic': ('/v1/messages', '/messages', '/v1')}.get(api, ('/chat/completions',)):
        if url.lower().endswith(suffix):
            url = url[:-len(suffix)].rstrip('/')
    return url


def settings_path():
    from . import host
    return host.data_folder() / 'natural-language.json'


# API keys are encrypted for the current Windows user with DPAPI. On macOS one
# generic-password item in the login keychain holds every provider's key, so an updated
# (newly signed) app asks for keychain access once. Elsewhere, or when the keychain
# refuses, keys are stored as given in a file readable only by the user.
KEYCHAIN = sys.platform == 'darwin'
#: (service, account) of the keychain item; the value is a JSON object provider -> key.
KEYCHAIN_ITEM = (b'LUMEN RAW', b'api-keys')
_keychain_cache = None  # keys last read from or written to the keychain
_NOT_FOUND = -25300  # errSecItemNotFound


def _security():
    import ctypes
    from ctypes import POINTER, c_char_p, c_int32, c_uint32, c_void_p
    lib = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/Security.framework/Security')
    cf = ctypes.cdll.LoadLibrary('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    lib.SecKeychainFindGenericPassword.argtypes = [c_void_p, c_uint32, c_char_p, c_uint32, c_char_p,
                                                   POINTER(c_uint32), POINTER(c_void_p), POINTER(c_void_p)]
    lib.SecKeychainAddGenericPassword.argtypes = [c_void_p, c_uint32, c_char_p, c_uint32, c_char_p,
                                                  c_uint32, c_char_p, POINTER(c_void_p)]
    lib.SecKeychainItemModifyAttributesAndData.argtypes = [c_void_p, c_void_p, c_uint32, c_char_p]
    lib.SecKeychainItemFreeContent.argtypes = [c_void_p, c_void_p]
    for function in (lib.SecKeychainFindGenericPassword, lib.SecKeychainAddGenericPassword,
                     lib.SecKeychainItemModifyAttributesAndData, lib.SecKeychainItemFreeContent):
        function.restype = c_int32
    cf.CFRelease.argtypes = [c_void_p]
    return ctypes, lib, cf


def _keychain_read():
    """The item's data, or None when there is no item."""
    ctypes, lib, cf = _security()
    service, account = KEYCHAIN_ITEM
    length, data, item = ctypes.c_uint32(), ctypes.c_void_p(), ctypes.c_void_p()
    status = lib.SecKeychainFindGenericPassword(None, len(service), service, len(account), account,
                                                ctypes.byref(length), ctypes.byref(data), ctypes.byref(item))
    if status == _NOT_FOUND:
        return None
    if status:
        raise OSError(tr('钥匙串拒绝读取 API Key（{status}）。', status=status))
    try:
        return ctypes.string_at(data, length.value)
    finally:
        lib.SecKeychainItemFreeContent(None, data)
        cf.CFRelease(item)


def _keychain_write(value):
    ctypes, lib, cf = _security()
    service, account = KEYCHAIN_ITEM
    item = ctypes.c_void_p()
    status = lib.SecKeychainFindGenericPassword(None, len(service), service, len(account), account,
                                                None, None, ctypes.byref(item))
    if status == _NOT_FOUND:
        status = lib.SecKeychainAddGenericPassword(None, len(service), service, len(account), account,
                                                   len(value), value, None)
    elif not status:
        try:
            status = lib.SecKeychainItemModifyAttributesAndData(item, None, len(value), value)
        finally:
            cf.CFRelease(item)
    if status:
        raise OSError(tr('钥匙串拒绝保存 API Key（{status}）。', status=status))


def keychain_keys():
    """provider -> key from the keychain item; empty when it is missing or access is denied."""
    global _keychain_cache
    try:
        data = _keychain_read()
        keys = json.loads(data.decode('utf-8')) if data else {}
    except (OSError, ValueError):
        return {}
    _keychain_cache = {str(k): str(v) for k, v in keys.items()} if isinstance(keys, dict) else {}
    return dict(_keychain_cache)


def keychain_store(keys):
    """Store provider -> key in the keychain item; False when the keychain refuses."""
    global _keychain_cache
    if keys == _keychain_cache or (not keys and _keychain_cache is None):
        return True
    try:
        _keychain_write(json.dumps(keys, ensure_ascii=False).encode('utf-8'))
    except OSError:
        return False
    _keychain_cache = dict(keys)
    return True


def _dpapi(data, protect):
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_char))]

    buffer = ctypes.create_string_buffer(data, len(data))
    source, target = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), Blob()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    call = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not call(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(target)):
        raise OSError(tr('无法使用 Windows 数据保护读取 API Key。'))
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)


def protect_key(key):
    if not key:
        return ''
    if os.name == 'nt':
        return 'dpapi:' + base64.b64encode(_dpapi(key.encode('utf-8'), True)).decode('ascii')
    return 'plain:' + key


def unprotect_key(value):
    if not value:
        return ''
    kind, _, data = value.partition(':')
    if kind == 'dpapi' and os.name == 'nt':
        try:
            return _dpapi(base64.b64decode(data), False).decode('utf-8')
        except (OSError, ValueError):
            return ''
    return data if kind == 'plain' else ''


def load_settings(path=None):
    result = default_settings()
    try:
        data = json.loads(Path(path or settings_path()).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return result
    if data.get('provider') in PROVIDERS:
        result['provider'] = data['provider']
    result['attach_image'] = bool(data.get('attach_image', False))
    result['auto_send'] = bool(data.get('auto_send', True))
    result['voice_language'] = data.get('voice_language') if data.get('voice_language') in ('auto', 'zh', 'en') else 'auto'
    result['reasoning'] = data.get('reasoning') if data.get('reasoning') in REASONING else 'default'
    try:
        result['timeout'] = int(min(900, max(10, float(data.get('timeout', 180)))))
    except (TypeError, ValueError):
        pass
    in_keychain = []
    for key, stored in (data.get('providers') or {}).items():
        if key in PROVIDERS and isinstance(stored, dict):
            entry = result['providers'][key]
            entry['base_url'] = str(stored.get('base_url', entry['base_url']))[:500]
            entry['model'] = str(stored.get('model', entry['model']))[:200]
            entry['key'] = unprotect_key(str(stored.get('key', '')))
            if stored.get('key') == 'keychain:':
                in_keychain.append(key)
            old_base, old_model = LEGACY.get(key, (None, None))
            if old_base is not None and entry['base_url'].rstrip('/') == old_base:
                entry['base_url'] = PROVIDERS[key][2]
            if old_model is not None and entry['model'] == old_model:
                entry['model'] = PROVIDERS[key][3]
    if in_keychain and KEYCHAIN:
        keys = keychain_keys()
        for key in in_keychain:
            result['providers'][key]['key'] = keys.get(key, '')
    return result


def save_settings(settings, path=None):
    path = Path(path or settings_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(settings)
    keys = {name: entry['key'] for name, entry in data['providers'].items() if entry.get('key')}
    in_keychain = KEYCHAIN and keychain_store(keys)
    for entry in data['providers'].values():
        entry['key'] = 'keychain:' if in_keychain and entry.get('key') else protect_key(entry.get('key', ''))
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    if os.name != 'nt':
        os.chmod(temporary, 0o600)
    temporary.replace(path)


def active(settings):
    """The chosen service: provider id, api, base URL, model, key, local, JSON mode and thinking setup."""
    key = settings['provider']
    label, api, _, _, local, json_mode, style = PROVIDERS[key]
    entry = settings['providers'][key]
    reasoning = settings.get('reasoning', 'default')
    model = entry['model'].strip()
    if style == 'qwen' and reasoning != 'off':
        json_mode = None  # Bailian: JSON mode is not available while the model thinks
    return dict(provider=key, label=tr(label), api=api, base_url=normalize_base(api, entry['base_url']),
                model=model, key=entry['key'].strip(), local=local, json_mode=json_mode,
                timeout=settings.get('timeout', 180), reasoning=reasoning, thinking=thinking_request(style, reasoning, model))


# --------------------------------------------------------------------------- parameters

#: Global sliders: key -> (name, low, high, meaning). Same ranges as ``model.validate``.
PARAMS = {
    'exposure': (N_('曝光'), -5, 5, '整体亮度，单位 EV；+1 为亮一倍，日常修正多在 ±1 以内'),
    'contrast': (N_('对比度'), -100, 100, '正值增加反差，负值更柔和'),
    'highlights': (N_('亮部'), -100, 100, '负值压暗高光、找回天空和亮处细节，正值提亮亮部'),
    'shadows': (N_('暗部'), -100, 100, '正值提亮阴影细节，负值压暗阴影'),
    'whites': (N_('白色'), -100, 100, '白点；正值让最亮处更亮，负值防止过曝'),
    'blacks': (N_('黑色'), -100, 100, '黑点；负值加深最暗处，正值让黑色发灰（胶片感）'),
    'temperature': (N_('冷暖'), -100, 100, '相对相机白平衡；正值更暖（偏黄），负值更冷（偏蓝），±20 已较明显'),
    'tint': (N_('色调'), -100, 100, '正值偏洋红，负值偏绿'),
    'saturation': (N_('饱和度'), -100, 100, '所有颜色的浓度；-100 为黑白'),
    'vibrance': (N_('自然饱和度'), -100, 100, '优先提升低饱和颜色，对肤色更温和'),
    'dehaze': (N_('去薄雾'), -100, 100, '正值去雾、通透；负值加雾、柔和梦幻'),
    'clarity': (N_('清晰度'), -100, 100, '中间调局部对比；正值更有力度，负值柔焦'),
    'texture': (N_('纹理'), -100, 100, '细小纹理；正值强调细节，负值柔化皮肤'),
    'sharpness': (N_('锐化'), 0, 100, '输出锐化强度'),
    'denoise': (N_('明度降噪'), 0, 100, '减少亮度噪点；高 ISO 照片可用 20–50'),
    'color_noise': (N_('彩色杂点'), 0, 100, '减少彩色噪点；高 ISO 照片可用 25–50'),
}
EFFECTS = {
    'vignette': (N_('暗角'), -100, 100, '负值压暗四周，正值提亮四周'),
    'midpoint': (N_('暗角中点'), 0, 100, '暗角范围，越小越大，默认 50'),
    'feather': (N_('暗角羽化'), 0, 100, '过渡柔和度，默认 70'),
    'grain': (N_('颗粒'), 0, 100, '胶片颗粒数量'),
    'grain_size': (N_('颗粒大小'), 0, 100, '默认 30'),
}
#: (key, name, aliases). Order and hues follow ``model.COLORS``.
HSL_COLORS = [('red', N_('红'), ('红色',)), ('orange', N_('橙'), ('橙色', '肤色')), ('yellow', N_('黄'), ('黄色',)),
              ('green', N_('绿'), ('绿色',)), ('aqua', N_('青'), ('cyan', '青色')), ('blue', N_('蓝'), ('蓝色',)),
              ('purple', N_('紫'), ('紫色', 'violet')), ('magenta', N_('洋红'), ('洋红色', 'pink', '粉'))]
HSL_PARTS = [('hue', N_('色相'), ('h', '色相')), ('saturation', N_('饱和度'), ('s', 'sat', '饱和度')),
             ('luminance', N_('明度'), ('l', 'lightness', 'lum', 'brightness', '明度', '亮度'))]
ZONES = [('shadows', N_('暗部')), ('midtones', N_('中间调')), ('highlights', N_('高光'))]
#: Mask regions: AI selections (computed locally) and simple gradients.
REGIONS = {
    'sky': (N_('天空'), 'sky'), 'subject': (N_('主体'), 'subject'), 'person': (N_('人物'), 'person'),
    'background': (N_('背景'), 'background'), 'foreground': (N_('近景'), 'foreground'),
    'top': (N_('上方渐变'), 'linear'), 'bottom': (N_('下方渐变'), 'linear'),
    'center': (N_('中心径向'), 'radial'), 'edges': (N_('四周径向'), 'radial'),
}
AI_REGIONS = ('sky', 'subject', 'person', 'background', 'foreground')
MAX_NEW_MASKS = 4
ALIASES = {
    '曝光': 'exposure', 'brightness': 'exposure', '亮度': 'exposure', '对比度': 'contrast', '对比': 'contrast',
    '高光': 'highlights', '亮部': 'highlights', '阴影': 'shadows', '暗部': 'shadows', '白色': 'whites',
    'white': 'whites', '黑色': 'blacks', 'black': 'blacks', '色温': 'temperature', '冷暖': 'temperature',
    'temp': 'temperature', 'warmth': 'temperature', '色调': 'tint', '饱和度': 'saturation',
    '自然饱和度': 'vibrance', '去薄雾': 'dehaze', '去雾': 'dehaze', 'haze': 'dehaze', '清晰度': 'clarity',
    '纹理': 'texture', '锐化': 'sharpness', 'sharpen': 'sharpness', 'sharpening': 'sharpness',
    '降噪': 'denoise', '明度降噪': 'denoise', 'noise_reduction': 'denoise', 'luminance_noise': 'denoise',
    '彩色杂点': 'color_noise', '彩色降噪': 'color_noise', 'colour_noise': 'color_noise',
    '暗角': 'vignette', '颗粒': 'grain', 'grain_amount': 'grain',
}


def _parameter_table():
    rows = [f'- {k}（{n}）{lo}…{hi}：{d}' for k, (n, lo, hi, d) in PARAMS.items()]
    effects = [f'- {k}（{n}）{lo}…{hi}：{d}' for k, (n, lo, hi, d) in EFFECTS.items()]
    colors = '、'.join(f'{k}（{n}）' for k, n, _ in HSL_COLORS)
    regions = '、'.join(f'{k}（{v[0]}）' for k, v in REGIONS.items())
    return '\n'.join([
        'adjustments（全局滑块，默认都是 0）：', *rows,
        '',
        f'hsl（八色混合器）：颜色 {colors}；每种颜色有 hue、saturation、luminance，范围 -100…100，默认 0。'
        'hue +100 约让该颜色的色相角增加 45°（红→橙→黄→绿→青→蓝→紫→洋红方向），saturation/luminance 正值更浓/更亮。',
        '',
        'grading（三段色彩分级）：shadows、midtones、highlights 各为 {"hue": 0…360, "saturation": 0…100}，'
        'saturation 为 0 表示不着色，常用 5–30；balance -100…100 把分界偏向暗部(负)或高光(正)。'
        '色相参考：0 红、30 橙、45 金、60 黄、120 绿、180 青、210 青蓝、240 蓝、280 紫、320 洋红。',
        '',
        'effects（效果）：', *effects,
        '',
        'monochrome：true / false，黑白处理。',
        '',
        'curve（RGB 色调曲线，可选）：3–8 个 [输入, 输出] 点，0…255 整数，必须包含 [0, y] 和 [255, y]；'
        '例如 [[0,20],[64,60],[192,200],[255,245]] 是黑位抬起、高光略压的胶片感。不需要时不要输出。',
        '',
        f'masks（局部调整，可选，最多 {MAX_NEW_MASKS} 个新蒙版）：区域 region 可选 {regions}。'
        '天空、主体、人物、背景、近景由本地 AI 自动识别；top/bottom 是从画面上方/下方向中间过渡的渐变，'
        'center 是中心椭圆，edges 是中心以外的四周。每个蒙版写 {"region": "sky", "adjustments": {…}}，'
        'adjustments 只用上面的全局滑块键，数值是该区域在全局调整之外再叠加的量。'
        '若要修改当前已有的蒙版，写 {"index": 序号, "adjustments": {…}}（最终值）。',
    ])


SYSTEM_PROMPT = '''你是 LUMEN RAW 修图软件里的调色助手。用户用一句话描述想要的效果，你只能通过下面列出的固定参数来实现，并且只返回一个 JSON 对象。

可调参数：
{table}

返回格式（只包含需要改变的键；数值都是调整后的最终值，不是增量）：
{{
  "explanation": "一两句话说明你做了什么，使用用户的语言",
  "reset": false,
  "adjustments": {{"exposure": 0.3, "highlights": -40}},
  "hsl": {{"blue": {{"saturation": 20, "luminance": -15}}}},
  "grading": {{"shadows": {{"hue": 210, "saturation": 12}}, "highlights": {{"hue": 40, "saturation": 15}}}},
  "effects": {{"vignette": -15}},
  "monochrome": false,
  "curve": [[0, 0], [255, 255]],
  "masks": [{{"region": "sky", "adjustments": {{"exposure": -0.4, "saturation": 15}}}}]
}}

规则：
1. 数值是最终值。当前值在用户消息里给出；“再亮一点”“更暖一些”表示在当前值上小幅继续（通常滑块 10–20、曝光 0.2–0.4 EV）。
2. 只改与指令相关的参数；用户要整体风格（如日系、电影感、胶片、黑白、通透）时可以组合多个参数。保持自然，避免极端值，除非用户明确要求。
3. reset 为 true 表示先把全局调色（滑块、八色、曲线、分级、效果、黑白）恢复默认再应用你的值；只有用户说“重新调”“从头开始”“恢复原样”等时才使用。
4. 软件做不到的事情（移除物体、换天空、改变构图等）在 explanation 里说明，并尽量用现有参数接近。
5. 参考照片统计信息判断曝光和偏色：中位亮度约 0.35–0.5 较为正常；高光溢出比例高时降低 highlights/whites，暗部死黑比例高时提高 shadows/blacks。
6. 只输出 JSON，不要 Markdown 代码块，不要其他文字。'''


def system_prompt():
    return SYSTEM_PROMPT.format(table=_parameter_table())


# --------------------------------------------------------------------------- context

def image_statistics(rgb):
    """A compact text description of the displayed (sRGB, 0–1) preview for text-only models."""
    rgb = np.asarray(rgb, np.float32)
    step = max(1, int(math.sqrt(rgb.shape[0] * rgb.shape[1] / 250000)))
    x = np.clip(rgb[::step, ::step, :3], 0, 1)
    lum = x[..., 0] * .2126 + x[..., 1] * .7152 + x[..., 2] * .0722
    p = np.percentile(lum, [1, 5, 50, 95, 99])
    high, low = x.max(axis=2), x.min(axis=2)
    saturation = np.where(high > 1e-4, (high - low) / np.maximum(high, 1e-4), 0)
    mean = x.reshape(-1, 3).mean(axis=0)
    thirds = np.array_split(lum, 3, axis=0)
    cast = []
    if mean[0] - mean[2] > .04:
        cast.append('整体偏暖')
    elif mean[2] - mean[0] > .04:
        cast.append('整体偏冷')
    if mean[1] - (mean[0] + mean[2]) / 2 > .03:
        cast.append('偏绿')
    elif (mean[0] + mean[2]) / 2 - mean[1] > .04:
        cast.append('偏洋红')
    return {
        '亮度分位（0–1）': {'1%': round(float(p[0]), 3), '5%': round(float(p[1]), 3), '中位': round(float(p[2]), 3),
                         '95%': round(float(p[3]), 3), '99%': round(float(p[4]), 3)},
        '高光溢出比例': round(float((high >= .985).mean()), 4),
        '暗部死黑比例': round(float((high <= .015).mean()), 4),
        '平均饱和度（0–1）': round(float(saturation.mean()), 3),
        '平均 RGB': [round(float(v), 3) for v in mean],
        '上/中/下三分之一平均亮度': [round(float(t.mean()), 3) for t in thirds],
        '色彩倾向': '、'.join(cast) or '基本中性',
    }


def photo_context(info):
    photo = (info or {}).get('photo') or {}
    context = {k: v for k, v in (('格式', info.get('format')), ('尺寸', f'{info.get("width")} × {info.get("height")}'),
               ('相机', photo.get('body')), ('镜头', photo.get('lens')), ('光圈', photo.get('aperture')),
               ('快门', photo.get('shutter')), ('ISO', photo.get('iso')), ('焦距', photo.get('focal'))) if v}
    date = photo.get('date', '')
    if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}', date):
        context['拍摄时刻'] = date[11:16]
    return context


def _rounded(value):
    return round(float(value), 2) if isinstance(value, float) else value


def current_state(edits):
    """Current values in the reply's own vocabulary (non-default parts only, except the sliders)."""
    state = {'adjustments': {k: _rounded(edits['adjustments'][k]) for k in PARAMS}}
    hsl = {}
    for (key, _, _), row in zip(HSL_COLORS, edits['hsl']):
        parts = {name: _rounded(v) for (name, _, _), v in zip(HSL_PARTS, row) if v}
        if parts:
            hsl[key] = parts
    if hsl:
        state['hsl'] = hsl
    grading = {z: {'hue': _rounded(edits['grading'][z][0]), 'saturation': _rounded(edits['grading'][z][1])}
               for z, _ in ZONES if edits['grading'][z][1]}
    if grading or edits['grading']['balance']:
        state['grading'] = dict(grading, balance=_rounded(edits['grading']['balance']))
    defaults = model.effects()
    effects = {k: _rounded(v) for k, v in edits['effects'].items() if v != defaults[k]}
    if effects:
        state['effects'] = effects
    state['monochrome'] = bool(edits['monochrome'])
    curve = edits['curves']['RGB']
    if curve != [[0., 0.], [1., 1.]]:
        state['curve'] = [[round(x * 255), round(y * 255)] for x, y in curve]
    masks = []
    for index, mask in enumerate(edits['masks']):
        masks.append(dict(index=index, name=mask['name'], kind=mask['kind'], enabled=mask['enabled'],
                          adjustments={k: _rounded(v) for k, v in mask['adjustments'].items() if v}))
    if masks:
        state['masks'] = masks
    return state


def preview_jpeg(rgb, size=768, quality=85):
    """Base64 JPEG of the displayed preview for models that accept images."""
    from PIL import Image
    image = Image.fromarray((np.clip(np.asarray(rgb)[..., :3], 0, 1) * 255 + .5).astype(np.uint8))
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, 'JPEG', quality=quality)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def user_message(instruction, edits, info=None, stats=None):
    parts = []
    context = photo_context(info or {})
    if context:
        parts.append('照片信息：' + json.dumps(context, ensure_ascii=False))
    if stats:
        parts.append('当前效果的统计信息：' + json.dumps(stats, ensure_ascii=False))
    parts.append('当前参数：' + json.dumps(current_state(edits), ensure_ascii=False))
    parts.append('用户指令：' + instruction.strip())
    return '\n'.join(parts)


# --------------------------------------------------------------------------- requests

class ServiceError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _is_local(url):
    host = (urllib.parse.urlsplit(url).hostname or '').lower()
    if host in ('localhost',) or host.endswith('.localhost') or host.endswith('.local'):
        return True
    try:
        address = ipaddress.ip_address(host)
        return address.is_loopback or address.is_private or address.is_link_local
    except ValueError:
        return False


def _http(url, payload=None, headers=None, timeout=60, method=None):
    if not url.startswith(('http://', 'https://')):
        raise ServiceError(tr('接口地址需要以 http:// 或 https:// 开头。'))
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method or ('POST' if data else 'GET'))
    request.add_header('User-Agent', 'LUMEN-RAW')
    if data is not None:
        request.add_header('Content-Type', 'application/json')
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    # Local runners must not be routed through a system proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if _is_local(url) else urllib.request.build_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(32 * 1024 * 1024)
    except urllib.error.HTTPError as error:
        detail = error.read(4000).decode('utf-8', 'replace')
        raise ServiceError(_http_message(error.code, detail), error.code) from None
    except urllib.error.URLError as error:
        reason = error.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ServiceError(tr('连接 {url} 超时。', url=url)) from None
        if isinstance(reason, ConnectionRefusedError) or 'refused' in str(reason).lower() or '积极拒绝' in str(reason):
            raise ServiceError(tr('无法连接 {origin}：服务未启动，或端口不对。'
                                  '本地模型请先在 Ollama / LM Studio / llama.cpp 中启动服务。', origin=_origin(url))) from None
        raise ServiceError(tr('无法连接 {origin}：{reason}', origin=_origin(url), reason=reason)) from None
    except (socket.timeout, TimeoutError):
        raise ServiceError(tr('等待 {origin} 回复超时，可在设置中延长超时时间。', origin=_origin(url))) from None
    try:
        return json.loads(body.decode('utf-8'))
    except ValueError:
        raise ServiceError(tr('服务返回的不是 JSON：') + body[:300].decode('utf-8', 'replace')) from None


def _origin(url):
    parts = urllib.parse.urlsplit(url)
    return f'{parts.scheme}://{parts.netloc}'


def _http_message(code, detail):
    try:
        data = json.loads(detail)
        error = data.get('error', data)
        text = error.get('message') if isinstance(error, dict) else str(error)
        detail = text or detail
    except (ValueError, AttributeError):
        pass
    detail = ' '.join(str(detail).split())[:400]
    hint = {401: tr('API Key 无效或未填写。'), 403: tr('没有访问该模型的权限，或账户未开通。'),
            404: tr('接口地址或模型名称不正确。'), 402: tr('账户余额不足。'), 429: tr('请求太频繁或额度已用完。')}.get(code, '')
    return tr('服务返回 HTTP {code}。', code=code) + hint + detail


def _openai_headers(service):
    headers = {}
    if service['key']:
        headers['Authorization'] = 'Bearer ' + service['key']
    if service['provider'] == 'openrouter':
        headers.update({'HTTP-Referer': 'https://github.com/hety0211/lumen-raw', 'X-Title': 'LUMEN RAW'})
    return headers


def _anthropic_headers(service):
    return {'x-api-key': service['key'], 'anthropic-version': '2023-06-01'}


def list_models(service):
    """Model names offered by the service."""
    api, base = service['api'], service['base_url']
    if not base:
        raise ServiceError(tr('请先填写接口地址。'))
    timeout = min(30, service.get('timeout', 30))
    if api == 'ollama':
        data = _http(base + '/api/tags', timeout=timeout)
        names = [m.get('name') or m.get('model') for m in data.get('models', [])]
    elif api == 'anthropic':
        data = _http(base + '/v1/models?limit=100', headers=_anthropic_headers(service), timeout=timeout)
        names = [m.get('id') for m in data.get('data', [])]
    else:
        data = _http(base + '/models', headers=_openai_headers(service), timeout=timeout)
        items = data.get('data', data.get('models', [])) if isinstance(data, dict) else data
        names = [m.get('id') or m.get('name') for m in items if isinstance(m, dict)]
        if service['provider'] == 'gemini':
            names = [n.removeprefix('models/') for n in names if n]
    names = [n for n in names if n and not re.search(r'embed|whisper|tts|dall-e|rerank|moderation|bge-|audio', n, re.I)]
    return sorted(dict.fromkeys(names))


#: Claude models that accept server-side refusal fallback (``fallbacks: "default"``).
_FALLBACK_MODELS = ('claude-fable-5-1', 'claude-opus-5-5', 'claude-opus-5', 'claude-sonnet-5-5')


def _post(url, payload, headers, timeout, optional=(), optional_headers=()):
    """POST; when the server rejects the request (400 / 422), retry once without the optional extras."""
    try:
        return _http(url, payload, headers, timeout)
    except ServiceError as error:
        if error.status not in (400, 422) or not (set(optional) & set(payload) or set(optional_headers) & set(headers)):
            raise
        payload = {k: v for k, v in payload.items() if k not in optional}
        headers = {k: v for k, v in headers.items() if k not in optional_headers}
        return _http(url, payload, headers, timeout)


def _usage(target, prompt, output):
    if target is not None and (prompt or output):
        target.update(input=int(prompt or 0), output=int(output or 0), total=int(prompt or 0) + int(output or 0))


def complete(service, messages, image=None, usage=None):
    """Send ``messages`` (system first) and return the reply text; token counts go into ``usage``."""
    api, base, name = service['api'], service['base_url'], service['model']
    timeout = service.get('timeout', 180)
    thinking = dict(service.get('thinking') or {})
    if not base:
        raise ServiceError(tr('请先在设置中填写接口地址。'))
    if not name:
        raise ServiceError(tr('请先在设置中选择或填写模型名称。'))
    if not service['local'] and not service['key'] and service['provider'] != 'custom':
        raise ServiceError(tr('请先在设置中填写 {label} 的 API Key。', label=service['label']))
    system, chat = messages[0]['content'], [dict(m) for m in messages[1:]]
    if api == 'ollama':
        if image:
            chat[-1]['images'] = [image]
        payload = dict(model=name, stream=False, format='json', options=dict(temperature=.3, num_ctx=8192),
                       messages=[dict(role='system', content=system)] + chat, **thinking)
        data = _post(base + '/api/chat', payload, {}, timeout, optional=thinking)
        _usage(usage, data.get('prompt_eval_count'), data.get('eval_count'))
        return (data.get('message') or {}).get('content', '')
    if api == 'anthropic':
        if image:
            chat[-1]['content'] = [dict(type='image', source=dict(type='base64', media_type='image/jpeg', data=image)),
                                   dict(type='text', text=chat[-1]['content'])]
        # Thinking counts against max_tokens; current models reject temperature, so none is sent.
        payload = dict(model=name, max_tokens=16000, system=system, messages=chat, **thinking)
        headers, optional = _anthropic_headers(service), set(thinking)
        if name.startswith(_FALLBACK_MODELS):
            payload['fallbacks'] = 'default'
            headers['anthropic-beta'] = 'server-side-fallback-2026-07-01'
            optional.add('fallbacks')
        data = _post(base + '/v1/messages', payload, headers, timeout, optional, ('anthropic-beta',))
        tokens = data.get('usage') or {}
        _usage(usage, (tokens.get('input_tokens') or 0) + (tokens.get('cache_read_input_tokens') or 0)
               + (tokens.get('cache_creation_input_tokens') or 0), tokens.get('output_tokens'))
        text = ''.join(block.get('text', '') for block in data.get('content', []) if block.get('type') == 'text')
        if data.get('stop_reason') == 'refusal':
            raise ServiceError(tr('Claude 拒绝了这次请求，请换一种说法或换用其他模型。'))
        if not text and data.get('stop_reason') == 'max_tokens':
            raise ServiceError(tr('回复超出长度上限（思考过长），可在设置中降低思考强度。'))
        return text
    if image:
        chat[-1]['content'] = [dict(type='text', text=chat[-1]['content']),
                               dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,' + image))]
    payload = dict(model=name, messages=[dict(role='system', content=system)] + chat, stream=False, **thinking)
    if service['local']:
        payload['temperature'] = .3
    if service['json_mode']:
        payload['response_format'] = dict(type=service['json_mode'])
    # Servers that reject response_format or a thinking field are retried with the prompt alone.
    data = _post(base + '/chat/completions', payload, _openai_headers(service), timeout, set(thinking) | {'response_format'})
    tokens = data.get('usage') or {}
    _usage(usage, tokens.get('prompt_tokens'), tokens.get('completion_tokens'))
    choices = data.get('choices') or []
    if not choices:
        raise ServiceError(tr('服务没有返回结果：') + json.dumps(data, ensure_ascii=False)[:300])
    content = (choices[0].get('message') or {}).get('content') or ''
    if isinstance(content, list):
        content = ''.join(part.get('text', '') for part in content if isinstance(part, dict))
    if not content.strip() and choices[0].get('finish_reason') == 'length':
        raise ServiceError(tr('回复超出长度上限（思考过长），可在设置中降低思考强度。'))
    return content


def build_messages(instruction, edits, info=None, stats=None, history=()):
    messages = [dict(role='system', content=system_prompt())]
    for previous, reply in list(history)[-3:]:
        messages += [dict(role='user', content='用户指令：' + previous), dict(role='assistant', content=reply)]
    messages.append(dict(role='user', content=user_message(instruction, edits, info, stats)))
    return messages


# --------------------------------------------------------------------------- replies

def parse_reply(text):
    """The first JSON object in a reply; tolerates code fences, <think> blocks and prose."""
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.S | re.I)
    text = re.sub(r'^.*?</think>', '', text, flags=re.S | re.I)
    start = text.find('{')
    while start != -1:
        depth, quoted, escaped = 0, False, False
        for end in range(start, len(text)):
            c = text[end]
            if quoted:
                escaped, quoted = (False, quoted) if escaped else (c == '\\', c != '"')
                continue
            if c == '"':
                quoted = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        value = json.loads(candidate)
                    except ValueError:
                        try:
                            value = json.loads(re.sub(r',\s*([}\]])', r'\1', candidate))
                        except ValueError:
                            break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find('{', start + 1)
    raise ServiceError(tr('模型没有返回可用的 JSON 参数：') + (text.strip()[:200] or tr('（空回复）')))


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        match = re.search(r'[-+]?\d+(?:\.\d+)?', value.replace('＋', '+').replace('－', '-'))
        value = float(match.group()) if match else None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _clip(value, low, high):
    return float(min(high, max(low, value)))


def _adjustment_key(name):
    name = str(name).strip()
    key = name.lower().replace(' ', '_').replace('-', '_')
    return key if key in PARAMS or key in EFFECTS else ALIASES.get(name, ALIASES.get(key))


def _adjustments(values, target):
    if not isinstance(values, dict):
        return
    for name, raw in values.items():
        key, value = _adjustment_key(name), _number(raw)
        if key in PARAMS and value is not None:
            target[key] = _clip(value, PARAMS[key][1], PARAMS[key][2])


def _color_index(name):
    name = str(name).strip().lower()
    for index, (key, title, aliases) in enumerate(HSL_COLORS):
        if name in (key, title) or name in aliases:
            return index
    return None


def _part_index(name):
    name = str(name).strip().lower()
    for index, (key, title, aliases) in enumerate(HSL_PARTS):
        if name in (key, title) or name in aliases:
            return index
    return None


def _curve(points):
    if not isinstance(points, list):
        return None
    pairs = [(_number(p[0]), _number(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    # 0–255 as asked; a reply entirely within 0–1 is read as normalized.
    scale = 1. if max(max(abs(x), abs(y)) for x, y in pairs) <= 1 else 255.
    result = []
    for x, y in sorted((_clip(x / scale, 0, 1), _clip(y / scale, 0, 1)) for x, y in pairs):
        if not result or x - result[-1][0] >= .004:
            result.append([round(x, 4), round(y, 4)])
    if result[0][0] > 0:
        result.insert(0, [0., 0.])
    result[0][0] = 0.
    if result[-1][0] < 1:
        result.append([1., 1.])
    result[-1][0] = 1.
    return result if 2 <= len(result) <= 16 else None


def _gradient(region, mask):
    if region == 'top':
        mask.update(start=[.5, .6], end=[.5, 0.])
    elif region == 'bottom':
        mask.update(start=[.5, .4], end=[.5, 1.])
    elif region == 'center':
        mask.update(start=[.2, .18], end=[.8, .82], feather=60.)
    elif region == 'edges':
        mask.update(start=[.12, .1], end=[.88, .9], feather=70., invert=True)


class Plan:
    """A reply turned into edits: ``target`` plus AI regions still to be computed."""

    def __init__(self, base, target, regions, explanation, notes):
        self.base, self.target, self.regions = base, target, regions
        self.explanation, self.notes = explanation, notes


def plan(edits, reply):
    """Apply a parsed reply to a copy of ``edits``.

    Returns a ``Plan`` whose ``target`` contains everything except AI-selected
    masks; those are listed in ``regions`` as ``(kind, mask)`` and are attached with
    ``attach_regions`` once their coverage has been computed.
    """
    if not isinstance(reply, dict):
        raise ServiceError(tr('模型回复的格式不正确。'))
    base = copy.deepcopy(edits)
    target = copy.deepcopy(edits)
    notes = []
    # Some models nest the other sections inside "adjustments"; lift them to the top level.
    if isinstance(reply.get('adjustments'), dict):
        nested = {k: v for k, v in reply['adjustments'].items()
                  if k in ('hsl', 'grading', 'effects', 'masks', 'curve', 'monochrome') and k not in reply}
        if nested:
            reply = dict(reply, **nested, adjustments={k: v for k, v in reply['adjustments'].items() if k not in nested})
    if reply.get('reset') is True:
        neutral = model.recipe()
        for key in model.LOOK_KEYS:
            target[key] = copy.deepcopy(neutral[key])
    adjustments = dict(reply.get('adjustments') or {}) if isinstance(reply.get('adjustments'), dict) else {}
    # Some models put slider keys at the top level.
    for name, value in reply.items():
        if name not in ('adjustments', 'effects') and _adjustment_key(name) in PARAMS:
            adjustments.setdefault(name, value)
    _adjustments(adjustments, target['adjustments'])
    effects = dict(reply.get('effects') or {}) if isinstance(reply.get('effects'), dict) else {}
    for name, value in list(adjustments.items()) + list(effects.items()):
        key, number = _adjustment_key(name), _number(value)
        if key in EFFECTS and number is not None:
            target['effects'][key] = _clip(number, EFFECTS[key][1], EFFECTS[key][2])
    hsl = reply.get('hsl')
    if isinstance(hsl, dict):
        for color, parts in hsl.items():
            index = _color_index(color)
            if index is None:
                continue
            if isinstance(parts, (list, tuple)) and len(parts) == 3:
                parts = dict(zip(('hue', 'saturation', 'luminance'), parts))
            if isinstance(parts, dict):
                for part, value in parts.items():
                    slot, number = _part_index(part), _number(value)
                    if slot is not None and number is not None:
                        target['hsl'][index][slot] = _clip(number, -100, 100)
    grading = reply.get('grading')
    if isinstance(grading, dict):
        for zone, _ in ZONES:
            value = grading.get(zone)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                value = dict(hue=value[0], saturation=value[1])
            if isinstance(value, dict):
                hue, saturation = _number(value.get('hue')), _number(value.get('saturation'))
                if hue is not None:
                    target['grading'][zone][0] = hue % 360
                if saturation is not None:
                    target['grading'][zone][1] = _clip(saturation, 0, 100)
        balance = _number(grading.get('balance'))
        if balance is not None:
            target['grading']['balance'] = _clip(balance, -100, 100)
    if isinstance(reply.get('monochrome'), bool):
        target['monochrome'] = reply['monochrome']
    if 'curve' in reply and reply['curve'] is not None:
        curve = _curve(reply['curve'])
        if curve:
            target['curves']['RGB'] = curve
            target['curve_mode'] = 'smooth'
        else:
            notes.append(tr('曲线格式无效，已忽略'))
    regions = []
    masks = reply.get('masks')
    if isinstance(masks, list):
        for request in masks:
            if not isinstance(request, dict):
                continue
            values = request.get('adjustments') if isinstance(request.get('adjustments'), dict) else \
                {k: v for k, v in request.items() if _adjustment_key(k) in PARAMS}
            index = _number(request.get('index'))
            if index is not None and request.get('region') is None:
                index = int(index)
                if 0 <= index < len(target['masks']):
                    _adjustments(values, target['masks'][index]['adjustments'])
                else:
                    notes.append(tr('没有序号为 {index} 的蒙版', index=index))
                continue
            region = str(request.get('region', '')).strip().lower()
            region = {'天空': 'sky', '主体': 'subject', '人物': 'person', '人像': 'person', '背景': 'background',
                      '近景': 'foreground', '前景': 'foreground', 'people': 'person', 'portrait': 'person'}.get(region, region)
            if region not in REGIONS:
                notes.append(tr('不支持的区域：{region}', region=region or tr('（未指定）')))
                continue
            if len(regions) + len(target['masks']) - len(base['masks']) >= MAX_NEW_MASKS:
                notes.append(tr('一次最多新增 {count} 个蒙版', count=MAX_NEW_MASKS))
                break
            if len(target['masks']) + len(regions) >= 32:
                notes.append(tr('蒙版数量已达上限 32'))
                break
            title, kind = REGIONS[region]
            mask = model.new_mask(kind, len(target['masks']) + len(regions) + 1)
            mask['name'] = tr('{title}（自然语言）', title=tr(title))
            if kind in ('linear', 'radial'):
                _gradient(region, mask)
            else:
                mask['feather'] = 10.
            _adjustments(values, mask['adjustments'])
            if not any(mask['adjustments'].values()):
                continue
            if region in AI_REGIONS:
                regions.append((kind, mask))
            else:
                target['masks'].append(mask)
    explanation = str(reply.get('explanation') or reply.get('说明') or '').strip()[:600]
    return Plan(base, target, regions, explanation, notes)


def attach_regions(plan_, alphas):
    """Append computed AI masks; ``alphas`` holds an encoded raster (or None) per region."""
    from . import selection
    for (kind, mask), alpha in zip(plan_.regions, alphas):
        if alpha is None:
            plan_.notes.append(tr('没有识别到{region}区域', region=tr(REGIONS[kind][0]) if kind in REGIONS else kind))
            continue
        mask = copy.deepcopy(mask)
        mask['raster'] = selection.encode(alpha) if not isinstance(alpha, str) else alpha
        plan_.target['masks'].append(mask)
    plan_.regions = []
    plan_.target = model.validate(plan_.target)
    return plan_


def blend(base, target, amount):
    """``target`` applied at ``amount`` (0–1.5) relative to ``base``, clamped to valid ranges."""
    t = float(amount)
    if abs(t - 1) < 1e-9:
        return copy.deepcopy(target)
    lerp = lambda a, b: a + (b - a) * t
    result = copy.deepcopy(target)
    for key, (_, low, high, _) in PARAMS.items():
        result['adjustments'][key] = _clip(lerp(base['adjustments'][key], target['adjustments'][key]), low, high)
    for key, (_, low, high, _) in EFFECTS.items():
        result['effects'][key] = _clip(lerp(base['effects'][key], target['effects'][key]), low, high)
    result['hsl'] = [[_clip(lerp(a, b), -100, 100) for a, b in zip(ra, rb)] for ra, rb in zip(base['hsl'], target['hsl'])]
    for zone, _ in ZONES:
        (h0, s0), (h1, s1) = base['grading'][zone], target['grading'][zone]
        hue = h1 if s0 == 0 else (h0 + ((h1 - h0 + 180) % 360 - 180) * t) % 360
        result['grading'][zone] = [float(hue), _clip(lerp(s0, s1), 0, 100)]
    result['grading']['balance'] = _clip(lerp(base['grading']['balance'], target['grading']['balance']), -100, 100)
    for channel, points in target['curves'].items():
        before = base['curves'][channel]
        if points != before:
            xs, ys = zip(*before)
            result['curves'][channel] = [[x, _clip(lerp(float(np.interp(x, xs, ys)), y), 0, 1)] for x, y in points]
    if target['tone_curve'] != base['tone_curve']:
        xs, ys = zip(*base['tone_curve'])
        tone = [[x, _clip(lerp(float(np.interp(x, xs, ys)), y), 0, 1)] for x, y in target['tone_curve']]
        result['tone_curve'] = [[x, max(y, tone[i - 1][1]) if i else y] for i, (x, y) in enumerate(tone)]
    result['monochrome'] = target['monochrome'] if t >= .5 else base['monochrome']
    for index, mask in enumerate(result['masks']):
        before = base['masks'][index]['adjustments'] if index < len(base['masks']) else model.adjustments()
        for key, (_, low, high, _) in PARAMS.items():
            mask['adjustments'][key] = _clip(lerp(before[key], target['masks'][index]['adjustments'][key]), low, high)
    return result


def _signed(value, digits=0):
    text = f'{value:+.{digits}f}'
    return '0' if float(text) == 0 else text


def changes(base, target):
    """Human-readable list of what differs between two recipes."""
    lines = []
    for key, (name, _, _, _) in PARAMS.items():
        a, b = base['adjustments'][key], target['adjustments'][key]
        if abs(a - b) > 1e-6:
            digits = 2 if key == 'exposure' else 0
            lines.append(f'{tr(name)}  {_signed(a, digits)} → {_signed(b, digits)}')
    for (_, color, _), ra, rb in zip(HSL_COLORS, base['hsl'], target['hsl']):
        for (_, part, _), a, b in zip(HSL_PARTS, ra, rb):
            if abs(a - b) > 1e-6:
                lines.append(f'{tr(color)} · {tr(part)}  {_signed(a)} → {_signed(b)}')
    for zone, title in ZONES:
        if base['grading'][zone] != target['grading'][zone]:
            h, s = target['grading'][zone]
            lines.append(tr('{zone}分级  {hue}° · {strength}', zone=tr(title), hue=round(h), strength=round(s)) if s
                         else tr('{zone}分级  清除', zone=tr(title)))
    if abs(base['grading']['balance'] - target['grading']['balance']) > 1e-6:
        lines.append(tr('分级平衡  {value}', value=_signed(target['grading']['balance'])))
    for key, (name, _, _, _) in EFFECTS.items():
        a, b = base['effects'][key], target['effects'][key]
        if abs(a - b) > 1e-6:
            lines.append(f'{tr(name)}  {a:g} → {round(b, 1):g}')
    if base['monochrome'] != target['monochrome']:
        lines.append(tr('黑白处理  开') if target['monochrome'] else tr('黑白处理  关'))
    if base['curves'] != target['curves']:
        lines.append(tr('RGB 曲线  已调整'))
    if base['tone_curve'] != target['tone_curve']:
        lines.append(tr('曝光曲线  已还原') if target['tone_curve'] == [[0., 0.], [1., 1.]] else tr('曝光曲线  已调整'))
    for index, mask in enumerate(target['masks']):
        values = tr('，').join(f'{tr(PARAMS[k][0])} {_signed(v, 2 if k == "exposure" else 0)}'
                          for k, v in mask['adjustments'].items() if abs(v) > 1e-6)
        if index >= len(base['masks']):
            lines.append(tr('新增蒙版 {name}：{values}', name=mask['name'], values=values or tr('无调整')))
        elif mask['adjustments'] != base['masks'][index]['adjustments']:
            lines.append(tr('蒙版 {name}：{values}', name=mask['name'], values=values or tr('无调整')))
    return lines
