"""1.5.0: natural-language editing (local / cloud language models) and offline speech input."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import copy
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import pytest
from lumen import model, nl_edit, speech

REAL_KEYCHAIN = (nl_edit._keychain_read, nl_edit._keychain_write)


# ----------------------------------------------------------------------------- fake services

class FakeService:
    """Answers OpenAI, Ollama and Anthropic style requests with a canned reply."""

    def __init__(self, reply):
        self.reply, self.requests, self.reject_json_mode, self.status = reply, [], False, None
        self.reject_keys, self.stop_reason = set(), 'end_turn'
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send(self, code, data):
                body = json.dumps(data).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                service.requests.append(('GET', self.path, {k.lower(): v for k, v in self.headers.items()}, None))
                if self.path == '/api/tags':
                    return self.send(200, {'models': [{'name': 'qwen3:8b'}, {'name': 'nomic-embed-text'}]})
                if self.path.startswith('/v1/models'):
                    return self.send(200, {'data': [{'id': 'claude-x'}, {'id': 'gpt-y'}, {'id': 'text-embedding-3'}]})
                self.send(404, {'error': 'no route'})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                service.requests.append(('POST', self.path, {k.lower(): v for k, v in self.headers.items()}, payload))
                if service.status:
                    return self.send(service.status, {'error': {'message': 'Incorrect API key provided'}})
                if service.reject_keys & set(payload):
                    return self.send(400, {'error': {'message': 'unknown parameter'}})
                text = service.reply if isinstance(service.reply, str) else json.dumps(service.reply, ensure_ascii=False)
                if self.path == '/api/chat':
                    return self.send(200, {'message': {'role': 'assistant', 'content': text}, 'prompt_eval_count': 1800, 'eval_count': 120})
                if self.path == '/v1/messages':
                    return self.send(200, {'content': [{'type': 'thinking', 'thinking': ''}, {'type': 'text', 'text': text}],
                                           'stop_reason': service.stop_reason, 'usage': {'input_tokens': 2100, 'output_tokens': 300}})
                if self.path == '/v1/chat/completions':
                    if service.reject_json_mode and 'response_format' in payload:
                        return self.send(400, {'error': {'message': "'response_format.type' must be 'json_schema' or 'text'"}})
                    return self.send(200, {'choices': [{'message': {'role': 'assistant', 'content': text}}],
                                           'usage': {'prompt_tokens': 2300, 'completion_tokens': 186, 'total_tokens': 2486}})
                self.send(404, {'error': 'no route'})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture(autouse=True)
def keychain(monkeypatch):
    """In-memory stand-in for the macOS keychain item, so tests never touch the real one."""
    store = {'value': None, 'writes': 0, 'refuse': False}

    def write(value):
        if store['refuse']:
            raise OSError('refused')
        store['value'], store['writes'] = value, store['writes'] + 1

    monkeypatch.setattr(nl_edit, '_keychain_read', lambda: store['value'])
    monkeypatch.setattr(nl_edit, '_keychain_write', write)
    monkeypatch.setattr(nl_edit, '_keychain_cache', None)
    return store


@pytest.fixture
def fake():
    services = []
    def make(reply):
        services.append(FakeService(reply))
        return services[-1]
    yield make
    for service in services:
        service.close()


def service_for(provider, url, key='', model_name='m'):
    settings = nl_edit.default_settings()
    settings['provider'] = provider
    settings['providers'][provider].update(base_url=url, key=key, model=model_name)
    settings['timeout'] = 20
    return nl_edit.active(settings)


REPLY = {
    'explanation': '天空更蓝，整体更暖。',
    'adjustments': {'exposure': 0.3, 'highlights': -35, 'temperature': 12},
    'hsl': {'blue': {'saturation': 20, 'luminance': -10}},
    'grading': {'highlights': {'hue': 40, 'saturation': 15}},
    'effects': {'vignette': -15},
    'masks': [{'region': 'top', 'adjustments': {'dehaze': 20}}],
}


# ----------------------------------------------------------------------------- parameters and prompt

def test_the_prompt_lists_every_adjustable_parameter_with_the_recipe_ranges():
    assert set(nl_edit.PARAMS) == set(model.adjustments())
    assert set(nl_edit.EFFECTS) == set(model.effects())
    assert [k for k, _, _ in nl_edit.HSL_COLORS] and len(nl_edit.HSL_COLORS) == len(model.COLORS)
    prompt = nl_edit.system_prompt()
    for key, (name, low, high, _) in {**nl_edit.PARAMS, **nl_edit.EFFECTS}.items():
        assert f'- {key}（{name}）{low}…{high}' in prompt
    for key in nl_edit.REGIONS:
        assert key in prompt
    # Extremes accepted by the prompt are exactly what a project may store.
    edits = model.recipe()
    for key, (_, low, high, _) in nl_edit.PARAMS.items():
        for value in (low, high):
            model.validate(dict(copy.deepcopy(edits), adjustments=dict(edits['adjustments'], **{key: value})))


def test_user_message_carries_photo_statistics_and_current_values():
    edits = model.recipe()
    edits['adjustments']['exposure'] = .5
    edits['hsl'][5] = [0., 25., 0.]
    rgb = np.zeros((60, 90, 3), np.float32)
    rgb[:20] = [.3, .5, .9]  # bright blue top third
    stats = nl_edit.image_statistics(rgb)
    assert stats['暗部死黑比例'] > .6 and stats['上/中/下三分之一平均亮度'][0] > .4
    info = dict(format='ARW', width=6000, height=4000, photo=dict(body='ILCE-7RM5', iso='ISO 6400', date='2026-09-01 18:42:10'))
    text = nl_edit.user_message('天空更蓝', edits, info, stats)
    assert '"ISO 6400"' in text and '"18:42"' in text and '天空更蓝' in text
    state = json.loads(text.split('当前参数：')[1].split('\n')[0])
    assert state['adjustments']['exposure'] == .5 and state['hsl'] == {'blue': {'saturation': 25.}}


# ----------------------------------------------------------------------------- replies

@pytest.mark.parametrize('text', [
    '{"adjustments": {"exposure": 0.4}}',
    '```json\n{"adjustments": {"exposure": 0.4}}\n```',
    '<think>The user wants {brighter}. Use {"exposure": 9}</think>\n{"adjustments": {"exposure": 0.4}}',
    'Sure! Here are the settings:\n{"adjustments": {"exposure": 0.4,},}\nHope it helps.',
    '好的。{"explanation": "括号 } 在字符串里", "adjustments": {"exposure": 0.4}}',
])
def test_replies_are_parsed_from_typical_model_output(text):
    assert nl_edit.parse_reply(text)['adjustments']['exposure'] == .4


def test_a_reply_without_json_is_reported():
    with pytest.raises(nl_edit.ServiceError, match='JSON'):
        nl_edit.parse_reply('I cannot edit photos.')


def test_plan_clamps_maps_aliases_and_always_validates():
    edits = model.recipe()
    edits['crop'] = [.1, .1, .9, .9]
    reply = {
        'adjustments': {'曝光': '+9', 'temp': -150, 'Contrast': 20, 'sharpen': 140, 'unknown': 5, 'vignette': -30},
        'shadows': 25,  # some models put sliders at the top level
        'hsl': {'蓝': [10, 20, -10], 'cyan': {'s': 30}, 'skin': {'hue': 5}},
        'grading': {'shadows': [570, 130], 'balance': -300},
        'effects': {'grain': 20},
        'monochrome': True,
        'curve': [[64, 50], [0, 10], [192, 210]],
    }
    p = nl_edit.plan(edits, reply)
    a = p.target['adjustments']
    assert (a['exposure'], a['temperature'], a['contrast'], a['sharpness'], a['shadows']) == (5, -100, 20, 100, 25)
    assert p.target['hsl'][5] == [10, 20, -10] and p.target['hsl'][4][1] == 30
    assert p.target['grading']['shadows'] == [210, 100] and p.target['grading']['balance'] == -100
    assert p.target['effects']['grain'] == 20 and p.target['effects']['vignette'] == -30
    assert p.target['monochrome'] and p.target['curve_mode'] == 'smooth'
    assert p.target['curves']['RGB'][0] == [0, round(10 / 255, 4)] and p.target['curves']['RGB'][-1] == [1, 1]
    assert p.target['crop'] == edits['crop'] and p.base == edits
    assert model.validate(p.target) == p.target


def test_sections_nested_inside_adjustments_are_applied():
    # Seen from qwen3-14b: grading placed inside "adjustments".
    reply = {'adjustments': {'contrast': 15, 'grading': {'shadows': {'hue': 210, 'saturation': 18},
                                                          'highlights': {'hue': 30, 'saturation': 16}},
                             'hsl': {'blue': {'saturation': 20}}, 'monochrome': True}}
    p = nl_edit.plan(model.recipe(), reply)
    assert p.target['adjustments']['contrast'] == 15
    assert p.target['grading']['shadows'] == [210, 18] and p.target['grading']['highlights'] == [30, 16]
    assert p.target['hsl'][5][1] == 20 and p.target['monochrome']


def test_reset_restores_the_look_but_keeps_framing_masks_and_white_balance():
    edits = model.recipe()
    edits['adjustments'].update(exposure=1, saturation=40)
    edits['crop'], edits['wb_gain'] = [0, 0, .5, .5], [1.2, 1., .8]
    edits['masks'].append(model.new_mask('radial', 1))
    p = nl_edit.plan(edits, {'reset': True, 'adjustments': {'contrast': 10}})
    assert p.target['adjustments'] == dict(model.adjustments(), contrast=10)
    assert p.target['crop'] == edits['crop'] and p.target['wb_gain'] == edits['wb_gain'] and len(p.target['masks']) == 1


def test_masks_create_gradients_queue_ai_regions_and_edit_existing_ones():
    edits = model.recipe()
    edits['masks'].append(model.new_mask('radial', 1))
    reply = {'masks': [
        {'region': 'top', 'adjustments': {'exposure': -.5}},
        {'region': '天空', 'adjustments': {'saturation': 20}},
        {'region': 'edges', 'adjustments': {'exposure': -.3}},
        {'index': 0, 'adjustments': {'clarity': 30}},
        {'index': 7, 'adjustments': {'clarity': 30}},
        {'region': 'moon', 'adjustments': {'exposure': 1}},
        {'region': 'person', 'adjustments': {}},
        {'region': 'bottom', 'adjustments': {'shadows': 10}},
        {'region': 'center', 'adjustments': {'exposure': .2}},
    ]}
    p = nl_edit.plan(edits, reply)
    kinds = [m['kind'] for m in p.target['masks']]
    assert kinds == ['radial', 'linear', 'radial', 'linear']
    top, edges = p.target['masks'][1], p.target['masks'][2]
    assert top['end'][1] < top['start'][1] and edges['invert']
    assert p.target['masks'][0]['adjustments']['clarity'] == 30
    assert [kind for kind, _ in p.regions] == ['sky']
    assert any('7' in n for n in p.notes) and any('moon' in n for n in p.notes)
    assert any(str(nl_edit.MAX_NEW_MASKS) in n for n in p.notes)  # the centre mask exceeded the limit
    alpha = np.zeros((40, 60), np.float32)
    alpha[:15] = 1
    p = nl_edit.attach_regions(p, [alpha])
    assert p.target['masks'][-1]['kind'] == 'sky' and p.target['masks'][-1]['raster']
    assert model.validate(p.target) == p.target


def test_strength_blends_between_start_and_result_and_clamps():
    edits = model.recipe()
    edits['adjustments']['highlights'] = -20
    p = nl_edit.attach_regions(nl_edit.plan(edits, REPLY), [])
    assert nl_edit.blend(p.base, p.target, 0) == p.base or nl_edit.blend(p.base, p.target, 0)['adjustments'] == p.base['adjustments']
    assert nl_edit.blend(p.base, p.target, 1) == p.target
    half = nl_edit.blend(p.base, p.target, .5)
    assert half['adjustments']['highlights'] == pytest.approx(-27.5) and half['hsl'][5][1] == 10
    assert half['grading']['highlights'] == [40, 7.5]
    assert half['masks'][0]['adjustments']['dehaze'] == 10
    strong = nl_edit.blend(p.base, p.target, 1.5)
    assert strong['adjustments']['exposure'] == pytest.approx(.45)
    p = nl_edit.plan(model.recipe(), {'adjustments': {'contrast': 80}})
    assert nl_edit.blend(p.base, p.target, 1.5)['adjustments']['contrast'] == 100
    for amount in (0, .3, .5, 1.2, 1.5):
        result = nl_edit.blend(p.base, p.target, amount)
        assert model.validate(result) == result


def test_change_list_describes_the_result():
    p = nl_edit.attach_regions(nl_edit.plan(model.recipe(), REPLY), [])
    lines = nl_edit.changes(p.base, p.target)
    assert '曝光  0 → +0.30' in lines and '亮部  0 → -35' in lines and '蓝 · 饱和度  0 → +20' in lines
    assert '高光分级  40° · 15' in lines and '暗角  0 → -15' in lines
    assert any(line.startswith('新增蒙版 上方渐变') and '去薄雾 +20' in line for line in lines)


# ----------------------------------------------------------------------------- services

def test_openai_compatible_request_falls_back_when_json_mode_is_rejected(fake):
    server = fake(REPLY)
    server.reject_json_mode = True
    service = service_for('llamacpp', server.url + '/v1', key='local-key')
    messages = nl_edit.build_messages('更暖', model.recipe(), history=[('更亮', '{"adjustments": {"exposure": 0.3}}')])
    text = nl_edit.complete(service, messages, image='QUJD')
    assert nl_edit.parse_reply(text)['adjustments']['temperature'] == 12
    first, second = server.requests[0][3], server.requests[1][3]
    assert first['response_format'] == {'type': 'json_object'} and 'response_format' not in second
    assert server.requests[1][2]['authorization'] == 'Bearer local-key'
    assert [m['role'] for m in second['messages']] == ['system', 'user', 'assistant', 'user']
    assert second['messages'][-1]['content'][1]['image_url']['url'] == 'data:image/jpeg;base64,QUJD'
    assert second['temperature'] == .3 and second['stream'] is False


def test_ollama_and_anthropic_requests_use_their_native_formats(fake):
    server = fake('<think>…</think>' + json.dumps(REPLY))
    messages = nl_edit.build_messages('更暖', model.recipe())
    ollama = service_for('ollama', server.url, model_name='qwen3:8b')
    assert nl_edit.parse_reply(nl_edit.complete(ollama, messages, image='QUJD'))['effects']['vignette'] == -15
    _, path, _, payload = server.requests[-1]
    assert path == '/api/chat' and payload['format'] == 'json' and payload['messages'][-1]['images'] == ['QUJD']
    assert payload['options']['num_ctx'] >= 8192
    claude = service_for('anthropic', server.url, key='sk-ant-test', model_name='claude-x')
    assert nl_edit.parse_reply(nl_edit.complete(claude, messages, image='QUJD'))['adjustments']['exposure'] == .3
    _, path, headers, payload = server.requests[-1]
    assert path == '/v1/messages' and headers['x-api-key'] == 'sk-ant-test' and headers['anthropic-version']
    assert payload['system'].startswith('你是 LUMEN RAW') and payload['max_tokens'] > 0
    assert payload['messages'][-1]['content'][0]['source']['data'] == 'QUJD'
    assert nl_edit.list_models(ollama) == ['qwen3:8b']
    assert nl_edit.list_models(claude) == ['claude-x', 'gpt-y']
    assert nl_edit.list_models(service_for('openai', server.url + '/v1', key='k')) == ['claude-x', 'gpt-y']


def test_service_errors_are_explained(fake):
    server = fake(REPLY)
    server.status = 401
    with pytest.raises(nl_edit.ServiceError, match='API Key'):
        nl_edit.complete(service_for('deepseek', server.url + '/v1', key='bad'), nl_edit.build_messages('x', model.recipe()))
    with pytest.raises(nl_edit.ServiceError, match='API Key'):
        nl_edit.complete(service_for('deepseek', server.url + '/v1'), nl_edit.build_messages('x', model.recipe()))
    with pytest.raises(nl_edit.ServiceError, match='模型'):
        nl_edit.complete(service_for('ollama', server.url, model_name=''), nl_edit.build_messages('x', model.recipe()))
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    with pytest.raises(nl_edit.ServiceError, match='服务未启动'):
        nl_edit.complete(service_for('lmstudio', f'http://127.0.0.1:{port}/v1'), nl_edit.build_messages('x', model.recipe()))


def test_settings_round_trip_without_storing_the_key_in_plain_text(tmp_path):
    settings = nl_edit.default_settings()
    settings.update(provider='deepseek', attach_image=True, timeout=99, voice_language='zh', reasoning='low')
    settings['providers']['deepseek'].update(key='sk-secret-123', model='deepseek-v4-pro')
    path = tmp_path / 'natural-language.json'
    nl_edit.save_settings(settings, path)
    assert 'sk-secret-123' not in path.read_text(encoding='utf-8') or os.name != 'nt'
    loaded = nl_edit.load_settings(path)
    assert loaded == settings
    assert nl_edit.load_settings(tmp_path / 'missing.json') == nl_edit.default_settings()
    path.write_text('{"provider": "nope", "timeout": "abc", "providers": {"ollama": {"key": "dpapi:!!"}}}', encoding='utf-8')
    broken = nl_edit.load_settings(path)
    assert broken['provider'] == 'ollama' and broken['providers']['ollama']['key'] == ''


def test_keys_live_in_one_keychain_item_on_macos(tmp_path, monkeypatch, keychain):
    monkeypatch.setattr(nl_edit, 'KEYCHAIN', True)
    settings = nl_edit.default_settings()
    settings['providers']['deepseek']['key'] = 'sk-deep'
    settings['providers']['anthropic']['key'] = 'sk-ant'
    path = tmp_path / 'natural-language.json'
    nl_edit.save_settings(settings, path)
    stored = json.loads(path.read_text(encoding='utf-8'))
    assert 'sk-' not in path.read_text(encoding='utf-8')
    assert stored['providers']['deepseek']['key'] == 'keychain:' and stored['providers']['ollama']['key'] == ''
    assert json.loads(keychain['value']) == {'deepseek': 'sk-deep', 'anthropic': 'sk-ant'}
    monkeypatch.setattr(nl_edit, '_keychain_cache', None)
    assert nl_edit.load_settings(path) == settings and keychain['writes'] == 1
    nl_edit.save_settings(settings, path)
    assert keychain['writes'] == 1  # unchanged keys: no keychain write (no access prompt)
    settings['providers']['anthropic']['key'] = ''
    nl_edit.save_settings(settings, path)
    assert json.loads(keychain['value']) == {'deepseek': 'sk-deep'}
    # Access denied: the key is missing rather than wrong; a refused write falls back to the file.
    monkeypatch.setattr(nl_edit, '_keychain_read', lambda: (_ for _ in ()).throw(OSError('denied')))
    assert nl_edit.load_settings(path)['providers']['deepseek']['key'] == ''
    keychain['refuse'] = True
    settings['providers']['deepseek']['key'] = 'sk-new'
    nl_edit.save_settings(settings, path)
    assert json.loads(path.read_text(encoding='utf-8'))['providers']['deepseek']['key'] != 'keychain:'
    assert nl_edit.load_settings(path)['providers']['deepseek']['key'] == 'sk-new'


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS keychain')
def test_real_keychain_item_round_trip(monkeypatch):
    read, write = REAL_KEYCHAIN
    item = (b'LUMEN RAW test', f'pytest-{os.getpid()}'.encode())
    monkeypatch.setattr(nl_edit, 'KEYCHAIN_ITEM', item)
    try:
        assert read() is None
        try:
            write(b'{"deepseek": "sk-1"}')
        except OSError as error:
            pytest.skip(f'login keychain not available here: {error}')
        write('{"deepseek": "sk-2 密钥"}'.encode('utf-8'))
        assert read().decode('utf-8') == '{"deepseek": "sk-2 密钥"}'
    finally:
        subprocess.run(['security', 'delete-generic-password', '-s', item[0].decode(), '-a', item[1].decode()],
                       capture_output=True)


def test_thinking_depth_is_mapped_to_each_providers_own_parameters():
    t = nl_edit.thinking_request
    assert t('deepseek', 'default') == {} and t('anthropic', 'default') == {}
    assert t('deepseek', 'off') == {'thinking': {'type': 'disabled'}}
    assert t('deepseek', 'medium') == {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
    assert t('deepseek', 'max')['reasoning_effort'] == 'max'
    assert t('openai', 'off') == {'reasoning_effort': 'none'} and t('openai', 'max') == {'reasoning_effort': 'max'}
    assert t('anthropic', 'off') == {'output_config': {'effort': 'low'}}
    assert t('anthropic', 'high') == {'output_config': {'effort': 'high'}}
    assert t('gemini', 'off') == {'reasoning_effort': 'minimal'} and t('gemini', 'max') == {'reasoning_effort': 'high'}
    assert t('qwen', 'off') == {'enable_thinking': False}
    assert t('qwen', 'low') == {'enable_thinking': True, 'thinking_budget': 1024}
    assert t('budget', 'max') == {'enable_thinking': True, 'thinking_budget': 32768}
    assert t('kimi', 'off', 'kimi-k2.6') == {'thinking': {'type': 'disabled'}}
    assert t('kimi', 'off', 'kimi-k3') == {'reasoning_effort': 'low'}
    assert t('kimi', 'max', 'kimi-k3') == {'reasoning_effort': 'max'}
    assert t('switch', 'high') == {'thinking': {'type': 'enabled'}}
    assert t('ark', 'low') == {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}
    assert t('openrouter', 'off') == {'reasoning': {'effort': 'none'}}
    assert t('ollama', 'off') == {'think': False} and t('ollama', 'max') == {'think': 'high'}
    assert t('lmstudio', 'off') == {'reasoning_effort': 'none'}
    assert t('llamacpp', 'off') == {'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False}}
    assert t('template', 'off') == {'chat_template_kwargs': {'enable_thinking': False}}
    # Every provider has a mapping and a note for the settings dialog.
    for key, entry in nl_edit.PROVIDERS.items():
        assert entry[6] in nl_edit.THINKING_NOTES and nl_edit.thinking_request(entry[6], 'low') != {}


def test_provider_defaults_follow_the_official_endpoints_and_old_defaults_migrate(tmp_path):
    defaults = nl_edit.default_settings()['providers']
    assert defaults['deepseek'] == dict(base_url='https://api.deepseek.com', model='deepseek-flash', key='')
    assert defaults['anthropic']['model'] == 'claude-opus-5-5' and defaults['openai']['model'] == 'gpt-6-luna'
    path = tmp_path / 'nl.json'
    stored = nl_edit.default_settings()
    stored['providers']['deepseek'].update(base_url='https://api.deepseek.com/v1', model='deepseek-chat')
    stored['providers']['zhipu']['model'] = 'glm-4-flash'
    stored['providers']['moonshot'].update(base_url='https://my-proxy.example/v1', model='my-kimi')
    nl_edit.save_settings(stored, path)
    loaded = nl_edit.load_settings(path)['providers']
    assert loaded['deepseek']['base_url'] == 'https://api.deepseek.com' and loaded['deepseek']['model'] == 'deepseek-flash'
    assert loaded['zhipu']['model'] == 'glm-5.3-flash'
    assert loaded['moonshot'] == dict(base_url='https://my-proxy.example/v1', model='my-kimi', key='')
    # Pasted endpoints are reduced to base URLs.
    assert nl_edit.normalize_base('openai', 'https://api.deepseek.com/chat/completions/') == 'https://api.deepseek.com'
    assert nl_edit.normalize_base('anthropic', 'https://api.anthropic.com/v1/messages') == 'https://api.anthropic.com'
    assert nl_edit.normalize_base('ollama', 'http://127.0.0.1:11434/v1') == 'http://127.0.0.1:11434'


def test_thinking_fields_are_sent_and_dropped_when_a_server_rejects_them(fake):
    server = fake(REPLY)
    settings = nl_edit.default_settings()
    settings.update(provider='deepseek', reasoning='off')
    settings['providers']['deepseek'].update(base_url=server.url + '/v1/', key='k')
    usage = {}
    nl_edit.complete(nl_edit.active(settings), nl_edit.build_messages('x', model.recipe()), usage=usage)
    payload = server.requests[-1][3]
    assert payload['thinking'] == {'type': 'disabled'} and payload['response_format'] == {'type': 'json_object'}
    assert 'temperature' not in payload and usage == dict(input=2300, output=186, total=2486)
    server.reject_keys = {'reasoning_effort'}
    settings['reasoning'] = 'high'
    nl_edit.complete(nl_edit.active(settings), nl_edit.build_messages('x', model.recipe()))
    first, second = server.requests[-2][3], server.requests[-1][3]
    assert first['reasoning_effort'] == 'high' and 'reasoning_effort' not in second and 'thinking' not in second
    # Bailian cannot combine JSON mode with thinking; only "off" keeps response_format.
    settings.update(provider='dashscope', reasoning='default')
    settings['providers']['dashscope'].update(base_url=server.url + '/v1', key='k')
    server.reject_keys = set()
    nl_edit.complete(nl_edit.active(settings), nl_edit.build_messages('x', model.recipe()))
    assert 'response_format' not in server.requests[-1][3]
    settings['reasoning'] = 'off'
    nl_edit.complete(nl_edit.active(settings), nl_edit.build_messages('x', model.recipe()))
    assert server.requests[-1][3]['enable_thinking'] is False
    assert server.requests[-1][3]['response_format'] == {'type': 'json_object'}


def test_claude_requests_use_effort_room_for_thinking_and_fallback(fake):
    server = fake(REPLY)
    settings = nl_edit.default_settings()
    settings.update(provider='anthropic', reasoning='low')
    settings['providers']['anthropic'].update(base_url=server.url + '/v1/messages', key='sk-ant')
    usage = {}
    text = nl_edit.complete(nl_edit.active(settings), nl_edit.build_messages('x', model.recipe()), usage=usage)
    assert nl_edit.parse_reply(text)['effects']['vignette'] == -15
    _, path, headers, payload = server.requests[-1]
    assert path == '/v1/messages' and payload['model'] == 'claude-opus-5-5'
    assert payload['output_config'] == {'effort': 'low'} and payload['max_tokens'] >= 16000 and 'temperature' not in payload
    assert payload['fallbacks'] == 'default' and headers['anthropic-beta'] == 'server-side-fallback-2026-07-01'
    assert usage['total'] == 2400
    server.reject_keys = {'fallbacks'}
    nl_edit.complete(nl_edit.active(settings), nl_edit.build_messages('x', model.recipe()))
    retried = server.requests[-1]
    assert 'fallbacks' not in retried[3] and 'anthropic-beta' not in retried[2]
    server.reject_keys, server.stop_reason = set(), 'refusal'
    with pytest.raises(nl_edit.ServiceError, match='拒绝'):
        nl_edit.complete(nl_edit.active(settings), nl_edit.build_messages('x', model.recipe()))


# ----------------------------------------------------------------------------- speech

def test_fbank_matches_the_kaldi_layout():
    banks = speech._mel_banks()
    assert banks.shape == (80, 256) and np.all(banks.sum(axis=1) > 0) and banks[:, 0].max() == 0
    tone = (np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000) * 8000).astype(np.int16)
    features = speech.fbank(tone)
    assert features.shape == (98, 80)
    peak = int(np.argmax(features.mean(axis=0)))
    centers = 700 * (np.exp((1127 * np.log(1 + 20 / 700) + (np.arange(80) + 1) *
                             (1127 * np.log(1 + 8000 / 700) - 1127 * np.log(1 + 20 / 700)) / 81) / 1127) - 1)
    assert abs(centers[peak] - 1000) < 60
    assert speech.lfr(features).shape == (16, 560)


def synthesize(text, culture, path):
    if sys.platform == 'darwin':
        return say(text, culture, path)
    script = (
        'Add-Type -AssemblyName System.Speech;'
        '$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;'
        f'$v=$s.GetInstalledVoices()|Where-Object {{$_.VoiceInfo.Culture.Name -eq "{culture}"}}|Select-Object -First 1;'
        'if(-not $v){exit 3};$s.SelectVoice($v.VoiceInfo.Name);'
        '$f=New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);'
        f'$s.SetOutputToWaveFile("{path}",$f);$s.Speak("{text}");$s.Dispose()')
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script], capture_output=True, timeout=60)
    return result.returncode == 0 and Path(path).is_file()


def say(text, culture, path):
    """macOS: a natural voice of the locale (Samantha / Tingting first), written as 16 kHz 16-bit WAV."""
    import re
    listing = subprocess.run(['say', '-v', '?'], capture_output=True, text=True, timeout=60).stdout
    voices = [m.group(1).strip() for m in re.finditer(r'^(.+?)\s+' + culture.replace('-', '_') + r'\s+#', listing, re.M)]
    preferred = ('Samantha', 'Tingting', 'Alex', 'Ava', 'Eddy', 'Flo', 'Reed', 'Sandy', 'Shelley')
    natural = sorted((v for v in voices if v.startswith(preferred)), key=lambda v: [v.startswith(p) for p in preferred].index(True))
    if not natural:
        return False  # only novelty voices (Albert, Bells, Zarvox, ...) that are not meant to be understood
    result = subprocess.run(['say', '-v', natural[0], '--file-format=WAVE', '--data-format=LEI16@16000',
                             '-o', str(path), text], capture_output=True, timeout=60)
    return result.returncode == 0 and Path(path).is_file()


@pytest.mark.skipif(not speech.available(), reason='SenseVoice model not installed')
@pytest.mark.skipif(os.name != 'nt' and sys.platform != 'darwin', reason='uses Windows or macOS text-to-speech')
@pytest.mark.parametrize('culture,text,expected', [
    ('zh-CN', '把天空调得更蓝一点，整体稍微暖一些', '把天空调得更蓝一点'),
    ('en-US', 'Make the sky bluer and lift the shadows a little', 'make the sky blue'),
])
def test_spoken_instructions_are_recognized_offline(tmp_path, culture, text, expected):
    path = tmp_path / 'voice.wav'
    if not synthesize(text, culture, path):
        pytest.skip(f'no {culture} voice')
    samples = speech.read_wav(path)
    started = time.perf_counter()
    result = speech.transcribe(samples)
    assert expected in result.lower()
    assert time.perf_counter() - started < 5
    # The microphone path delivers 48 kHz float audio that is resampled to 16 kHz.
    from lumen.nl_panel import VoiceRecorder
    recorder, out = VoiceRecorder(), []
    recorder.finished.connect(out.append)
    up = np.interp(np.arange(len(samples) * 3) / 3, np.arange(len(samples)), samples / 32768).astype(np.float32)

    class Format:
        sampleRate = lambda self: 48000
        channelCount = lambda self: 2
        def sampleFormat(self):
            from PySide6.QtMultimedia import QAudioFormat
            return QAudioFormat.SampleFormat.Float

    class Stub:
        stop = deleteLater = lambda self: None

    recorder.format, recorder.source, recorder.device, recorder.timer = Format(), Stub(), None, Stub()
    recorder.chunks = [recorder._samples(np.repeat(up[:, None], 2, axis=1).tobytes())]
    recorder.stop()
    assert abs(len(out[0]) - len(samples)) <= 1 and expected in speech.transcribe(out[0]).lower()


# ----------------------------------------------------------------------------- window

from test_ui import app, window, wait_until  # noqa: E402,F401


def use_service(w, server):
    w.nl_settings = nl_edit.default_settings()
    w.nl_settings.update(provider='lmstudio', attach_image=True, timeout=20)
    w.nl_settings['providers']['lmstudio'].update(base_url=server.url + '/v1', model='fake-model')
    w.nl_update_service()


def test_tab_sits_between_presets_and_snapshots(window):
    tabs = window.library_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == ['预设', '自然语言输入', '快照']


def test_one_click_edit_applies_as_one_step_with_strength_and_undo(window, fake):
    w = window
    server = fake(REPLY)
    use_service(w, server)
    before = copy.deepcopy(w.edits)
    w.nl_input.setPlainText('天空更蓝，整体更暖')
    w.nl_send()
    assert w.nl_busy and w.nl_go.text() == '取消'
    wait_until(lambda: not w.nl_busy and w.nl_plan is not None)
    payload = server.requests[-1][3]
    assert '天空更蓝' in payload['messages'][-1]['content'][0]['text']
    assert payload['messages'][-1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert w.edits['adjustments']['exposure'] == .3 and w.edits['masks'][-1]['kind'] == 'linear'
    assert w.controls['exposure'].spin.value() == .3
    assert w.nl_changes.count() >= 7 and '天空更蓝' in w.nl_explanation.text()
    # Strength: the user-facing spin box drives a blend from the starting point.
    w.nl_amount.spin.setValue(50)
    assert w.edits['adjustments']['exposure'] == pytest.approx(.15)
    w.nl_amount.spin.setValue(100)
    w.commit()
    # A single undo returns to the state before the request.
    w.undo(-1)
    assert w.edits['adjustments'] == before['adjustments'] and w.edits['masks'] == before['masks']
    w.undo(1)
    w.nl_undo()
    assert w.edits['adjustments'] == before['adjustments']
    assert not w.nl_explanation.isVisible()


def test_slider_edits_after_the_result_release_the_strength_handle(window, fake):
    w = window
    use_service(w, fake({'adjustments': {'contrast': 30}}))
    w.nl_send('更有对比')
    wait_until(lambda: not w.nl_busy and w.nl_plan is not None)
    assert w.nl_amount.isEnabled()
    w.controls['saturation'].spin.setValue(25)
    w.nl_amount.spin.setValue(10)
    assert w.edits['adjustments']['contrast'] == 30 and w.edits['adjustments']['saturation'] == 25
    assert not w.nl_amount.isEnabled() and not w.nl_revert.isEnabled()


def test_regenerate_starts_again_from_the_same_point(window, fake):
    w = window
    server = fake({'adjustments': {'exposure': .6}})
    use_service(w, server)
    w.nl_send('更亮')
    wait_until(lambda: not w.nl_busy and w.nl_plan is not None)
    server.reply = {'adjustments': {'exposure': .4}}
    w.nl_regenerate()
    wait_until(lambda: not w.nl_busy and w.nl_plan is not None and w.edits['adjustments']['exposure'] == .4)
    last = server.requests[-1][3]['messages']
    assert len(last) == 2 and '"exposure": 0.0' in last[-1]['content'][0]['text']
    assert len(w.nl_history) == 1


def test_cancel_and_errors_leave_the_photo_unchanged(window, fake):
    w = window
    server = fake(REPLY)
    use_service(w, server)
    before = copy.deepcopy(w.edits)
    w.nl_send('更暖')
    w.nl_send()  # the button reads 取消 while waiting
    assert not w.nl_busy and '取消' in w.nl_status.text()
    time.sleep(.3)
    wait_until(lambda: True)
    assert w.edits == before
    server.reply = 'Sorry, I can only chat.'
    w.nl_send('更暖')
    wait_until(lambda: not w.nl_busy)
    assert 'JSON' in w.nl_status.text() and w.edits == before


def test_voice_instruction_is_recognized_and_sent(window, fake, tmp_path):
    if not speech.available() or (os.name != 'nt' and sys.platform != 'darwin') or not synthesize('整体更暖一点', 'zh-CN', tmp_path / 'v.wav'):
        pytest.skip('speech model or Chinese voice unavailable')
    w = window
    server = fake({'adjustments': {'temperature': 15}})
    use_service(w, server)
    w.nl_recorded(speech.read_wav(tmp_path / 'v.wav'))
    wait_until(lambda: w.nl_plan is not None and not w.nl_busy, 30)
    assert '更暖' in w.nl_input.toPlainText() and w.edits['adjustments']['temperature'] == 15
    assert '更暖' in server.requests[-1][3]['messages'][-1]['content'][0]['text']


def test_region_masks_are_found_on_the_lens_corrected_frame(window, fake, monkeypatch):
    # Masks refer to the lens-corrected frame (1.5.1); the language path used the uncorrected
    # one, so with corrections on its masks sat beside the subject's edges.
    from lumen import engine, lens, selection
    w = window
    entry = lens.database().find('Nikon', 'Nikkor Z 14-24mm f/2.8 S')
    w.edits['lens'] = {**lens.defaults(), 'enabled': True, 'profile': lens.solve(entry, 14, 4, 1000, 1.0)[0]}
    seen = []
    def automatic(rgb, kind, cuda=False):
        seen.append(rgb.copy())
        return np.ones(rgb.shape[:2], np.float32), 'stub'
    monkeypatch.setattr(selection, 'automatic', automatic)
    use_service(w, fake({'masks': [{'region': 'sky', 'adjustments': {'exposure': -.3}}]}))
    w.nl_send('天空暗一点')
    wait_until(lambda: not w.nl_busy and w.nl_plan is not None)
    assert w.edits['masks'][-1]['kind'] == 'sky' and w.edits['masks'][-1]['feather'] == 3 and len(seen) == 1
    np.testing.assert_allclose(seen[0], engine.develop_view(lens.correct(w.source, w.edits), w.edits), atol=1e-6)
    assert not np.allclose(seen[0], engine.develop_view(w.source, w.edits), atol=1e-3)
