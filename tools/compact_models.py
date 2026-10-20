"""Store the bundled ONNX models' weights as float16 for the packages (1.8.0).

Every large float32 initializer becomes a float16 initializer followed by a ``Cast`` back to
float32.  ONNX Runtime folds those casts while it loads a model (constant folding is a basic
optimization; it runs before the graph is split between execution providers), so the CPU,
DirectML, CUDA and Core ML providers all still compute in float32 and only the file shrinks to
about half.  Rounding the weights to float16 moves the outputs by about 1e-5 of their range
(tests/test_v180.py).

Like ``lumen/onnx_graph.py`` this needs no ``onnx`` package: the protobuf is rewritten directly.
The sources in ``assets/models`` stay the pinned float32 files of ``assets/runtime-assets.json``;
the PyInstaller specs bundle the compact copies written to ``build/compact-models``.

Usage: python tools/compact_models.py <source folder> <destination folder>
"""
from __future__ import annotations

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

#: Below this many elements a tensor stays float32 (biases, norms, PReLU slopes): no real saving.
MIN_ELEMENTS = 1024
#: The U²-Net-p matte (4 MB) keeps float32: its soft mask edges moved by up to 6 % in float16.
#: 2.0.0: so does the YuNet face detector (0.2 MB), which OpenCV's DNN runs, not ONNX Runtime.
KEEP_FLOAT32 = frozenset({'u2netp.onnx', 'face-yunet.onnx'})
FLOAT, FLOAT16 = 1, 10
FLOAT16_MAX = 65504.
#: Field numbers of onnx.proto3.
MODEL_GRAPH = 7
GRAPH_NODE, GRAPH_INITIALIZER, GRAPH_INPUT = 1, 5, 11
TENSOR_DIMS, TENSOR_TYPE, TENSOR_FLOATS, TENSOR_NAME, TENSOR_RAW, TENSOR_EXTERNAL, TENSOR_LOCATION = 1, 2, 4, 8, 9, 13, 14
VALUE_NAME = 1


def _varint(data, position):
    result = shift = 0
    while True:
        byte = data[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, position
        shift += 7


def _encode_varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def fields(data):
    """``(number, wire type, value, raw field bytes)`` of a serialized message; ``value`` is an int
    for varint / fixed fields and a memoryview of the payload for length-delimited ones."""
    view = memoryview(data)
    position = 0
    while position < len(view):
        start = position
        key, position = _varint(view, position)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, position = _varint(view, position)
        elif wire == 1:
            value, position = int.from_bytes(view[position:position + 8], 'little'), position + 8
        elif wire == 2:
            length, position = _varint(view, position)
            value, position = view[position:position + length], position + length
        elif wire == 5:
            value, position = int.from_bytes(view[position:position + 4], 'little'), position + 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
        yield number, wire, value, view[start:position]


def _field(number, payload):
    return _encode_varint(number << 3 | 2) + _encode_varint(len(payload)) + bytes(payload)


def _int_field(number, value):
    return _encode_varint(number << 3) + _encode_varint(value)


def _float32_values(tensor):
    """(name, dims field bytes, float32 array) of a float32 tensor stored in the file, else None."""
    name, dims, kind, raw, floats, external = '', [], None, None, [], False
    for number, wire, value, raw_field in fields(tensor):
        if number == TENSOR_NAME:
            name = bytes(value).decode('utf-8')
        elif number == TENSOR_DIMS:
            dims.append(bytes(raw_field))
        elif number == TENSOR_TYPE:
            kind = value
        elif number == TENSOR_RAW:
            raw = value
        elif number == TENSOR_FLOATS:
            if wire == 2:
                floats.append(np.frombuffer(value, '<f4'))
            else:
                floats.append(np.array([struct.unpack('<f', value.to_bytes(4, 'little'))[0]], np.float32))
        elif number in (TENSOR_EXTERNAL, TENSOR_LOCATION) and (number == TENSOR_EXTERNAL or value):
            external = True
    if kind != FLOAT or external:
        return None
    if raw is not None:
        values = np.frombuffer(raw, '<f4')
    elif floats:
        values = np.concatenate(floats)
    else:
        return None
    return name, b''.join(dims), values


def compact_model(data):
    """The model ``data`` (bytes) with large float32 weights stored as float16, and the number
    of tensors converted.  Unchanged bytes come back when nothing qualifies."""
    model = list(fields(data))
    graphs = [i for i, f in enumerate(model) if f[0] == MODEL_GRAPH]
    if len(graphs) != 1:
        raise ValueError('expected one graph in the model')
    graph = list(fields(model[graphs[0]][2]))
    names = set()
    for number, _, value, _ in graph:
        if number == GRAPH_NODE:
            for n, _, v, _ in fields(value):
                if n in (1, 2):
                    names.add(bytes(v).decode('utf-8'))
        elif number in (GRAPH_INITIALIZER, GRAPH_INPUT):
            for n, _, v, _ in fields(value):
                if n == (TENSOR_NAME if number == GRAPH_INITIALIZER else VALUE_NAME):
                    names.add(bytes(v).decode('utf-8'))
    casts, body, converted = [], [], set()
    for number, wire, value, raw_field in graph:
        if number == GRAPH_INITIALIZER:
            found = _float32_values(value)
            if found is not None:
                name, dims, values = found
                half_name = name + '__fp16'
                if (values.size >= MIN_ELEMENTS and half_name not in names and np.isfinite(values).all()
                        and float(np.abs(values).max(initial=0)) <= FLOAT16_MAX):
                    half = values.astype('<f2')
                    tensor = dims + _int_field(TENSOR_TYPE, FLOAT16) + _field(TENSOR_NAME, half_name.encode('utf-8'))
                    tensor += _field(TENSOR_RAW, half.tobytes())
                    body.append(_field(GRAPH_INITIALIZER, tensor))
                    attribute = _field(1, b'to') + _int_field(3, FLOAT) + _int_field(20, 2)
                    node = (_field(1, half_name.encode('utf-8')) + _field(2, name.encode('utf-8'))
                            + _field(3, (name + '__cast').encode('utf-8')) + _field(4, b'Cast') + _field(5, attribute))
                    casts.append(_field(GRAPH_NODE, node))
                    converted.add(name)
                    continue
        body.append(bytes(raw_field))
    if not converted:
        return bytes(data), 0
    # IR < 4 models also list their initializers as graph inputs; a converted one is now produced
    # by its Cast, so it must not stay a (required) input.
    kept = []
    for part in body:
        number, _, value, _ = next(fields(part))
        if number == GRAPH_INPUT:
            name = next((bytes(v).decode('utf-8') for n, _, v, _ in fields(value) if n == VALUE_NAME), '')
            if name in converted:
                continue
        kept.append(part)
    # The casts come first so the node list stays in topological order.
    new_graph = b''.join(casts) + b''.join(kept)
    out = [bytes(f[3]) if i != graphs[0] else _field(MODEL_GRAPH, new_graph) for i, f in enumerate(model)]
    return b''.join(out), len(converted)


def compact_folder(source, destination, log=print):
    """Write compact copies of ``source/*.onnx`` (and copy the other files there) into
    ``destination``; unchanged sources are skipped.  Returns ``destination``.

    2.0.0: a copy is only reused while it is still the file this function wrote (size and
    modification time are recorded); anything else in ``destination`` is written again."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    state_file = destination.with_name(destination.name + '.state.json')
    try:
        state = json.loads(state_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    wanted = set()
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        wanted.add(item.name)
        stamp = [item.stat().st_size, item.stat().st_mtime_ns, MIN_ELEMENTS, item.name in KEEP_FLOAT32]
        target = destination / item.name
        recorded = state.get(item.name)
        if isinstance(recorded, dict) and recorded.get('source') == stamp and target.is_file() and \
                recorded.get('target') == [target.stat().st_size, target.stat().st_mtime_ns]:
            continue
        if item.suffix.lower() == '.onnx' and item.name not in KEEP_FLOAT32:
            data, count = compact_model(item.read_bytes())
            target.write_bytes(data)
            log(f'{item.name}: {item.stat().st_size / 2**20:.1f} MB -> {len(data) / 2**20:.1f} MB '
                f'({count} float16 tensors)')
        else:
            shutil.copy2(item, target)
        state[item.name] = dict(source=stamp, target=[target.stat().st_size, target.stat().st_mtime_ns])
    for stale in destination.iterdir():
        if stale.is_file() and stale.name not in wanted:
            stale.unlink()
    state = {k: v for k, v in state.items() if k in wanted}
    state_file.write_text(json.dumps(state, indent=1), encoding='utf-8')
    return destination


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    compact_folder(sys.argv[1], sys.argv[2])
