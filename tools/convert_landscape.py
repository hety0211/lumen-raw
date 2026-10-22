"""Export EfficientViT-Seg (ADE20K) as the app's landscape mask network (developer-only, 2.1.0).

Architecture and weights: MIT Han Lab EfficientViT, Apache-2.0
(https://github.com/mit-han-lab/efficientvit, https://huggingface.co/han-cai/efficientvit-seg).
The graph takes ImageNet-normalised RGB [1, 3, H, W] (H, W multiples of 32; the short side
about 512) and returns, at 1/8 resolution, the softmax probability of each landscape group:
the ADE20K classes of ``GROUPS`` summed.  Needs PyTorch, which the app does not ship.
"""
import argparse
import sys
import types
import importlib.util
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

# ADE20K-150 class indices (0-based, EfficientViT's ADE20KDataset.classes order).
GROUPS = {
    'water': [21, 26, 60, 109, 113, 128],                # water, sea, river, swimming pool, waterfall, lake
    'vegetation': [4, 9, 17, 29, 66, 72],                # tree, grass, plant, field, flower, palm
    'mountain': [16, 34, 68],                            # mountain, rock, hill
    'building': [0, 1, 25, 42, 48, 61, 79, 84, 140],     # wall, building, house, column, skyscraper, bridge, hovel, tower, pier
    'ground': [3, 6, 11, 13, 46, 52, 54, 91, 94],        # floor, road, sidewalk, earth, sand, path, runway, dirt track, land
}


def load_efficientvit(root):
    """Import only the segmentation modules (the package __init__ files pull in Triton, SAM, …)."""
    root = Path(root) / 'efficientvit'
    def package(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
        return module
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    package('efficientvit', root)
    package('efficientvit.models', root / 'models')
    utils = package('efficientvit.models.utils', root / 'models' / 'utils')
    for part in ('list', 'network'):
        vars(utils).update({k: v for k, v in vars(load(f'efficientvit.models.utils.{part}', root / 'models' / 'utils' / f'{part}.py')).items() if not k.startswith('_')})
    nn = package('efficientvit.models.nn', root / 'models' / 'nn')
    stub = types.ModuleType('efficientvit.models.nn.triton_rms_norm')
    stub.TritonRMSNorm2dFunc = None
    sys.modules[stub.__name__] = stub
    for part in ('act', 'norm', 'ops'):
        vars(nn).update({k: v for k, v in vars(load(f'efficientvit.models.nn.{part}', root / 'models' / 'nn' / f'{part}.py')).items() if not k.startswith('_')})
    package('efficientvit.models.efficientvit', root / 'models' / 'efficientvit')
    load('efficientvit.models.efficientvit.backbone', root / 'models' / 'efficientvit' / 'backbone.py')
    return load('efficientvit.models.efficientvit.seg', root / 'models' / 'efficientvit' / 'seg.py'), nn


class Landscape(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        groups = torch.zeros(len(GROUPS), 150)
        for row, classes in enumerate(GROUPS.values()):
            groups[row, classes] = 1
        self.register_buffer('groups', groups)

    def forward(self, image):
        probability = self.model(image).softmax(1)
        return torch.einsum('gc,bchw->bghw', self.groups, probability)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='EfficientViT checkout (mit-han-lab/efficientvit)')
    parser.add_argument('weights', type=Path, help='efficientvit_seg_<variant>_ade20k.pt')
    parser.add_argument('output', type=Path)
    parser.add_argument('--variant', default='b2')
    args = parser.parse_args()
    seg, nn = load_efficientvit(args.source)
    model = getattr(seg, f'efficientvit_seg_{args.variant}')(dataset='ade20k')
    nn.set_norm_eps(model, 1e-5 if args.variant.startswith('b') else 1e-7)
    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state.get('state_dict', state))
    wrapped = Landscape(model).eval()
    sample = torch.rand(1, 3, 512, 768)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapped, sample, str(args.output), input_names=['image'], output_names=['groups'],
                      dynamic_axes={'image': {2: 'height', 3: 'width'}, 'groups': {2: 'h8', 3: 'w8'}},
                      opset_version=17, dynamo=False)
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    for shape in ((1, 3, 512, 768), (1, 3, 704, 512)):
        sample = torch.rand(*shape)
        actual = session.run(None, {'image': sample.numpy()})[0]
        with torch.no_grad():
            expected = wrapped(sample).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
        print(f'{shape}: maximum absolute error {np.max(np.abs(actual - expected)):.3g}')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
