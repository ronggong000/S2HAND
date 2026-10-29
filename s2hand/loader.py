#!/usr/bin/env python3
"""Download and load official Sapiens2 + S2HAND replacement tensors.

Returns 308-channel heatmaps for normalized RGB tensors. This loader preserves
the final complete-decoder routing; it does not require a research checkout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

REPO = 'mage0001/S2HAND'
WEIGHT_REVISION = 'e37456b131139e186d9cb5552beb3e99820004d4'
BASES = {
    '0.4b': '73572e121602e8a6eb40eb841732e88f386d250b',
    '0.8b': '76f06b9854d04bb1e59e0624de39b1e8921b12aa',
    '1b': 'f5fed8b97b99698d5eea1d14ff0855d0b4c3f000',
    '5b': 'ada1f29aa1fd454ca28665c700923a0101b6b24f',
}
NAMES = ('S2HAND-MT-5B', 'S2HAND-Pose-5B', 'S2HAND-Pose-1B', 'S2HAND-Pose-0.8B', 'S2HAND-Pose-0.4B')
TRAINING_WARM_START = 'S2HAND-Training-WarmStart-5B'
# Filled with the exact bundled official standalone source fingerprint.
BACKBONE_SHA256 = '7046d482da440cbedf56d9c7fd5596fa0fb35747bb7dc9d34d3c5a638d8c51db'


def verify(path, expected):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError(f'SHA256 mismatch: {path}')


def prepare(model_name='S2HAND-MT-5B', *, cache_dir=None, base_path=None, delta_path=None, manifest_path=None, local_files_only=False):
    """Download pinned weights once, or verify supplied local files offline."""
    if model_name not in NAMES:
        raise ValueError(f'Choose one of {NAMES}')
    scale = model_name.rsplit('-', 1)[1].lower()
    options = dict(cache_dir=cache_dir, local_files_only=local_files_only)
    manifest_path = manifest_path or hf_hub_download(REPO, f'models/{model_name}/manifest.json', revision=WEIGHT_REVISION, **options)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['name'] != model_name:
        raise ValueError('Manifest names a different model')
    base_path = base_path or hf_hub_download(f'facebook/sapiens2-pose-{scale}', manifest['official_base']['filename'], revision=BASES[scale], **options)
    delta_path = delta_path or hf_hub_download(REPO, manifest['file'], revision=WEIGHT_REVISION, **options)
    print(f'Verifying {model_name} base and delta SHA256...', flush=True)
    verify(base_path, manifest['official_base']['sha256'])
    verify(delta_path, manifest['sha256'])
    return Path(base_path), Path(delta_path), manifest


def prepare_training_warm_start(*, cache_dir=None, base_path=None,
                                delta_path=None, manifest_path=None,
                                local_files_only=False):
    """Download the pinned common 5B training/Fisher initialization.

    This compact artifact is not a deployment model. It reconstructs the
    shared initialization from which the final MT and Pose branches were
    trained independently.
    """
    options = dict(cache_dir=cache_dir, local_files_only=local_files_only)
    prefix = f'training/{TRAINING_WARM_START}'
    manifest_path = manifest_path or hf_hub_download(
        REPO, f'{prefix}/manifest.json', revision=WEIGHT_REVISION, **options)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('name') != TRAINING_WARM_START:
        raise ValueError('Manifest names a different training artifact')
    if manifest.get('deployment_model') is not False:
        raise ValueError('Training warm-start must be marked non-deployment')
    base = manifest['official_base']
    base_path = base_path or hf_hub_download(
        base['repo'], base['filename'], revision=base['revision'], **options)
    delta_path = delta_path or hf_hub_download(
        REPO, manifest['file'], revision=WEIGHT_REVISION, **options)
    print('Verifying 5B training warm-start base and delta SHA256...', flush=True)
    verify(base_path, base['sha256'])
    verify(delta_path, manifest['sha256'])
    return Path(base_path), Path(delta_path), manifest


def backbone_class(cache_dir=None, local_files_only=False):
    source = Path(__file__).with_name('sapiens2_backbone.py')
    if not source.is_file():
        source = Path(hf_hub_download(REPO, 'sapiens2_backbone.py', cache_dir=cache_dir, local_files_only=local_files_only))
    verify(source, BACKBONE_SHA256)
    spec = importlib.util.spec_from_file_location('_s2hand_official_backbone', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sapiens2


class RoutedPoseHead(nn.Module):
    def __init__(self, channels, deconv_channels=(1024,768),
                 decoder_channels=(512,512,256),
                 allow_protected_input_grad=False):
        super().__init__()
        self.allow_protected_input_grad = bool(allow_protected_input_grad)
        first, second = deconv_channels
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(channels, first, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(first), nn.SiLU(inplace=True),
            nn.ConvTranspose2d(first, second, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(second), nn.SiLU(inplace=True),
        )
        layers = []
        cin = second
        for cout in decoder_channels:
            layers += [nn.Conv2d(cin, cout, 1), nn.InstanceNorm2d(cout), nn.SiLU(inplace=True)]
            cin = cout
        self.conv_layers = nn.Sequential(*layers)
        self.conv_pose = nn.Conv2d(cin, 308, 1)
        self.public_decoder = None
        self.public_projection = None
        protected = torch.ones(308, dtype=torch.bool, device='cpu')
        protected[21:63] = False
        protected[[41, 62]] = True
        self.register_buffer('protected_mask', protected, persistent=False)

    def freeze_public_route(self):
        self.public_decoder = copy.deepcopy(self.conv_layers).requires_grad_(False)
        self.public_projection = copy.deepcopy(self.conv_pose).requires_grad_(False)

    def forward(self, features):
        features = self.deconv_layers(features)
        adapted = self.conv_pose(self.conv_layers(features))
        context = torch.enable_grad if self.allow_protected_input_grad else torch.no_grad
        with context():
            public = self.public_projection(self.public_decoder(features))
        return torch.where(self.protected_mask[None, :, None, None], public, adapted)


class S2Hand(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.decode_head = head

    def forward(self, normalized_rgb):
        """B×3×1024×768 RGB, ImageNet normalized -> B×308×256×192 heatmaps."""
        return self.decode_head(self.backbone(normalized_rgb)[0])


def _load_prepared(model_name, scale, base, delta, manifest, *, device='cpu',
                   cache_dir=None, local_files_only=False):
    """Construct a routed model from verified base and replacement files.

    Uses meta initialization and mmap-backed safetensors to avoid allocating a
    second randomly initialized 5B model. Keep model parameters in FP32; use
    autocast for optional BF16 inference, as in the research pipeline.
    """
    cls = backbone_class(cache_dir, local_files_only)
    widths = {'0.4b':1024, '0.8b':1280, '1b':1536, '5b':2432}
    state = load_file(str(base), device='cpu')
    # Decoder widths differ by scale; obtain the exact architecture from base tensors.
    deconv_channels = tuple(state[f'decode_head.deconv_layers.{i}.weight'].shape[1] for i in (0,3))
    decoder_channels = tuple(state[f'decode_head.conv_layers.{i}.weight'].shape[0] for i in (0,3,6))
    with torch.device('meta'):
        backbone = cls(arch=f'sapiens2_{scale}', img_size=(1024, 768), patch_size=16,
                       out_type='featmap', pos_embed_rope_dtype='bf16')
        model = S2Hand(backbone, RoutedPoseHead(widths[scale], deconv_channels, decoder_channels))
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Original runtime uses BF16 RoPE periods; 5B deltas explicitly override it.
    model.backbone.rope_embed.periods = model.backbone.rope_embed.periods.to(torch.bfloat16)
    model.decode_head.freeze_public_route()
    replacements = load_file(str(delta), device='cpu')
    if set(replacements) != set(manifest['tensor_names']):
        raise ValueError('Delta tensor names differ from manifest')
    current = model.state_dict()
    for name, value in replacements.items():
        if name not in current or value.shape != current[name].shape:
            raise ValueError(f'Invalid replacement tensor: {name}')
    # Copy into the runtime tensor dtypes, especially the BF16 RoPE buffer,
    # exactly as the research loader does. assign=True would change its dtype.
    model.load_state_dict(replacements, strict=False)
    del replacements, current
    model.requires_grad_(False).eval()
    model.model_name = model_name
    return model.to(device)


def load_model(model_name='S2HAND-MT-5B', *, device='cpu', cache_dir=None,
               base_path=None, delta_path=None, manifest_path=None,
               local_files_only=False):
    """Load a final deployment model with its protected decoder route."""
    base, delta, manifest = prepare(
        model_name, cache_dir=cache_dir, base_path=base_path,
        delta_path=delta_path, manifest_path=manifest_path,
        local_files_only=local_files_only)
    scale = model_name.rsplit('-', 1)[1].lower()
    return _load_prepared(
        model_name, scale, base, delta, manifest, device=device,
        cache_dir=cache_dir, local_files_only=local_files_only)


def load_training_warm_start(*, device='cpu', cache_dir=None, base_path=None,
                             delta_path=None, manifest_path=None,
                             local_files_only=False, return_manifest=False):
    """Load the exact common 5B initialization for training/Fisher probes."""
    base, delta, manifest = prepare_training_warm_start(
        cache_dir=cache_dir, base_path=base_path, delta_path=delta_path,
        manifest_path=manifest_path, local_files_only=local_files_only)
    model = _load_prepared(
        TRAINING_WARM_START, '5b', base, delta, manifest, device=device,
        cache_dir=cache_dir, local_files_only=local_files_only)
    return (model, manifest) if return_manifest else model


def load_training_checkpoint(delta_path, manifest_path=None, *, device='cpu',
                             cache_dir=None, base_path=None,
                             local_files_only=False):
    """Load a delta written by ``tools/train.py`` over its pinned base."""
    delta_path = Path(delta_path)
    manifest_path = Path(manifest_path or delta_path.with_suffix('.json'))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 's2hand_training_checkpoint_v1':
        raise ValueError('Unsupported training checkpoint manifest')
    if Path(manifest.get('file', '')).name != delta_path.name:
        raise ValueError('Training manifest names a different delta file')
    if manifest.get('scale') != '5b':
        raise ValueError('Released standalone trainer currently supports 5B')
    base = manifest['official_base']
    if base_path is None:
        base_path = hf_hub_download(
            base['repo'], base['filename'], revision=base['revision'],
            cache_dir=cache_dir, local_files_only=local_files_only,
        )
    verify(base_path, base['sha256'])
    verify(delta_path, manifest['sha256'])
    return _load_prepared(
        manifest.get('name', 'S2HAND-Custom-5B'), '5b', Path(base_path),
        delta_path, manifest, device=device, cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=NAMES, default=NAMES[0])
    parser.add_argument('--cache-dir')
    parser.add_argument('--base-path')
    parser.add_argument('--delta-path')
    parser.add_argument('--manifest-path')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--input-tensor', help='Safetensors file containing ImageNet-normalized RGB tensor named images')
    parser.add_argument('--output', default='s2hand_heatmaps.safetensors')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 autocast during inference; keep FP32 parameters')
    args = parser.parse_args()
    options = dict(cache_dir=args.cache_dir, base_path=args.base_path, delta_path=args.delta_path,
                   manifest_path=args.manifest_path, local_files_only=args.offline)
    if args.download_only:
        base, delta, _ = prepare(args.model, **options)
        print(f'Base: {base}\nDelta: {delta}\nReady for load_model().')
        return
    model = load_model(args.model, device=args.device, **options)
    print(f'Loaded {args.model}: full base + delta + protected decoder route.', flush=True)
    if args.input_tensor:
        from safetensors.torch import save_file
        images = load_file(args.input_tensor)['images']
        if images.ndim != 4 or tuple(images.shape[1:]) != (3,1024,768):
            raise ValueError('Expected images shape B×3×1024×768; RGB with ImageNet normalization')
        with torch.inference_mode(), torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.bf16):
            heatmaps = model(images.to(device=args.device, dtype=torch.float32))
        save_file({'heatmaps':heatmaps.float().cpu().contiguous()},args.output)
        print(f'Saved heatmaps: {args.output}')


if __name__ == '__main__':
    main()
