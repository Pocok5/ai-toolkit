"""Quantization cache utilities.

After the first run, the quantized transformer state dict is written to a local
directory so that subsequent runs can restore it directly without re-running the
expensive block-by-block quantization pass.

Cache directory layout
----------------------
    <cache_root>/<cache_key>/
        config.json         – diffusers model config (architecture only)
        model.safetensors   – raw quantized state dict
        cache_info.json     – metadata: qtype, model class name, created_at

The raw state dict is the unpatched one (bypassing ``patch_dequantization_on_save``)
so that optimum-quanto ``weight._data`` / ``weight._scale`` sub-keys are preserved
as plain fp8/int8 tensors and can be round-tripped through safetensors.

Cache key
---------
``MD5(name_or_path + "::" + qtype + "::" + cache_tag)[:16]``

Set ``use_quantize_cache: false`` in the model config to opt out entirely.
Set ``quantize_cache_dir`` to override the default ``~/.cache/ai-toolkit/quantized``
root.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional, Type, TypeVar

import torch
from safetensors.torch import load_file, save_file

from toolkit.print import print_acc

T = TypeVar("T", bound=torch.nn.Module)

_DEFAULT_CACHE_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache", "ai-toolkit", "quantized"
)
_CACHE_INFO_FILE = "cache_info.json"
_WEIGHTS_FILE = "model.safetensors"
_CONFIG_FILE = "config.json"
_CURRENT_CACHE_VERSION = 1


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_cache_dir(
    name_or_path: str,
    qtype: str,
    cache_root: Optional[str] = None,
    cache_tag: Optional[str] = None,
) -> str:
    """Return the full path of the cache directory for the given model."""
    root = cache_root or _DEFAULT_CACHE_ROOT
    raw = f"{name_or_path}::{qtype}"
    if cache_tag:
        raw += f"::{cache_tag}"
    key = hashlib.md5(raw.encode()).hexdigest()[:16]
    return os.path.join(root, key)


def has_valid_cache(cache_dir: str) -> bool:
    """Return ``True`` if a complete, loadable cache exists at *cache_dir*."""
    for fname in (_CACHE_INFO_FILE, _WEIGHTS_FILE, _CONFIG_FILE):
        if not os.path.exists(os.path.join(cache_dir, fname)):
            return False
    # Validate cache version
    try:
        with open(os.path.join(cache_dir, _CACHE_INFO_FILE)) as f:
            info = json.load(f)
        if info.get("cache_version", 0) != _CURRENT_CACHE_VERSION:
            return False
    except Exception:
        return False
    return True


def save_quantized_cache(
    model: torch.nn.Module,
    cache_dir: str,
    qtype: str,
) -> None:
    """Persist *model*'s quantized state dict to *cache_dir*.

    Saves the unpatched state dict so that optimum-quanto ``._data`` /
    ``._scale`` sub-keys are stored as plain fp8/int8 tensors rather than being
    dequantized first.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Retrieve the raw (unpatched) state dict.  ``patch_dequantization_on_save``
    # replaces ``model.state_dict`` with a dequantizing version; the original is
    # preserved as ``model.orig_state_dict``.
    if hasattr(model, "orig_state_dict"):
        sd = model.orig_state_dict()
    else:
        # Bypass any instance-level override by calling through the class.
        sd = type(model).state_dict(model)

    # Move to CPU and detach – safetensors requires contiguous CPU tensors.
    # Tensor subclasses such as QTensor (optimum-quanto) and AffineQuantizedTensor
    # (torchao) do not expose a usable raw storage pointer, so safetensors cannot
    # serialize them directly.  We expand optimum-quanto QTensors into their
    # raw ._data / ._scale components (preserving the quantized representation)
    # and dequantize any other tensor subclass as a fallback.
    # NOTE: `type(t) is not torch.Tensor` (exact-type check) intentionally matches
    # *only* plain tensors.  `isinstance` would return True for subclasses too,
    # which is exactly what we want to avoid here.
    cpu_sd: dict = {}
    for key, tensor in sd.items():
        try:
            t = tensor.detach()
            if type(t) is not torch.Tensor:
                if hasattr(t, '_data') and hasattr(t, '_scale'):
                    # optimum-quanto QTensor: store the raw quantized components
                    cpu_sd[key + "._data"] = t._data.detach().cpu().contiguous()
                    cpu_sd[key + "._scale"] = t._scale.detach().cpu().contiguous()
                elif hasattr(t, 'dequantize'):
                    # torchao or other subclass: dequantize to a plain float tensor
                    cpu_sd[key] = t.dequantize().detach().cpu().contiguous()
                else:
                    print_acc(
                        f"[quantize_cache] Skipping unsupported tensor subclass "
                        f"{type(t).__name__!r} for key {key!r}"
                    )
            else:
                cpu_sd[key] = t.cpu().contiguous()
        except Exception as exc:
            print_acc(f"[quantize_cache] Skipping key {key!r} ({exc})")

    if not cpu_sd:
        raise RuntimeError("quantize_cache: state dict is empty; nothing to save.")

    # Weights
    save_file(cpu_sd, os.path.join(cache_dir, _WEIGHTS_FILE))

    # Model config (architecture only, no weights)
    try:
        model.save_config(cache_dir)
    except AttributeError:
        # Fallback for models that don't implement save_config
        try:
            cfg = model.config.to_dict() if hasattr(model.config, "to_dict") else dict(model.config)
            with open(os.path.join(cache_dir, _CONFIG_FILE), "w") as f:
                json.dump(cfg, f, indent=2)
        except Exception as exc:
            raise RuntimeError(
                f"quantize_cache: could not save model config: {exc}"
            ) from exc

    # Cache metadata
    info = {
        "cache_version": _CURRENT_CACHE_VERSION,
        "qtype": qtype,
        "model_class": type(model).__name__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(cache_dir, _CACHE_INFO_FILE), "w") as f:
        json.dump(info, f, indent=2)

    print_acc(f"[quantize_cache] Saved to {cache_dir}")


def load_quantized_cache(
    cache_dir: str,
    model_class: Type[T],
    qtype: str,
) -> T:
    """Reconstruct a quantized model from *cache_dir*.

    Steps:
    1. Create the model architecture from the saved ``config.json``.
    2. Apply ``quantize + freeze`` to set up the quantized module structure
       (this step is fast – it only replaces ``nn.Linear`` → ``QLinear``
       and does not involve a full-precision→fp8 data conversion, as the
       weights will be overwritten in step 3).
    3. Load the cached ``._data`` / ``._scale`` state dict into the model.
    4. Patch ``state_dict`` so that LoRA merging / saving works correctly.
    """
    from optimum.quanto import freeze

    from toolkit.dequantize import patch_dequantization_on_save
    from toolkit.util.quantize import get_qtype, quantize

    # 1. Rebuild architecture from saved config.
    config = model_class.load_config(cache_dir)
    model: T = model_class(**config)

    # 2. Set up quantized module structure (QLinear etc.).
    qtype_obj = get_qtype(qtype)
    quantize(model, weights=qtype_obj)
    freeze(model)

    # 3. Restore quantized weights from cache.
    cached_sd = load_file(os.path.join(cache_dir, _WEIGHTS_FILE))
    # assign=True (requires PyTorch >= 2.1) assigns tensors directly rather
    # than copying, which is necessary when the model may contain meta tensors
    # and avoids shape-mismatch issues with quantized tensor subclasses.
    result = model.load_state_dict(cached_sd, strict=False, assign=True)
    if result.missing_keys:
        head = result.missing_keys[:5]
        tail = f"... (+{len(result.missing_keys)-5} more)" if len(result.missing_keys) > 5 else ""
        print_acc(f"[quantize_cache] Warning – missing keys: {head}{tail}")
    if result.unexpected_keys:
        head = result.unexpected_keys[:5]
        tail = f"... (+{len(result.unexpected_keys)-5} more)" if len(result.unexpected_keys) > 5 else ""
        print_acc(f"[quantize_cache] Warning – unexpected keys: {head}{tail}")

    # 4. Patch so that LoRA saves are dequantized correctly.
    patch_dequantization_on_save(model)

    print_acc(f"[quantize_cache] Loaded from {cache_dir}")
    return model
