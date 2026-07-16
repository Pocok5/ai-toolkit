"""Utilities for loading pre-quantized models from Hugging Face or local paths.

Pre-quantized models are diffuser/transformer or text-encoder models that have
already been quantized and saved in a format compatible with *diffusers*
``from_pretrained``.  Loading them avoids the memory and time cost of running
on-the-fly quantization.

Supported *model_id* formats
-----------------------------
``"username/repo-name"``
    Plain Hugging Face repo ID.  The model config and weights are expected in
    the root of the repository.

``"username/repo-name/subfolder"``
    Hugging Face repo with an explicit subfolder (e.g. ``"transformer"``).

``"/local/path/to/model"``
    Absolute or relative local path pointing to a directory that contains the
    model ``config.json`` and weight files.

``"/local/path/to/model.safetensors"``
    Path to a specific ``.safetensors`` file.  The parent directory must
    contain a ``config.json``; that config is used and the named weight file
    is loaded via the ``weights_name`` argument to ``from_pretrained``.

Appending ``:filename.safetensors`` to any of the above forms lets you pick a
specific weight file when a directory or repo contains several quantization
variants::

    "/local/path/to/model:model-fp8.safetensors"
    "username/repo-name:model-q4.safetensors"
    "username/repo-name/subfolder:model-int8.safetensors"

The config.json must be co-located with the weight file (same directory or
same HF repo/subfolder).
"""

import os
from typing import Optional, Type, TypeVar

import torch

T = TypeVar("T", bound=torch.nn.Module)


def load_prequantized_model(model_id: str, model_class: Type[T], **kwargs) -> T:
    """Load a pre-quantized model and return an instance of *model_class*.

    Parameters
    ----------
    model_id:
        One of the formats described in the module-level docstring.
    model_class:
        The diffusers / transformers model class whose ``from_pretrained``
        method will be called (e.g. ``FluxTransformer2DModel``,
        ``T5EncoderModel``).
    **kwargs:
        Extra keyword arguments forwarded to ``model_class.from_pretrained``.
        ``torch_dtype`` defaults to ``"auto"`` so that the quantized weight
        dtypes saved in the checkpoint are preserved.

    Returns
    -------
    An instance of *model_class* with the pre-quantized weights loaded.

    Raises
    ------
    ValueError
        If a local path is given but does not exist or is otherwise unusable.
    """
    kwargs.setdefault("torch_dtype", "auto")

    # Normalise any Windows-style separators so the path-splitting logic below
    # works correctly on all platforms.  This must happen before the
    # os.path.exists() check so that paths like "C:\\some\\model" are converted
    # to "C:/some/model" before being tested.
    model_id = model_id.replace("\\", "/")

    # ------------------------------------------------------------------ #
    # Parse an optional specific weights filename appended with ':'       #
    # e.g.  "user/repo:model-fp8.safetensors"                            #
    #       "/local/dir:model-q4.safetensors"                             #
    # ------------------------------------------------------------------ #
    weights_name: Optional[str] = None
    if ":" in model_id:
        # Split on the *last* colon so that Windows drive letters (C:/) also
        # work after normalisation (C:/ → already normalised above, but be safe).
        loc, _, wname = model_id.rpartition(":")
        if wname.endswith((".safetensors", ".bin", ".gguf")):
            model_id = loc
            weights_name = wname
            kwargs.setdefault("weights_name", weights_name)

    # ------------------------------------------------------------------ #
    # Local path handling                                                  #
    # ------------------------------------------------------------------ #
    if os.path.exists(model_id):
        if os.path.isfile(model_id):
            # A direct path to a .safetensors file.  Use the parent directory
            # as the pretrained_model_name_or_path and pass the filename as
            # weights_name so from_pretrained picks the right weights but still
            # loads the config.json from the same folder.
            if not model_id.endswith((".safetensors", ".bin", ".gguf")):
                raise ValueError(
                    f"quantized_model_id / quantized_te_id points to a file "
                    f"that is not a .safetensors, .bin, or .gguf file: {model_id!r}."
                )
            parent = os.path.dirname(os.path.abspath(model_id))
            fname = os.path.basename(model_id)
            if not os.path.exists(os.path.join(parent, "config.json")):
                raise ValueError(
                    f"No config.json found in {parent!r}.  When pointing "
                    f"quantized_model_id / quantized_te_id at a specific "
                    f".safetensors file, the parent directory must contain "
                    f"config.json."
                )
            kwargs.setdefault("weights_name", fname)
            return model_class.from_pretrained(parent, **kwargs)
        # Directory – load directly (weights_name may already be in kwargs).
        return model_class.from_pretrained(model_id, **kwargs)

    # ------------------------------------------------------------------ #
    # Remote HF repo ID                                                    #
    # ------------------------------------------------------------------ #
    parts = model_id.split("/")
    if len(parts) >= 3:
        # "user/repo/subfolder[/deeper]"
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        return model_class.from_pretrained(repo_id, subfolder=subfolder, **kwargs)
    else:
        # "user/repo" – no subfolder
        return model_class.from_pretrained(model_id, **kwargs)
