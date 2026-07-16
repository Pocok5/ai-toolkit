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
"""

import os
from typing import Type, TypeVar

import torch

T = TypeVar("T", bound=torch.nn.Module)


def load_prequantized_model(model_id: str, model_class: Type[T], **kwargs) -> T:
    """Load a pre-quantized model and return an instance of *model_class*.

    Parameters
    ----------
    model_id:
        HF repo ID, ``"repo/id/optional-subfolder"``, or a local directory
        path.  See module-level docstring for details.
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
        If a local path is given but does not exist.
    """
    kwargs.setdefault("torch_dtype", "auto")

    # Local path – validate and load directly.
    if os.path.exists(model_id):
        if not os.path.isdir(model_id):
            raise ValueError(
                f"quantized_model_id / quantized_te_id points to a file, not a "
                f"directory: {model_id!r}.  Please provide a directory that "
                f"contains the model config.json and weight files."
            )
        return model_class.from_pretrained(model_id, **kwargs)

    # Remote HF repo ID.  Split into repo_id + optional subfolder.
    norm = model_id.replace("\\", "/")
    parts = norm.split("/")
    if len(parts) >= 3:
        # "user/repo/subfolder[/deeper]"
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        return model_class.from_pretrained(repo_id, subfolder=subfolder, **kwargs)
    else:
        # "user/repo" – no subfolder
        return model_class.from_pretrained(model_id, **kwargs)
