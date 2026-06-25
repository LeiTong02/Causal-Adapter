# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared inference helpers for the Causal-Adapter (SD1.5) benchmark notebooks.

The three counterfactual notebooks (``counterfactuals_pendulum``,
``counterfactuals_celeA``, ``counterfactuals_ADNI``) duplicated ~70 lines of
boilerplate to load the Causal_ControlNetModel, attach the pretrained SCM head,
load MCPL pseudo-token embeddings into a CLIPTextModel and assemble the final
StableDiffusionCausalControlNetPipeline. This module centralises that setup so
each notebook only declares the dataset name, paths, and dataset-specific
intervention logic.

Per-dataset metadata kept here:

* ``A_MATRICES``    : ground-truth adjacency mask passed to
  ``controlnet.controlnet_cond_embedding.update_mask``.
* ``DATASET_PROMPTS``: default prompt / pseudo-word string used to learn the
  MCPL textual-inversion embeddings during training.
* ``_dataset_transform_prefix``: dataset-specific preprocessing applied before
  the shared 256x256 resize (e.g. CenterCrop for celeA, Pad(6) for ADNI).

These match the values used in ``train.py`` (see ``NUM_CAUSAL_CONCEPTS`` /
``_build_a_matrix``). Keep them in sync if a new dataset is added there.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torchvision import transforms
from transformers import CLIPTokenizer

from diffusers import (
    Causal_ControlNetModel,
    DDIMScheduler,
    StableDiffusionCausalControlNetPipeline,
)

from causal_modules.ddim_modules import load_mcpl_embeddings


# ---------------------------------------------------------------------------
# Per-dataset metadata
# ---------------------------------------------------------------------------

A_MATRICES = {
    "pendulum": [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    "celeA_complex": [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ],
    "ADNI": [
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
}

# (default_prompt, default_presudo_words) used at training time.
DATASET_PROMPTS = {
    "pendulum":      ("a image of @ and * and & and !", "@,*,&,!"),
    "celeA_complex": ("a human of @ and * and & and !",         "@,*,&,!"),
    "ADNI":          ("a mri image of @ and * and &",           "@,*,&"),
}


def _dataset_transform_prefix(dataset: str):
    """Dataset-specific torchvision transforms applied before the shared resize."""
    if dataset.startswith("celeA"):
        return [transforms.CenterCrop(150)]
    if dataset == "ADNI":
        return [transforms.Pad(padding=6)]
    return []


def _maybe_pad_prompt(dataset: str, prompt: str) -> str:
    """ADNI training pads the prompt with repetitions of its last character so
    the tokenizer always sees a fixed-length sequence of 10 trailing tokens.
    Inference must reproduce that exact prompt or the pseudo-token positions
    drift."""
    if dataset == "ADNI":
        return prompt + (" " + prompt[-1]) * (10 - 1)
    return prompt


def build_transforms(
    dataset: str,
    size: int = 256,
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Return (image, original, conditioning) transforms for ``dataset``.

    * ``image_transforms``        : feeds the VAE — Resize + ToTensor + Normalize.
    * ``original_transforms``     : human-viewable PIL preview (no normalize).
    * ``conditioning_image_transforms`` : skips the dataset-specific prefix,
      kept for callers that want a plain Resize+ToTensor variant.
    """
    prefix = _dataset_transform_prefix(dataset)
    resize = transforms.Resize(
        (size, size), interpolation=transforms.InterpolationMode.BILINEAR
    )

    image_transforms = transforms.Compose(
        prefix + [resize, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    original_transforms = transforms.Compose(prefix + [resize])
    conditioning_image_transforms = transforms.Compose([resize, transforms.ToTensor()])
    return image_transforms, original_transforms, conditioning_image_transforms


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------

@dataclass
class CausalAdapterAssets:
    """Bag of objects every benchmark notebook needs after pipeline setup."""

    pipe: StableDiffusionCausalControlNetPipeline
    controlnet: Causal_ControlNetModel
    tokenizer: CLIPTokenizer
    prompt: str
    presudo_words: str
    presudo_list: List[str]
    presudo_token_ids: List[int]
    device: torch.device


def load_causal_adapter(
    dataset: str,
    base_model_path: str,
    controlnet_path: str,
    text_embedding_path: str,
    scm_path: Optional[str] = None,
    *,
    prompt: Optional[str] = None,
    presudo_words: Optional[str] = None,
    device: Optional[torch.device] = None,
    torch_dtype: torch.dtype = torch.float32,
) -> CausalAdapterAssets:
    """Assemble a Causal-Adapter inference pipeline for one of the supported datasets.

    Replicates the per-notebook ``Load pipeline`` cell:

    1. Load ``Causal_ControlNetModel`` from ``controlnet_path``.
    2. Optionally overwrite the SCM head with the pretraining-stage weights at
       ``scm_path`` (path produced by ``SCM_modeling``).
    3. Apply the dataset's adjacency mask (``A_MATRICES[dataset]``).
    4. Load MCPL pseudo-token embeddings from ``text_embedding_path`` into a
       fresh CLIPTextModel.
    5. Build the StableDiffusionCausalControlNetPipeline with the explicit DDIM
       scheduler config required for inversion (``clip_sample=False``,
       ``set_alpha_to_one=False``).

    Notebook callers can override ``prompt`` / ``presudo_words`` if they trained
    with non-default tokens; otherwise the helper picks the values logged in
    ``DATASET_PROMPTS``.
    """
    if dataset not in A_MATRICES:
        raise ValueError(
            f"Unsupported dataset {dataset!r}; expected one of {list(A_MATRICES)}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. ControlNet + (optional) pretrained SCM head
    controlnet = Causal_ControlNetModel.from_pretrained(
        controlnet_path, torch_dtype=torch_dtype
    )
    if scm_path is not None:
        print("load pretrained causalnet weights")
        controlnet.controlnet_cond_embedding.load_state_dict(
            torch.load(scm_path, weights_only=True)
        )

    a_matrix = torch.tensor(A_MATRICES[dataset], dtype=torch_dtype, device=device)
    controlnet.controlnet_cond_embedding.update_mask(a_matrix)
    controlnet.eval()
    print("training_mode", controlnet.task_cond)

    # 2. Tokenizer + MCPL textual-inversion embeddings
    default_prompt, default_presudo = DATASET_PROMPTS[dataset]
    prompt = _maybe_pad_prompt(dataset, prompt if prompt is not None else default_prompt)
    presudo_words = presudo_words if presudo_words is not None else default_presudo
    presudo_list = presudo_words.split(",")

    tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    presudo_token_ids = tokenizer.encode(
        " ".join(presudo_list), add_special_tokens=False
    )
    text_encoder = load_mcpl_embeddings(
        base_model_path, tokenizer, text_embedding_path, presudo_token_ids
    )

    # 3. Pipeline + DDIM scheduler tuned for inversion
    pipe = StableDiffusionCausalControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe = pipe.to(device)

    return CausalAdapterAssets(
        pipe=pipe,
        controlnet=controlnet,
        tokenizer=tokenizer,
        prompt=prompt,
        presudo_words=presudo_words,
        presudo_list=presudo_list,
        presudo_token_ids=presudo_token_ids,
        device=device,
    )
