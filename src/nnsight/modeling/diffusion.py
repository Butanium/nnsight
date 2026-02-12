from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import pipelines
from diffusers import DiffusionPipeline
from transformers import PreTrainedTokenizerBase

from .. import util
from .huggingface import HuggingFaceModel
from typing import Type


class Diffuser(util.WrapperModule):
    """Wrapper module that loads a diffusion pipeline and exposes its components as submodules.

    All pipeline components that are ``torch.nn.Module`` or
    ``PreTrainedTokenizerBase`` instances are registered as attributes
    so they appear in the Envoy tree and can be traced. The exact
    component names depend on the pipeline (e.g. ``unet`` for Stable
    Diffusion, ``transformer`` for Flux, plus ``vae``, ``text_encoder``,
    etc.).

    Args:
        automodel (Type[DiffusionPipeline]): The diffusers pipeline
            class to use for loading. Defaults to ``DiffusionPipeline``.
        *args: Forwarded to ``automodel.from_pretrained()``.
        **kwargs: Forwarded to ``automodel.from_pretrained()``.

    Attributes:
        pipeline (DiffusionPipeline): The underlying diffusers pipeline.
    """

    def __init__(
        self, automodel: Type[DiffusionPipeline] = DiffusionPipeline, *args, **kwargs
    ) -> None:
        super().__init__()

        self.pipeline = automodel.from_pretrained(*args, **kwargs)

        for key, value in self.pipeline.__dict__.items():
            if isinstance(value, torch.nn.Module) or isinstance(
                value, PreTrainedTokenizerBase
            ):
                setattr(self, key, value)

    def generate(self, *args, **kwargs):
        """Run the full diffusion pipeline.

        Calls the pipeline's ``__call__`` method (not ``.generate()``,
        which does not exist on ``DiffusionPipeline``).

        Returns:
            The pipeline output (typically a dataclass with ``.images``).
        """
        return self.pipeline(*args, **kwargs)


class DiffusionModel(HuggingFaceModel):
    """NNsight wrapper for diffusion pipelines.

    Wraps any ``diffusers.DiffusionPipeline`` so that its components
    can be traced and intervened on. Works with UNet-based pipelines
    (Stable Diffusion) and transformer-based pipelines (Flux, DiT)
    alike — the denoiser is accessible as whatever attribute the
    pipeline exposes (``model.unet`` or ``model.transformer``).

    By default, ``.trace()`` runs the full diffusion pipeline with
    ``num_inference_steps=1`` for fast single-step tracing. Use
    ``.generate()`` to run the full pipeline with the default or
    user-specified number of inference steps.

    Examples::

        # Stable Diffusion (UNet-based)
        sd = DiffusionModel("stabilityai/stable-diffusion-2-1")
        with sd.generate("A cat", num_inference_steps=50) as tracer:
            for step in tracer.iter[:]:
                denoiser_out = sd.unet.output.save()

        # Flux (Transformer-based)
        flux = DiffusionModel("black-forest-labs/FLUX.1-schnell")
        with flux.trace("A cat"):
            denoiser_out = flux.transformer.output.save()

    Args:
        *args: Forwarded to :class:`HuggingFaceModel`.  The first
            positional argument is typically a repo ID string.
        automodel (Type[DiffusionPipeline]): The diffusers pipeline
            class (or a string name resolvable from ``diffusers.pipelines``).
            Defaults to ``DiffusionPipeline``.
        **kwargs: Forwarded to the pipeline's ``from_pretrained()``.

    Attributes:
        automodel (Type[DiffusionPipeline]): The pipeline class used for loading.
    """

    def __init__(
        self, *args, automodel: Type[DiffusionPipeline] = DiffusionPipeline, **kwargs
    ) -> None:

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(pipelines, automodel)
        )

        self._model: Diffuser = None

        super().__init__(*args, **kwargs)

    def _load_meta(self, repo_id: str, revision: Optional[str] = None, **kwargs):
        """Load a meta (placeholder) version of the diffusion model.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Git revision of the repository.
            **kwargs: Forwarded to ``Diffuser()``.

        Returns:
            A :class:`Diffuser` instance.
        """

        model = Diffuser(
            self.automodel,
            repo_id,
            revision=revision,
            device_map=None,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        return model

    def _load(
        self, repo_id: str, revision: Optional[str] = None, device_map=None, **kwargs
    ) -> Diffuser:
        """Load the diffusion model with full weights.

        Args:
            repo_id: HuggingFace repository ID.
            revision: Git revision of the repository.
            device_map: Device placement strategy.
            **kwargs: Forwarded to ``Diffuser()``.

        Returns:
            A :class:`Diffuser` instance.
        """

        model = Diffuser(
            self.automodel, repo_id, revision=revision, device_map=device_map, **kwargs
        )

        return model

    def _prepare_input(self, *inputs, **kwargs):
        """Normalize raw user input into a consistent format for batching.

        Accepts a single string prompt or a list of string prompts.
        Returns ``(args, kwargs, batch_size)`` where args is a tuple
        containing the prompt list.

        Args:
            *inputs: A single string or list of strings.
            **kwargs: Additional keyword arguments (passed through).

        Returns:
            Tuple of ``((prompts,), kwargs, batch_size)``.
        """
        if len(inputs) == 0:
            return tuple(), kwargs, 0

        assert len(inputs) == 1
        prompt = inputs[0]

        if isinstance(prompt, str):
            prompt = [prompt]

        return (prompt,), kwargs, len(prompt)

    def _batch(self, batched_input, *args, **kwargs):
        """Combine a new invoke's prepared prompts with already-batched prompts.

        Merges prompt lists from multiple invokes into a single list
        for batched pipeline execution.

        Args:
            batched_input: A tuple of ``(batched_args, batched_kwargs)``
                from all previous invokes.
            *args: The new invoke's prepared positional arguments.
            **kwargs: The new invoke's prepared keyword arguments.

        Returns:
            Tuple of ``(combined_args, combined_kwargs)``.
        """
        batched_args, batched_kwargs = batched_input

        if len(args) > 0:
            combined_prompts = list(batched_args[0]) + list(args[0])
        else:
            combined_prompts = list(batched_args[0])

        combined_kwargs = {**batched_kwargs, **kwargs}

        return (combined_prompts,), combined_kwargs

    def _run_pipeline(self, prepared_inputs, *args, seed=None, **kwargs):
        """Shared pipeline execution logic for both trace and generate.

        Sets up iteration step counting on the interleaver, handles
        seed/generator creation, runs the pipeline, wraps the output
        through the model's forward (for hook access), and resets
        ``default_all`` afterward.

        Args:
            prepared_inputs: The prompt list from ``_prepare_input``.
            *args: Additional positional arguments for the pipeline.
            seed: Random seed for reproducibility. If provided with
                multiple prompts, each prompt gets ``seed + offset``.
            **kwargs: Keyword arguments forwarded to the pipeline
                (e.g. ``num_inference_steps``, ``guidance_scale``).

        Returns:
            The pipeline output passed through the wrapper module.
        """
        if self._interleaver is not None:
            steps = kwargs.get("num_inference_steps")
            if steps is None:
                try:
                    steps = (
                        inspect.signature(self._model.pipeline.__call__)
                        .parameters["num_inference_steps"]
                        .default
                    )
                except Exception:
                    steps = 50
            self._interleaver.default_all = steps

        generator = torch.Generator(self.device)

        if seed is not None:
            if isinstance(prepared_inputs, list) and len(prepared_inputs) > 1:
                generator = [
                    torch.Generator(self.device).manual_seed(seed + offset)
                    for offset in range(
                        len(prepared_inputs) * kwargs.get("num_images_per_prompt", 1)
                    )
                ]
            else:
                generator = generator.manual_seed(seed)

        output = self._model.pipeline(
            prepared_inputs, *args, generator=generator, **kwargs
        )

        if self._interleaver is not None:
            self._interleaver.default_all = None

        output = self._model(output)

        return output

    def __call__(self, prepared_inputs, *args, **kwargs):
        """Run the full diffusion pipeline with a 1-step default.

        Used by ``.trace()`` — defaults to ``num_inference_steps=1``
        for fast single-step tracing unless the user overrides it.

        Args:
            prepared_inputs: The prompt list from ``_prepare_input``.
            *args: Additional positional arguments for the pipeline.
            **kwargs: Keyword arguments forwarded to the pipeline.

        Returns:
            The pipeline output passed through the wrapper module.
        """
        kwargs.setdefault("num_inference_steps", 1)
        return self._run_pipeline(prepared_inputs, *args, **kwargs)

    def __nnsight_generate__(self, prepared_inputs, *args, **kwargs):
        """Run the full diffusion pipeline for ``.generate()`` contexts.

        Unlike ``__call__``, this does not set a default for
        ``num_inference_steps``, allowing the pipeline's own default
        (or the user's explicit value) to take effect.

        Args:
            prepared_inputs: The prompt list from ``_prepare_input``.
            *args: Additional positional arguments for the pipeline.
            **kwargs: Keyword arguments forwarded to the pipeline.

        Returns:
            The pipeline output passed through the wrapper module.
        """
        return self._run_pipeline(prepared_inputs, *args, **kwargs)
