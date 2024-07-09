from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import util
from ..patching import Patch, Patcher
from ..tracing import protocols
from ..tracing.Node import Node
from ..tracing.Proxy import Proxy

if TYPE_CHECKING:

    from .Tracer import Tracer


class Invoker(AbstractContextManager):
    """An Invoker is meant to work in tandem with a :class:`nnsight.contexts.Tracer.Tracer` to enter input and manage intervention tracing.

    Attributes:
        tracer (nnsight.contexts.Tracer.Tracer): Tracer object to enter input and manage context.
        inputs (tuple[Any]): Initially entered inputs, then post-processed inputs from model's ._prepare_inputs(...) method.
        scan (bool): If to execute the model using `FakeTensor` in order to update the potential sizes/dtypes of all modules' Envoys' inputs/outputs as well as validate things work correctly.
            Scanning is not free computation wise so you may want to turn this to false when running in a loop.
            When making interventions, you made get shape errors if scan is false as it validates operations based on shapes so
            for looped calls where shapes are consistent, you may want to have scan=True for the first loop. Defaults to True.
        kwargs (Dict[str,Any]): Keyword arguments passed to the model's _prepare_inputs method.
        scanning (bool): If currently scanning.
    """

    def __init__(
        self,
        tracer: "Tracer",
        *inputs: Any,
        scan: bool = True,
        **kwargs,
    ) -> None:

        self.tracer = tracer
        self.inputs = inputs
        self.scan = scan
        self.kwargs = kwargs

        self.scanning = False

    def __enter__(self) -> Invoker:
        """Enters a new invocation context with a given input.

        Calls the model's _prepare_inputs method using the input and other arguments.
        If scan is True, uses the model's ._execute method to update and validate module Envoy's inputs/outputs using a fake mode.
        Gets a batched version of the post processed input using the model's ._batched_inputs method to update the Tracer's
            current batch_size and batched_input.

        Returns:
            Invoker: Invoker.
        """

        self.tracer.invoker = self

        preserved_inputs = None

        # If were accumulating, we might have Proxies in the input.
        # Therefore we first: Check to see if there are any Proxies.
        # If there are, preserve the raw inputs with Proxies converted to a Locked Bridge protocol.
        # Set self.inputs to be the proxy_value so we can prepare_inputs, get the batch size, and scan.
        if self.tracer.model._session is not None:

            def check_for_nodes(proxy: Proxy):

                if not proxy.node.done():

                    nonlocal preserved_inputs

                    preserved_inputs = self.inputs

                    node = proxy.node

                    return protocols.LockProtocol.add(
                        protocols.BridgeProtocol.add(node, self.tracer.graph).node
                    ).node

            inputs = util.apply(self.inputs, check_for_nodes, Proxy)

            if preserved_inputs is not None:

                preserved_inputs = inputs

                self.inputs = util.apply(
                    self.inputs, lambda x: x.node.proxy_value, Proxy
                )

        self.inputs, batch_size = self.tracer.model._prepare_inputs(
            *self.inputs, **self.kwargs
        )

        if self.scan:
            self.tracer.model._envoy._clear()

            self.scanning = True

            with Patcher() as patcher:

                # Some logic (like gpt-j rotary embeddings) gets "poisoned" by FakeTensors.
                # This does not happen when `torch.jit.is_tracing() returns True.`
                patcher.add(Patch(torch.jit, lambda: True, "is_tracing"))

                with FakeTensorMode(
                    allow_non_fake_inputs=True,
                    shape_env=ShapeEnv(assume_static_by_default=True),
                ) as fake_mode:
                    with FakeCopyMode(fake_mode):
                        self.tracer.model._execute(
                            *copy.deepcopy(self.inputs),
                            **copy.deepcopy(self.tracer._kwargs),
                        )

            self.scanning = False

        else:
            self.tracer.model._envoy._reset()

        self.tracer._batch_start += self.tracer._batch_size
        self.tracer._batch_size = batch_size

        # If there were no Proxies in the input, batch together the input.
        if preserved_inputs is None:

            self.tracer._batched_input = self.tracer.model._batch_inputs(
                self.tracer._batched_input,
                *self.inputs,
            )

        # Otherwise we don't know how to batch the Proxies so just assume we can add each input to a list?
        # TODO: revisit this.
        else:

            if self.tracer._batched_input is None:

                self.tracer._batched_input = [*preserved_inputs]
            else:

                self.tracer._batched_input.extend(preserved_inputs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self.tracer.invoker = None

        if isinstance(exc_val, BaseException):
            raise exc_val
