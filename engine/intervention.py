from __future__ import annotations

from typing import Any, Union

import torch.futures

from . import util
from .fx.Graph import Graph
from .fx.Node import Node
from .fx.Proxy import Proxy


class TokenIndexer:
    def __init__(self, proxy: InterventionProxy) -> None:
        self.proxy = proxy

    def convert_idx(self, idx: int):
        if idx >= 0:
            n_tokens = self.proxy.node.proxy_value.shape[1]
            idx = -(n_tokens - idx)

        return idx

    def __getitem__(self, key: int) -> Proxy:

        key = self.convert_idx(key)

        return self.proxy[:, key]

    def __setitem__(self, key: int, value: Union[Proxy, Any]) -> None:
        key = self.convert_idx(key)

        self.proxy[:, key] = value


class InterventionProxy(Proxy):
    @staticmethod
    def proxy_save(value: Any) -> None:
        return util.apply(value, lambda x: x.clone(), torch.Tensor)

    def save(self) -> InterventionProxy:
        proxy = self.node.graph.add(
            graph=self.node.graph,
            value=self.node.proxy_value,
            target=InterventionProxy.proxy_save,
            args=[self.node],
        )

        self.node.graph.add(
            graph=self.node.graph,
            value=None,
            target="null",
            args=[proxy.node],
        )

        return proxy

    @property
    def token(self) -> TokenIndexer:
        return TokenIndexer(self)

    @property
    def t(self) -> TokenIndexer:
        return self.token

    @property
    def shape(self):
        return util.apply(self.node.proxy_value, lambda x: x.shape, torch.Tensor)

    @property
    def value(self):
        return self.node.future.value()


def intervene(activations, module_path: str, graph: Graph, key: str):
    """Entry to intervention graph. This should be hooked to all modules involved in intervention graph.

    Args:
        activations (_type_): _description_
        module_path (str): _description_
        graph (Graph): _description_
        key (str): _description_

    Returns:
        _type_: _description_
    """

    # Key to module activation argument nodes has format: <module path>.<output/input>.<generation index>.<batch index>
    module_path = f"{module_path}.{key}.{graph.generation_idx}"

    # TODO
    # Probably need a better way to do this. Should be a dict of argument name to list of nodes and their batch_idx?
    argument_node_names = [
        name for name in graph.argument_node_names if name.startswith(module_path)
    ]

    for argument_node_name in argument_node_names:
        node = graph.nodes[graph.argument_node_names[argument_node_name]]

        batch_idx = int(argument_node_name.split(".")[-1])

        # We set its result to the activations, indexed by only the relevant batch index.
        node.future.set_result(
            util.apply(
                activations, lambda x: x.select(0, batch_idx).unsqueeze(0), torch.Tensor
            )
        )

    return activations