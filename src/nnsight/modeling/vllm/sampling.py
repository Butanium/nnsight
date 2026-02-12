import copy
from typing import Optional
from vllm.sampling_params import SamplingParams
from ...intervention.interleaver import Mediator
from ...intervention.serialization import save
from msgspec import structs
from typing import List


def rebuild(state):
    return NNsightSamplingParams(**state)


class NNsightSamplingParams(SamplingParams):
    """Extended vLLM ``SamplingParams`` that carries a serialized :class:`Mediator`.

    When sent to a vLLM worker, the mediator is serialized via
    :func:`save` so the worker can deserialize it and run the
    user's intervention code alongside model execution.

    Attributes:
        mediator (Optional[Mediator | bytes]): The mediator instance
            (or its serialized bytes) for this request.
    """

    mediator: Optional[Mediator | bytes] = None

    def __reduce__(self):

        state = structs.asdict(self)

        state["mediator"] = self.mediator

        if isinstance(self.mediator, Mediator):
            
            self.mediator.intervention.__source__ = "".join(self.mediator.info.source)

            state["mediator"] = save(self.mediator)

        return (rebuild, (state,))

    def clone(self):

        memo = (
            {}
            if self.logits_processors is None
            else {
                id(lp): lp.clone() if hasattr(lp, "clone") else lp
                for lp in self.logits_processors
            }
        )

        memo[id(self.mediator)] = self.mediator

        return copy.deepcopy(self, memo=memo)
