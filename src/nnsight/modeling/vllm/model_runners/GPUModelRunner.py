import pickle
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from nnsight.intervention.tracing.globals import Globals

from ....intervention.serialization import load
from ..batching import VLLMBatcher

if TYPE_CHECKING:
    from ..vllm import VLLM
else:
    VLLM = Any

if TYPE_CHECKING:

    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput


class NNsightGPUModelRunner(GPUModelRunner):
    """Custom vLLM GPU model runner that interleaves NNsight interventions with model execution.

    Wraps the model with an NNsight :class:`Envoy`, deserializes
    mediators from incoming :class:`NNsightSamplingParams`, and manages
    batch group mappings so each invoke's intervention code sees the
    correct slice of the batch.
    """

    class NNsightRequestHelper:
        """
        Helper class for batching requests in the GPUModelRunner.

        Attributes:
            ids_to_batch_group (Dict[str, int]): Dictionary mapping request IDs to their assigned batch group indices.
            interleaver_to_ids (Dict[Interleaver, Set[str]]): Dictionary mapping interleavers to sets of request IDs.
            flat_batch_groups (Dict[Interleaver, List[Tuple[int, int]]]): Dictionary mapping interleavers to their flattened batch groups.

        Methods:
            process_new_reqs(new_reqs: List[NewRequestData]) -> None: Process new requests and compute the flat batch groups.
            process_finished_req(req_id: str, interleaver: Interleaver) -> None: Process a finished request,
                by updating batch groups and cleaning up mappings.
        """

        def __init__(self):

            self.req_id_to_batch_group_idx: Dict[str, int] = {}
            self.mediators: Dict[str, Any] = {}  # req_id -> Mediator
            self.trace_contexts: Dict[str, dict] = {}  # trace_id -> context

        def process_new_reqs(
            self, new_reqs: List["NewRequestData"], model: VLLM
        ) -> None:
            """
            Process new requests and organize them into batch groups for execution.

            Each request carries its own serialized mediator. When multiple
            mediators belong to the same trace (identified by trace_id), the
            first arrival's ``__globals__`` become the canonical reference.
            Subsequent arrivals graft the saved variable entries from the
            canonical globals into their own ``__globals__``, so all mediators
            share the same Python objects for cross-invoke state.

            Args:
                new_reqs (List[NewRequestData]): List of new request data objects to process.
            """

            for new_req in new_reqs:

                extra_args = getattr(new_req.sampling_params, 'extra_args', None)
                if not extra_args:
                    continue

                trace_id = extra_args.get("nnsight_trace_id")
                if trace_id is None:
                    # Non-NNsight request, skip
                    continue

                mediator = load(
                    extra_args["nnsight_mediator"],
                    model._remoteable_persistent_objects(),
                )

                saved_names = extra_args.get("nnsight_saved_names", [])

                # First mediator for this trace: create context and register
                # its __globals__ as canonical for shared variable grafting.
                if trace_id not in self.trace_contexts:
                    canonical_globals = mediator.intervention.__globals__

                    # Register saved vars in worker-side Globals.saves
                    # (.save() was called on the client with a different id).
                    for name in saved_names:
                        if name in canonical_globals:
                            Globals.saves.add(id(canonical_globals[name]))

                    self.trace_contexts[trace_id] = {
                        "saved_names": saved_names,
                        "canonical_globals": canonical_globals,
                        "expected_count": extra_args.get("nnsight_expected_count", 1),
                        "received_count": 0,
                        "pending_req_ids": set(),
                    }
                else:
                    # Subsequent mediator: graft saved vars from canonical
                    # globals so all mediators share the same Python objects.
                    ctx = self.trace_contexts[trace_id]
                    canonical = ctx["canonical_globals"]
                    med_globals = mediator.intervention.__globals__
                    for name in saved_names:
                        if name in canonical:
                            med_globals[name] = canonical[name]

                ctx = self.trace_contexts[trace_id]

                model._interleaver.mediators.append(mediator)
                mediator.start(model._interleaver)

                self.mediators[new_req.req_id] = mediator
                ctx["pending_req_ids"].add(new_req.req_id)
                ctx["received_count"] += 1

        def unflatten(self, model: VLLM):

            batch_start = 0

            for mediator in model._interleaver.mediators:

                mediator.batch_group = [batch_start, 1]

                batch_start += 1

                model._interleaver.batcher.last_batch_group = mediator.batch_group

        def process_batch_groups(
            self,
            num_tokens_scheduled: Dict[str, int],
            requests,
            model: VLLM,
        ) -> None:

            batch_start = 0

            mediators = []

            for req_id, num_tokens in num_tokens_scheduled.items():

                mediator = self.mediators.get(req_id)

                if mediator is None:
                    batch_start += num_tokens
                    continue

                mediators.append(mediator)
                mediator.batch_group = [batch_start, num_tokens]

                batch_start += num_tokens

            if mediators:
                model._interleaver.batcher.last_batch_group = mediators[-1].batch_group
            else:
                model._interleaver.batcher.last_batch_group = None

            model._interleaver.mediators = mediators

    def __init__(self, *args, **kwargs):

        from .. import VLLM

        super().__init__(*args, **kwargs)

        self.nnsight_model: VLLM

        self.nnsight_request_helper = self.NNsightRequestHelper()

    def load_model(self, *args, **kwargs) -> None:

        from .. import VLLM

        super().load_model(*args, **kwargs)

        self.nnsight_model = VLLM(self.model)

        self.nnsight_model.tokenizer = cached_tokenizer_from_config(self.model_config)

        self.nnsight_model._interleaver.mediators = []

        self.nnsight_model._interleaver.batcher = VLLMBatcher()

        self.nnsight_model._interleaver.batcher.wrap(self.nnsight_model)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:

        super()._update_states(scheduler_output)

        self.nnsight_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs, self.nnsight_model
        )

        self.nnsight_request_helper.process_batch_groups(
            scheduler_output.num_scheduled_tokens, self.requests, self.nnsight_model
        )

        self.nnsight_model._interleaver.batcher.needs_batching = (
            len(self.nnsight_model._interleaver.mediators) > 1
        )

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ):

        Globals.enter()
        with self.nnsight_model._interleaver:

            return_value = super().execute_model(scheduler_output, intermediate_tensors)

            self.nnsight_request_helper.unflatten(self.nnsight_model)

            if self.execute_model_state is not None:

                logits = self.nnsight_model.logits(
                    self.execute_model_state.logits, hook=True
                )

                state = self.execute_model_state

                self.execute_model_state = type(state)(
                    **{**state._asdict(), "logits": logits}
                )

        Globals.exit()

        return return_value

    def _sample(self, *args, **kwargs):

        Globals.enter()

        with self.nnsight_model._interleaver:

            sampler_output = super()._sample(*args, **kwargs)

            sampler_output.sampled_token_ids = self.model.samples(
                sampler_output.sampled_token_ids, hook=True
            )

        Globals.exit()

        return sampler_output

    def finish_nnsight(
        self, finished_req_ids: list[str]
    ) -> Optional[bytes]:
        result = None

        finished_req_id_set = set(finished_req_ids)

        if get_pp_group().rank == 0:

            # Match finished engine-level req_ids to our stored mediators.
            # Use the mediators dict (keyed by internal req_id like "0-abc123")
            # since self.requests may already be cleaned up in multiprocessing mode.
            matched = []
            matched_keys = []

            for req_id, mediator in self.nnsight_request_helper.mediators.items():
                # vLLM appends a hash suffix to request IDs:
                #   sync path: "0-abc123" -> engine reports "0"
                #   async path: "uuid-abc123" -> engine reports "uuid"
                # Use rsplit to strip only the suffix, preserving
                # hyphens in UUIDs. Fall back to split for legacy compat.
                base_id = req_id.rsplit("-", 1)[0]
                if base_id in finished_req_id_set:
                    matched.append((base_id, mediator))
                    matched_keys.append(req_id)
                elif req_id in finished_req_id_set:
                    matched.append((req_id, mediator))
                    matched_keys.append(req_id)

            Globals.enter()

            for internal_id, mediator in matched:

                if mediator.alive:

                    self.nnsight_model._interleaver.mediators = [mediator]
                    mediator.batch_group = None

                    with self.nnsight_model._interleaver:
                        self.nnsight_model._interleaver.handle("result", [internal_id])

                        mediator.cancel()

                        self.nnsight_model._interleaver.handle()

            Globals.exit()

            saves = {}

            removals = set()

            # Per-invoke local saves: collect from each mediator's frame
            for _, mediator in matched:
                frame = mediator.info.frame

                for key, value in frame.f_locals.items():

                    if id(value) in Globals.saves:
                        saves[key] = value
                        removals.add(id(value))

            # Trace-shared saves: decrement pending counts and collect
            # shared vars when ALL mediators for a trace have been received
            # AND completed. Must wait for all to avoid premature cleanup
            # (e.g., request 0 finishes before request 1 is even scheduled).
            for req_id in matched_keys:
                for _, ctx in self.nnsight_request_helper.trace_contexts.items():
                    if req_id in ctx["pending_req_ids"]:
                        ctx["pending_req_ids"].discard(req_id)
                        trace_fully_done = (
                            not ctx["pending_req_ids"]
                            and ctx["received_count"] == ctx["expected_count"]
                        )
                        if trace_fully_done:
                            # All mediators received and finished —
                            # collect shared saves from canonical globals
                            canonical = ctx["canonical_globals"]
                            for name in ctx["saved_names"]:
                                if name in canonical:
                                    value = canonical[name]
                                    if id(value) in Globals.saves:
                                        saves[name] = value
                                        removals.add(id(value))
                        break

            # Clean up fully completed trace contexts
            done_traces = [
                tid for tid, ctx in self.nnsight_request_helper.trace_contexts.items()
                if (not ctx["pending_req_ids"]
                    and ctx["received_count"] == ctx["expected_count"])
            ]
            for tid in done_traces:
                del self.nnsight_request_helper.trace_contexts[tid]

            for _id in removals:
                Globals.saves.discard(_id)

            # Pickle so it survives msgpack transport in multiprocessing mode
            result = pickle.dumps(saves)

            # Clean up mediator entries for finished requests
            for req_id in matched_keys:
                self.nnsight_request_helper.mediators.pop(req_id, None)

        return result
