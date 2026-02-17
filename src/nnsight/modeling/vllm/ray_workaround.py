"""Custom Ray executor for NNsight that works around a vLLM + Ray actor crash.

vLLM v0.15.1 + Ray 2.53.0 have a compatibility issue where Ray actor processes
crash during module-level imports of heavy vllm submodules (particularly
``vllm.multimodal``) during the actor construction phase. The crash occurs at
the C level (in grpcio's ``cygrpc`` extension) with no Python traceback.

The root cause: when Ray spawns an actor process and imports the module
containing ``RayWorkerWrapper``, the transitive module-level imports
(``worker_base.py`` → ``vllm.multimodal``, etc.) conflict with Ray's internal
gRPC event engine, causing the process to die before the actor is fully
constructed.

The fix: ``LazyRayWorkerWrapper`` is a thin wrapper class with no heavy
module-level imports. It defers all vllm imports to ``__init__`` time, which
runs after the actor process is fully constructed and Ray's gRPC connection is
stable. All methods are explicitly defined to satisfy Ray's remote method
resolution (``__getattr__`` delegation does not work with Ray actor handles).

``NNsightRayExecutor`` is a subclass of ``RayDistributedExecutor`` that swaps
in ``LazyRayWorkerWrapper`` before creating workers. Pass it as the
``distributed_executor_backend`` to ``vllm.LLM()`` instead of ``"ray"``.
"""

from vllm.v1.executor.ray_executor import RayDistributedExecutor


class LazyRayWorkerWrapper:
    """Drop-in replacement for ``vllm.v1.executor.ray_utils.RayWorkerWrapper``.

    Defers heavy vllm imports to ``__init__`` (actor method execution time)
    rather than module import time (actor construction time).
    """

    def __init__(self, *args, **kwargs):
        from vllm.v1.executor.ray_utils import RayWorkerWrapper

        self._w = RayWorkerWrapper(*args, **kwargs)

    # --- WorkerWrapperBase methods ---

    def update_environment_variables(self, envs_list):
        return self._w.update_environment_variables(envs_list)

    def init_worker(self, all_kwargs):
        return self._w.init_worker(all_kwargs)

    def adjust_rank(self, rank_mapping):
        return self._w.adjust_rank(rank_mapping)

    def execute_method(self, method, *args, **kwargs):
        return self._w.execute_method(method, *args, **kwargs)

    def shutdown(self):
        return self._w.shutdown()

    # --- RayWorkerWrapper methods ---

    def get_node_ip(self):
        return self._w.get_node_ip()

    def get_node_and_gpu_ids(self):
        return self._w.get_node_and_gpu_ids()

    def setup_device_if_necessary(self):
        return self._w.setup_device_if_necessary()

    def execute_model_ray(self, execute_model_input):
        return self._w.execute_model_ray(execute_model_input)


class NNsightRayExecutor(RayDistributedExecutor):
    """Ray executor that uses ``LazyRayWorkerWrapper`` to avoid actor crashes.

    Pass this class as ``distributed_executor_backend`` instead of ``"ray"``::

        LLM("gpt2", distributed_executor_backend=NNsightRayExecutor)

    This works regardless of multiprocessing mode because vLLM pickles the
    executor class to the EngineCore subprocess, where ``_init_executor``
    runs and swaps in the lazy wrapper before any Ray actors are created.
    """

    def _init_executor(self) -> None:
        import vllm.v1.executor.ray_utils as ray_utils
        import vllm.v1.executor.ray_executor as ray_exec

        ray_utils.RayWorkerWrapper = LazyRayWorkerWrapper
        ray_exec.RayWorkerWrapper = LazyRayWorkerWrapper
        super()._init_executor()
