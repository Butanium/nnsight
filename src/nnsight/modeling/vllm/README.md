# vLLM Integration

This document details the design and implementation of NNsight's vLLM integration. It is written for contributors working on this code and assumes familiarity with NNsight's core concepts (tracing, interleaving, mediators, envoys) but is otherwise self-contained.

---

## Table of Contents

1. [Overview](#overview)
2. [File Structure](#file-structure)
3. [Key Classes](#key-classes)
4. [Execution Flow](#execution-flow)
5. [Model Loading](#model-loading)
6. [Mediator Transport via SamplingParams](#mediator-transport-via-samplingparams)
7. [Batch Group Management](#batch-group-management)
8. [Multiple Interleaving Phases](#multiple-interleaving-phases)
9. [Tensor Parallelism](#tensor-parallelism)
10. [Continuous Batching](#continuous-batching)
11. [Multi-Token Generation](#multi-token-generation)

---

## Overview

The vLLM integration enables NNsight interventions (observing and modifying intermediate activations) on models served through vLLM's high-performance inference engine. This is one of the most complex integrations in NNsight because vLLM's architecture differs substantially from standard PyTorch model execution:

- **Separate processes**: vLLM runs model execution in worker processes, not the user's process. Intervention code must be serialized and transported across process boundaries.
- **Flat tensor format**: vLLM concatenates all tokens from all prompts into a single `[total_tokens, hidden]` tensor rather than the standard `[batch, tokens, hidden]` format.
- **Continuous batching**: Requests can join and leave the batch between generation steps.
- **Tensor parallelism**: When using multiple GPUs, tensors are sharded. Intervention code must see complete, unsharded tensors.
- **Phased execution**: The forward pass, logit computation, and sampling are separate stages that NNsight must hook into independently.

The integration solves each of these by subclassing vLLM's engine, worker, and model runner classes, injecting NNsight's interleaving machinery at key points.

---

## File Structure

```
vllm/
├── __init__.py                    # Exports VLLM class
├── vllm.py                        # VLLM model wrapper (user-facing class)
├── sampling.py                    # NNsightSamplingParams — carries serialized mediators
├── batching.py                    # VLLMBatcher — tensor-parallel gather/split + flat-batch slicing
├── engines/
│   ├── __init__.py
│   └── engine.py                  # NNsightLLMEngine — collects saved results after requests finish
├── workers/
│   ├── __init__.py
│   └── GPUWorker.py               # NNsightGPUWorker — monkey-patches model runner at init
└── model_runners/
    ├── __init__.py
    └── GPUModelRunner.py           # NNsightGPUModelRunner — core: interleaves interventions with vLLM execution
```

### File Responsibilities

**`vllm.py`** — The `VLLM` class that users instantiate. Handles:
- Meta/real model loading via mixin inheritance (`RemoteableMixin → MetaMixin → LoadableMixin → NNsight`)
- Input preparation (`_prepare_input`) — normalizes strings, token ID lists, and HuggingFace tokenizer dicts
- Batching multiple invokes together (`_batch`)
- Forwarding calls to the vLLM engine (`__call__`)
- Creating wrapper modules for `logits`, `samples`, and `generator`

**`sampling.py`** — `NNsightSamplingParams` extends vLLM's `SamplingParams` with a `mediator` field. Implements `__reduce__()` so mediators survive pickling across process boundaries.

**`batching.py`** — `VLLMBatcher` extends NNsight's base `Batcher` to handle tensor parallelism. Registers pre/post hooks on all modules to track which module is currently executing and whether its tensors are sharded. When intervention code requests a value, the batcher transparently gathers sharded tensors; when intervention code returns a modified value, the batcher re-shards before passing back to vLLM.

**`engines/engine.py`** — `NNsightLLMEngine` extends vLLM's `LLMEngine`. After each engine step, checks for finished requests and calls `finish_nnsight()` on the model executor to collect saved intervention results.

**`workers/GPUWorker.py`** — `NNsightGPUWorker` extends vLLM's `Worker`. Its only job is to monkey-patch `GPUModelRunner` with `NNsightGPUModelRunner` before vLLM's init runs, and to expose `finish_nnsight()`.

**`model_runners/GPUModelRunner.py`** — `NNsightGPUModelRunner` is the core of the integration. It:
- Creates a second `VLLM` wrapper around the model loaded by vLLM (inside the worker process)
- Deserializes mediators from incoming requests
- Manages batch group mappings (flat token-level during forward, prompt-level after)
- Enters the interleaver at three phases: forward pass, logit wrapping, and sampling
- Collects saved values when requests finish

---

## Key Classes

### VLLM (vllm.py)

The user-facing class. Exists in two contexts:

1. **User process**: Created by the user (`model = VLLM("gpt2", dispatch=True)`). Handles tracing, input preparation, and dispatching to the vLLM engine.
2. **Worker process**: Created by `NNsightGPUModelRunner.load_model()` to wrap the model that vLLM loaded. This instance has the interleaver and batcher attached.

Key attributes:
- `vllm_entrypoint` — The actual `vllm.LLM` instance (user process only)
- `tokenizer` — vLLM's tokenizer
- `logits` — `WrapperModule` envoy for intercepting logits
- `samples` — `WrapperModule` envoy for intercepting sampled tokens
- `generator` — `WrapperModule` envoy for generation output

### NNsightSamplingParams (sampling.py)

Extends `vllm.SamplingParams` with a `mediator` field. The mediator is serialized to bytes via `save()` for transport and deserialized via `load()` in the worker. Implements `clone()` to deep-copy while preserving mediator references (important because vLLM clones params internally).

### VLLMBatcher (batching.py)

Extends NNsight's `Batcher`. Handles two concerns:

1. **Batch slicing**: `narrow(batch_group)` extracts a mediator's slice from the flat batch; `swap(batch_group, value)` puts a modified value back. When `batch_group` is `None` (empty invoke), the full batch is returned/replaced.

2. **Tensor parallelism**: Tracks the current module and whether its tensors are sharded. `check_gathered()` gathers sharded tensors before intervention code sees them. Post-hooks re-shard after intervention.

### NNsightGPUModelRunner (model_runners/GPUModelRunner.py)

The most complex class. Contains an inner `NNsightRequestHelper` that manages:
- Deserializing mediators from new requests
- Mapping request IDs to batch groups (token-level start position and count)
- Switching batch groups from flat (token-level) to unflattened (prompt-level) after the forward pass

Key methods:
- `load_model()` — Creates the worker-side `VLLM` wrapper and `VLLMBatcher`
- `_update_states(scheduler_output)` — Processes new/finished requests, updates batch groups
- `execute_model(scheduler_output, ...)` — Runs the forward pass inside an interleaver context, wraps logits
- `_sample()` — Runs sampling inside an interleaver context, wraps sampled tokens
- `finish_nnsight(finished_requests)` — Collects saved values from finished mediators

### NNsightLLMEngine (engines/engine.py)

Thin extension of vLLM's engine. After each `step()`, checks for finished requests and delegates to `finish_nnsight()` on the executor to gather saved results.

### NNsightGPUWorker (workers/GPUWorker.py)

Thin extension of vLLM's worker. Monkey-patches the model runner class before init, and exposes `finish_nnsight()` which delegates to the model runner.

---

## Execution Flow

### End-to-End: From User Trace to Saved Values

**1. User enters trace context:**
```python
with model.trace("Hello", temperature=0.0, max_tokens=3) as tracer:
    logits = model.logits.output.save()
```

NNsight captures, parses, and compiles the intervention code into a `Mediator`.

**2. `VLLM.__call__()` is invoked:**
- `_prepare_input()` normalizes the input (tokenizes strings, etc.)
- `_batch()` combines inputs from all invokes
- Each invoke's mediator is attached to an `NNsightSamplingParams` instance
- The mediator is serialized to bytes inside the params
- `vllm_entrypoint.generate(prompts, sampling_params)` is called

**3. vLLM schedules the request:**
- The engine passes the request through its scheduler
- The worker's `_update_states()` is called with the scheduler output

**4. `NNsightGPUModelRunner._update_states()`:**
- Calls `process_new_reqs()` — deserializes mediators from new requests' `SamplingParams`
- Calls `process_batch_groups()` — computes each mediator's `[start_token, num_tokens]` batch group based on scheduled token counts
- Registers mediators with the interleaver

**5. `NNsightGPUModelRunner.execute_model()`:**
- Enters `Globals` context (NNsight thread-local state)
- Enters interleaver context (`with self.nnsight_model._interleaver:`)
  - This starts mediator worker threads
- Calls `super().execute_model()` — vLLM's forward pass runs, module hooks fire, mediators interleave
- After forward pass: calls `unflatten()` to switch batch groups from token-level to prompt-level
- Wraps logits through `model.logits(logits, hook=True)` — mediators can observe/modify logits
- Updates `execute_model_state` with the (potentially modified) logits

**6. `NNsightGPUModelRunner._sample()`:**
- Enters `Globals` context and interleaver context
- Calls `super()._sample()` — vLLM samples next tokens
- Wraps sampled token IDs through `model.samples(token_ids, hook=True)` — mediators can observe/modify samples

**7. Steps 4-6 repeat for each generation step** (if `max_tokens > 1`).

**8. When all requests in an invoke group finish:**
- `NNsightLLMEngine.step()` detects finished requests
- Calls `finish_nnsight(finished_requests)` on the executor → worker → model runner
- Model runner enters interleaver, calls `interleaver.handle("result", outputs)` — mediators can interact with final output
- Extracts saved values from mediator frames (any variable marked with `.save()`)
- Returns saves dict, which gets attached to the `RequestOutput`

**9. Back in user process:**
- `VLLM.__call__()` receives the `RequestOutput` with attached saves
- Saved values are pushed back into the user's local variables

---

## Model Loading

The `VLLM` class uses `MetaMixin` for lazy/eager loading.

### Meta Loading (`_load_meta`)

When `dispatch=False` (default), the model is loaded with meta tensors (no real weights allocated). This uses vLLM's `DummyModelLoader` with `device="meta"`. The purpose is to build the Envoy tree (module hierarchy) so users can write intervention code referencing `model.transformer.h[0].output` etc. without allocating GPU memory.

### Real Loading (`_load`)

When `dispatch=True` or when `interleave()` auto-dispatches:
- Destroys any existing distributed environment
- Creates a `vllm.LLM` instance with `enforce_eager=True`
- Sets the worker class to `NNsightGPUWorker` via `worker_cls` kwarg
- After creation, monkey-patches the engine class to `NNsightLLMEngine`

### Worker-Side Loading

Inside the worker process, `NNsightGPUModelRunner.load_model()`:
- Calls vLLM's normal `load_model()` (loads real weights)
- Creates a new `VLLM` wrapper around the loaded model
- Creates a `VLLMBatcher` and attaches it to the interleaver
- Calls `batcher.wrap(model)` to register tensor-parallelism hooks on all modules

This means there are **two VLLM instances**: one in the user process (for tracing/input prep) and one in the worker process (for interleaving).

---

## Mediator Transport via SamplingParams

The core challenge: intervention code is compiled into a `Mediator` in the user process, but must execute in the worker process.

### How It Works

1. During tracing, each invoke produces a `Mediator` containing the compiled intervention function.
2. `VLLM.__call__()` creates `NNsightSamplingParams` with the mediator attached.
3. `NNsightSamplingParams.__reduce__()` serializes the mediator to bytes using `save()` (which uses `pickle` + `dill` for closures).
4. vLLM's internal pipeline pickles the `SamplingParams` when passing to worker processes.
5. In the worker, `process_new_reqs()` checks if the mediator is bytes and deserializes it using `load()`, passing the worker-side model as context.

### Why SamplingParams?

vLLM already passes `SamplingParams` through its entire pipeline — from engine to scheduler to worker to model runner. Attaching mediators here avoids creating a separate transport mechanism. Each prompt in the batch has its own `SamplingParams`, so each can carry a different mediator (or the same one for prompts in the same invoke).

### Cloning

vLLM clones `SamplingParams` internally. `NNsightSamplingParams.clone()` ensures the mediator reference is preserved (not deep-copied) so all prompts in the same invoke share the same mediator instance.

---

## Batch Group Management

### The Problem: Flat Tensor Format

Standard NNsight uses `[batch, tokens, hidden]` tensors. vLLM concatenates all tokens into a flat `[total_tokens, hidden]` tensor for efficiency.

```
Standard NNsight:
  Prompt "Hello World" (5 tokens):  [1, 5, 768]  → batch_group = [0, 1]
  Prompt "Hi" (2 tokens):           [1, 2, 768]  → batch_group = [1, 1]

vLLM (flat):
  All tokens concatenated:          [7, 768]      → batch_group = [0, 5] for "Hello World"
                                                     batch_group = [5, 2] for "Hi"
```

### Token-Level vs Prompt-Level Batch Groups

During the forward pass, batch groups are **token-level**: `[start_token_index, num_tokens]`. This allows `narrow()` to slice the correct tokens for each invoke's intervention code.

After the forward pass (for logits and sampling), batch groups switch to **prompt-level**: `[start_prompt_index, num_prompts]`. This is because logits and sampled tokens are per-prompt, not per-token.

`NNsightRequestHelper` manages this transition:
- `process_batch_groups()` computes token-level batch groups from the scheduler's `num_scheduled_tokens`
- `unflatten()` switches to prompt-level batch groups after the forward pass

### Batch Group Updates Per Step

Because vLLM uses continuous batching, batch groups are recomputed every generation step via `process_batch_groups()`. The scheduler may schedule different numbers of tokens per request at each step (e.g., full prompt on prefill, single token on decode).

---

## Multiple Interleaving Phases

vLLM separates execution into distinct stages. NNsight enters the interleaver at each:

### Phase 1: Forward Pass (`execute_model`)

The interleaver context wraps `super().execute_model()`. Module hooks fire as the model runs, and mediator threads interleave to observe/modify intermediate activations. Batch groups are token-level during this phase.

After the forward pass completes, `unflatten()` switches batch groups to prompt-level.

### Phase 2: Logits

Still inside the same `execute_model()` call, logits are wrapped through `model.logits(logits, hook=True)`. This fires the logits envoy's hooks, letting mediators observe/modify logits before sampling. The user accesses this as `model.logits.output`.

### Phase 3: Sampling (`_sample`)

A separate interleaver context wraps `super()._sample()`. After sampling, the sampled token IDs are wrapped through `model.samples(token_ids, hook=True)`. The user accesses this as `model.samples.output`.

### Phase 4: Finish (`finish_nnsight`)

When all requests in an invoke group complete, the interleaver handles the `"result"` provider. This lets mediators interact with the final generation output (accessed via `tracer.result`). Saved values are then extracted from mediator frames.

### Shared Mediator Threads

The same mediator threads persist across all phases within a generation step. The interleaver context is entered/exited multiple times, but mediator threads are not restarted — they continue waiting for the next value from wherever they left off in the user's intervention code.

---

## Tensor Parallelism

When `tensor_parallel_size > 1`, vLLM shards tensors across GPUs using `ColumnParallelLinear` and `RowParallelLinear` layers. Intervention code must see complete, unsharded tensors.

### VLLMBatcher's Role

`VLLMBatcher.wrap(model)` registers PyTorch hooks (not NNsight hooks) on every module:

- **Pre-input hooks**: Track the current module and whether its input is sharded (`RowParallelLinear` with `input_is_parallel=True`)
- **Pre-output hooks**: Track whether the output is sharded (`ColumnParallelLinear` with `gather_output=False`)

When a mediator requests a value via `narrow()`, `check_gathered()` is called first:

| Layer Type | Access | Gather Operation |
|------------|--------|------------------|
| `ColumnParallelLinear` | output | `tensor_model_parallel_all_gather()` on last dim |
| `RowParallelLinear` | input | `tensor_model_parallel_all_gather()` on last dim |
| `RowParallelLinear` | output | `tensor_model_parallel_all_reduce()` then divide by `tp_size` |

After intervention code runs and returns (potentially modified) values, post-hooks re-shard:

| Layer Type | Access | Re-shard Operation |
|------------|--------|--------------------|
| `ColumnParallelLinear` | output | `split_tensor_along_last_dim` → take `tp_rank` shard |
| `RowParallelLinear` | output | Divide by `tp_size` (to undo the all-reduce) |

Every GPU runs the **same intervention code** on the **same complete tensor**, ensuring consistency across the distributed system.

---

## Continuous Batching

vLLM uses continuous batching: new requests can join and finished requests can leave the batch between generation steps.

### How NNsight Handles This

**New requests**: `process_new_reqs()` deserializes mediators and registers them with the interleaver.

**Batch group updates**: `process_batch_groups()` recomputes batch groups every step based on what the scheduler has actually scheduled. Only currently-scheduled requests are reflected in batch groups.

**Finished requests**: When all requests belonging to an invoke group are finished (per-invoke-group, not per-request), `finish_nnsight()`:
1. Enters the interleaver and handles the `"result"` provider
2. Extracts saved values from the mediator's frame locals
3. Cancels the mediator
4. Returns saved values, which get attached to `RequestOutput`

After finishing, remaining active requests have their batch groups re-computed for subsequent steps.

---

## Multi-Token Generation

When `max_tokens > 1`, the execute/sample cycle repeats for each token:

```
Step 0 (prefill):
  _update_states() → process new requests, compute batch groups
  execute_model()  → forward pass, wrap logits
  _sample()        → sample, wrap samples

Step 1 (decode):
  _update_states() → update batch groups (now 1 token per request)
  execute_model()  → forward pass, wrap logits
  _sample()        → sample, wrap samples

Step 2 (decode):
  ...same pattern...

All requests in group finish:
  finish_nnsight() → collect saves, cleanup
```

### Iteration Tracking

The interleaver appends iteration suffixes to provider strings: `model.layer.output.i0`, `.i1`, `.i2`, etc. This disambiguates the same module being called multiple times across generation steps.

In user code, `tracer.iter[:]` or `tracer.iter[0:3]` iterates over generation steps:

```python
with model.trace("Hello", max_tokens=3) as tracer:
    logits = list().save()
    for step in tracer.iter[:]:
        logits.append(model.logits.output)
```

Each iteration of the loop corresponds to one generation step. The mediator's iteration counter advances, matching the interleaver's provider iteration suffix.

### Batch Group Differences: Prefill vs Decode

During prefill (step 0), a request's batch group covers all prompt tokens: `[start, num_prompt_tokens]`. During decode (steps 1+), it covers a single token: `[start, 1]`. The `unflatten()` call after the forward pass normalizes these back to prompt-level regardless.
