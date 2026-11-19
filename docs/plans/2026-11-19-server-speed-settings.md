# Server speed settings: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development to build this plan ticket by ticket, by the owner's ticket method ("How the tickets are run" below). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every speed setting of the three model servers, of Uvicorn, of the in-process model and of the workflow app's HTTP client gets a variable, and each variable, new or existing, gets the default its sources name as best for this project.

**Architecture:** a setting an App owns is a field of that App's flat settings class (`QP_OVMS_*`, and three new ones: `QP_VLLM_*`, `QP_UVICORN_*`, `QP_PROCESS_*`); a setting several components share is a field of a block of the workflow file (`local_models`, `calibration`, `http`). Each server takes named settings and a pass-through; a value given both ways is refused at start-up. A default equal to the server's own is left off the command line, so the command a machine has run does not change unless the table below says "yes". vLLM gets its variables by becoming a third model-server kind, rebuilt from `claude/vllm-model-servers` in the toolkit's layout.

**Tech stack:** Python 3.11, pydantic-settings, FastAPI 0.162.2, Uvicorn 0.54.0, httpx 0.28.1, Sentence Transformers 6.2.0 on OpenVINO, OVMS 2026.4.1, vLLM 0.32.0, bash launch scripts, pytest.

**Spec:** `docs/server-settings-for-throughput-and-latency.md` (what each server's documentation says; today it is not committed, see decision 14), `docs/model-servers.md` and `experiments/` (what this project measured), and the branch `claude/vllm-model-servers` (the vLLM design the owner grilled). A measurement here wins over a document's general advice.

Base commit: `d37e46b` (`origin/main`; the local `main` ref is stale). Work branch: `claude/server-speed-settings`, cut from it.

## Global constraints

- The project's case, which every default is chosen for: CPU only (Intel x86); one Domino App is one container on Kubernetes serving one component; scale is by replicas; machines of 2 to 16 vCPUs; prompts of 750 to 920 tokens and answers capped at 128; load tests at 4 to 32 users.
- Every tunable is a settings field. No module-level constant holds one.
- A default equal to the server's own value is not written: no flag, no property, no graph line. "Not set" is `""`, `0` or an unset optional, as each field says.
- Where calibration chooses (streams, threads, batch size), 0 stays "calibrated".
- Named settings and a pass-through for each server. A parametrized test shows that both ways give the same result. A value given both ways is refused, naming both variables and both values.
- No hand-written list of component names. A setting that applies to one component type is refused on an App of another type, by the type read from the workflow file.
- No new top-level key in the workflow file. `workflow_settings.py` (488 lines) and `qp_serve.py` (476 lines) are near the 500-line limit: no ticket adds more than a few lines to either.
- Rule 9 of the README's "Where a file goes": at most 12 modules a directory. Ticket 1 exists for it.
- The toolkit never names a use case and ships no use case's value: a model, a context length or a tier is stated in the use case's files.
- Network calls keep the timeout, retries and backoff of the settings. No circuit breaker, no rate limit, no graceful degradation is added.
- Words of `CONTEXT.md`: calibration (never tuning), model-server kind, App, App part, source id, alias, weight format, stream.
- Commits are authored by the owner alone: no `Co-Authored-By`, no `--author`, no signing, no change to `git config`.
- No model download, no `docker`, no install beyond the project's virtualenv, no cloud command on the laptop. A test that runs a launch script uses the stand-ins of `tests/helpers/launch_script_stand_ins.py` for `ovms`, `uv`, `vllm` and the exporter, with no network.

## Review focus

Input classes the sources imply and that are most likely to bite. Each has its test in the ticket named.

1. A flag given both ways under two spellings (`--rest_workers 8` and `--rest-workers=8`; `-O0` and `--optimization-level 0`): refused, not passed twice. Tickets 2, 5, 11.
2. A setting of another component type on an App (an LLM flag on a reranker's App, an embeddings graph option on a reranker's, an OpenVINO property on an LLM's): refused with the variable named, not ignored. Tickets 4, 5.
3. `NUM_STREAMS` or `INFERENCE_NUM_THREADS` in a pass-through, or a performance hint (named or passed through) beside a pinned stream count: refused, since calibration owns the first two and OVMS refuses the third. Tickets 3, 4.
4. `WEB_CONCURRENCY=4` left in an environment: Uvicorn would read it and exit on an app object. The workers count is always passed, so it is ignored. Ticket 7.
5. An export written before the graph-pool directive existed, or an embeddings graph that already holds a `truncate` line: the rewrite adds or replaces, and never writes a line twice. Ticket 4.
6. An allocator library path that names no file: refused before the process restarts, and the restart happens once. Ticket 8.

---

## 1. Every variable

"CL" answers "changes the command line run today". "Doc" is `docs/server-settings-for-throughput-and-latency.md`; "MS" is `docs/model-servers.md`; "G16" is `experiments/2026-11-16-generator-ovms-pytorch-vllm-16-vcpus/`.

### 1.1 OVMS App, embedder or reranker

Read by `qp-ovms-config` from `OvmsSettings` (`config/apps/ovms_settings.py`), set in `apps.<component>.settings` of `deploy.yaml` or in `.env`. The seven OpenVINO properties are one block, `OpenVinoCpuSettings`, used twice: here as `QP_OVMS_OPENVINO__*`, and in-process as `QP_LOCAL_MODELS__OPENVINO__*` (1.5).

| Variable | Becomes | Default | Why this default | CL |
|---|---|---|---|---|
| `QP_OVMS_NUM_STREAMS` (existing) | `NUM_STREAMS` | 0 = calibrated | Calibration chooses. Its candidates now stop at the cores (1.6) | no |
| `QP_OVMS_THREADS` (existing) | `INFERENCE_NUM_THREADS` | 0 = calibrated | Calibration chooses | no |
| `QP_OVMS_OPENVINO__PERFORMANCE_HINT` | `PERFORMANCE_HINT` | `""` = not written | OVMS: never with `NUM_STREAMS`, which this project always writes (Doc, OVMS section 4). Measured: the throughput hint alone, 7.7 s against 7.6 s (MS). Set, it replaces `NUM_STREAMS`: streams are not calibrated, and a pinned `QP_OVMS_NUM_STREAMS` is refused | no |
| `QP_OVMS_OPENVINO__PERFORMANCE_HINT_NUM_REQUESTS` | `PERFORMANCE_HINT_NUM_REQUESTS` | 0 = not written | Only caps what the throughput hint creates; refused without `THROUGHPUT` | no |
| `QP_OVMS_OPENVINO__HYPER_THREADING` | `ENABLE_HYPER_THREADING` | `true` | Measured: 2.94 s against 3.38 s (`openvino_cpu_properties.py`, sitting of 15 November). The measurement wins over OpenVINO's latency default | no |
| `QP_OVMS_OPENVINO__CPU_PINNING` | `ENABLE_CPU_PINNING` | unset = not written | Left to the server: OVMS turns pinning off by itself under a container CPU quota and on otherwise (Doc, `src/ov_utils.cpp`). Measured off: 7.9 s against 7.6 s (MS) | no |
| `QP_OVMS_OPENVINO__CPU_RESERVATION` | `ENABLE_CPU_RESERVATION` | unset = not written | The advice is for several models in one server (issue 3136); an App serves one | no |
| `QP_OVMS_OPENVINO__INFERENCE_PRECISION_HINT` | `INFERENCE_PRECISION_HINT` | `""` = not written | OpenVINO already picks `bf16` on a CPU with AMX under its default execution mode. Forcing either value is conditional on answers not checked here (decision 3) | no |
| `QP_OVMS_OPENVINO__EXECUTION_MODE_HINT` | `EXECUTION_MODE_HINT` | `""` = not written | OpenVINO's own is `PERFORMANCE`, the value the docs name for speed | no |
| `QP_OVMS_OPENVINO__PROPERTIES` | any other property, as a JSON object | `{}` | The pass-through of the block | no |
| `QP_OVMS_GRAPH_QUEUE_MAX_SIZE` | the graph's `# OVMS_GRAPH_QUEUE_MAX_SIZE:` line | `""` = as exported (`AUTO`) | `AUTO` is what both exporters write and what the docs call beneficial (`docs/mediapipe.md`) | no |
| `QP_OVMS_EMBEDDINGS_TRUNCATE` | the embeddings graph's `truncate` | unset = as exported (off) | The docs give no CPU effect. An embedder's App only | no |
| `QP_OVMS_EMBEDDINGS_MAX_LENGTH` | the embeddings graph's `max_length` | 0 = as exported | The docs tie it to speed on NPU only. An embedder's App only | no |
| `QP_OVMS_REST_WORKERS` | `--rest_workers` | 0 = left out | OVMS's own, from the CPUs. Measured: 4 REST workers on 4 streams, no gain and a longer tail (MS) | no |
| `QP_OVMS_METRICS` | `--metrics_enable` | `false` | The docs: a share of CPU on light models. On, it settles "OVMS does not serve in arrival order" (MS, not done) | no |
| `QP_OVMS_LOG_LEVEL` | `--log_level` | `""` = left out (`INFO`) | The docs: keep `INFO` | no |
| `QP_OVMS_CACHE_DIR` | `--cache_dir` | `""` = left out | Shortens a start, not a request; whether it shortens calibration's starts is not measured | no |
| `QP_OVMS_ARGS` | any other `ovms` argument, as written on its command line | `""` | The pass-through of the flags | no |

### 1.2 OVMS App, LLM

Same class; passed as flags of `ovms --pull --task text_generation`. An LLM's App refuses the stream and thread pins (today's rule), the OpenVINO block, the graph queue size and the embeddings options: OVMS writes that graph itself.

| Variable | Becomes | Default | Why this default | CL |
|---|---|---|---|---|
| `QP_OVMS_DYNAMIC_SPLIT_FUSE` (existing, default changes) | `--dynamic_split_fuse` | `false` (was `true`) | Measured (G16): 20.4 tokens a second against 11.6 at 4 users, p50 23 s against 42 s, nothing changed at 1 user. The docs: off for long prompts, with the tokens below at the context length. Not measured at 16 and 32 users (decision 1) | **yes** |
| `QP_OVMS_MAX_BATCHED_TOKENS` (existing, default changes) | `--max_num_batched_tokens` | `8192` (was 0 = left out, 256) | The same run; the docs name 4096, 8192 or the context length for long prompts. 0 still leaves it out | **yes** |
| `QP_OVMS_MAX_NUM_SEQS` | `--max_num_seqs` | 0 = left out (256) | The docs give no throughput advice. Refused above the batched tokens | no |
| `QP_OVMS_CACHE_SIZE_GB` | `--cache_size` | 0 = left out (dynamic) | The docs: dynamic is "most efficient" and avoids preemption. It can take all the memory: a tier with little memory sets a size | no |
| `QP_OVMS_PREFIX_CACHING` | `--enable_prefix_caching` | unset = left out (pull mode: on) | Measured (G16): 8.3 s with the cache, 8.0 s without. The docs disagree on the default; pull mode turns it on | no |
| `QP_OVMS_KV_CACHE_PRECISION` | `--kv_cache_precision` | `""` = left out | `u8` is the one value pull mode takes. Its effect on answers is not checked here (decision 4) | no |
| `QP_OVMS_MODEL_DISTRIBUTION_POLICY` | `--model_distribution_policy` | `""` = left out | For several sockets; machines here have one | no |
| `QP_OVMS_REST_WORKERS`, `QP_OVMS_METRICS`, `QP_OVMS_LOG_LEVEL`, `QP_OVMS_CACHE_DIR`, `QP_OVMS_ARGS` | as in 1.1 | as in 1.1 | For an LLM `rest_workers` also caps its connections and its tokenizer streams; left to OVMS | no |

### 1.3 vLLM App (new kind)

Read by the new `qp-vllm-config` from `VllmSettings` (`config/apps/vllm_settings.py`), flat `QP_VLLM_*`, one App one component, as `QP_OVMS_*` (the branch had one triple per component because its settings were one global class; decision 16). Nothing runs today, so CL is "new".

| Variable | Becomes | Default | Why this default |
|---|---|---|---|
| `QP_VLLM_MAX_MODEL_LEN` (branch) | `--max-model-len` | 0 = left out | The model's own. The RAG's LLM states 4096 in its deploy file, the one value run on a machine (G16) |
| `QP_VLLM_OPTIMIZATION_LEVEL` (branch) | `--optimization-level` | unset = left out (2) | Steady speed over a fast start: the docs call 2 the production level. 0 covers `-O0` and `--enforce-eager`. The start took 17 minutes (G16): a deploy raises `wait_timeout_s` |
| `QP_VLLM_MAX_NUM_SEQS` | `--max-num-seqs` | 0 = left out (128 on CPU) | Above the 32 users of a load test; vLLM's recipes set it to the expected concurrency |
| `QP_VLLM_MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | 0 = left out (2048 on CPU) | The docs: larger gives a sooner first token. 8192 is the candidate of the sitting, by analogy with OVMS; not measured on vLLM |
| `QP_VLLM_PERFORMANCE_MODE` | `--performance-mode` | `""` = left out (`balanced`) | `throughput` only doubles the two above when unset; not measured |
| `QP_VLLM_PREFIX_CACHING` | `--enable-prefix-caching` / `--no-enable-prefix-caching` | unset = left out (on for an LLM) | vLLM's own; it turns it off itself for encoder models |
| `QP_VLLM_KV_CACHE_GIB` | `--kv-cache-memory-bytes` (GiB times 2^30) | `8` | Left out, the worker reserves 0.92 of the NUMA node's memory, the host's and not the container's (Doc, `cpu_worker.py`). 8 GiB is what the one run asked, through the legacy variable, which sets the same value (G16). The one default that differs from the server's own (decision 5). 0 leaves it out |
| `QP_VLLM_KV_CACHE_DTYPE` | `--kv-cache-dtype` | `""` = left out (`auto`) | `fp8` fits more tokens; its effect on answers is not checked (decision 4) |
| `QP_VLLM_DTYPE` | `--dtype` | `""` = left out | The docs advise `bfloat16` only when `float16` misbehaves |
| `QP_VLLM_TENSOR_PARALLEL_SIZE` | `-tp` | 0 = left out (1) | One rank per NUMA node; machines here have one |
| `QP_VLLM_STREAM_INTERVAL` | `--stream-interval` | 0 = left out (1) | The workflow app does not stream |
| `QP_VLLM_CPU_OMP_THREADS_BIND` | exports `VLLM_CPU_OMP_THREADS_BIND` | `""` = not exported (`auto`) | `auto` is what the docs describe as right: one thread per core, one NUMA node a rank |
| `QP_VLLM_CPU_RESERVED_CPUS` | exports `VLLM_CPU_NUM_OF_RESERVED_CPU` | unset = not exported (1 with one rank) | vLLM already keeps one CPU for its API server; the docs advise 1 or 2 |
| `QP_VLLM_ARGS` (branch) | any other argument, as written on vLLM's command line | `""` | The pass-through |

### 1.4 Uvicorn, both apps, and the process

`UvicornSettings` (`QP_UVICORN_*`) and `ProcessEnvironmentSettings` (`QP_PROCESS_*`), both in `config/apps/`, read by `qp-serve` and `qp-model-server`, set per App as `QP_OVMS_*` are.

| Variable | Becomes | Default | Why this default | CL |
|---|---|---|---|---|
| `QP_UVICORN_WORKERS` | `uvicorn.run(workers=)` | 1 | FastAPI's docs: on Kubernetes one process a container and more replicas; each worker loads its own copy of the model. At 1 the app object is passed as today; above 1 the app is passed as an import string, and each worker's thread budget is the CPUs divided by the workers | no |
| `QP_UVICORN_LIMIT_CONCURRENCY` | `limit_concurrency` | 0 = `None` | It sheds load with a 503 at once. This project queues overload, and lost fewer requests that way (MS) (decision 11) | no |
| `QP_UVICORN_BACKLOG` | `backlog` | 2048 | Uvicorn's own. It matters only while the event loop is blocked, and model calls run on a thread pool | no |
| `QP_UVICORN_TIMEOUT_KEEP_ALIVE_S` | `timeout_keep_alive` | 5 | Uvicorn's own; the docs tie it to neither throughput nor latency (decision 15) | no |
| `QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY` | the library put first in `LD_PRELOAD`, by one restart of the process | `""` = none | PyTorch's guide: jemalloc or TCMalloc "can get better performance". Conditional on the library being installed in the Domino environment, and not measured under `openvino` (decision 7) | no |
| `QP_PROCESS_OMP_PROC_BIND` | exports `OMP_PROC_BIND` | `""` = not exported | Under `torch` only: OpenVINO pins its own threads (the block's `cpu_pinning`). Not measured | no |
| `QP_PROCESS_KMP_AFFINITY` | exports `KMP_AFFINITY` | `""` = not exported | Reads only under Intel's OpenMP library, which the project does not install | no |

### 1.5 In-process model

The `local_models` block of the workflow file and each component's `local` block.

| Variable | Becomes | Default | Why this default | CL |
|---|---|---|---|---|
| `QP_LOCAL_MODELS__OPENVINO__PERFORMANCE_HINT` | `ov_config` in `model_kwargs` | `""` = not written | OpenVINO's own is `LATENCY`, one stream: right for a model that runs one call at a time (`thread_budget.py`) | no |
| `QP_LOCAL_MODELS__OPENVINO__INFERENCE_PRECISION_HINT` | same | `""` = not written | As in 1.1. Sentence Transformers' 0.17 times figure is for `torch` half precision, not for OpenVINO | no |
| `QP_LOCAL_MODELS__OPENVINO__HYPER_THREADING` | same | `true` | The measurement of 1.1 | no |
| `QP_LOCAL_MODELS__OPENVINO__CPU_PINNING`, `__CPU_RESERVATION`, `__EXECUTION_MODE_HINT`, `__PERFORMANCE_HINT_NUM_REQUESTS`, `__PROPERTIES` | same | not written | The block is one class; same reasons as 1.1 | no |
| `QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT` (existing) | both exports | `fp32` kept | `int8` is the faster one: about 1.5 times, measured on both servers (MS), and the export helper's default. Ranking quality on `int8` is not checked (decision 2) | no |
| `QP_LOCAL_MODELS__WORKERS` (existing) | calls at once under `torch` | 4 kept | Measured: `torch` with 4 workers served 0.90 to 0.98 requests a second against 1.00 under `openvino` on 8 vCPUs (MS). No source names a better count | no |
| `QP_LOCAL_MODELS__THREADS_PER_CORE` | the cores, as CPUs divided by it | 2 | Intel x86 with two hyper-threads a core, as every machine of the sittings. Used by 1.6 (decision 9) | see 1.6 |
| `QP_COMPONENTS__EMBEDDER__LOCAL__BATCH_SIZE` (existing) | texts per encode call | 64 kept | A query is one text; the size counts at ingestion. Sentence Transformers gives no rule but measuring | no |
| `QP_COMPONENTS__RERANKER__LOCAL__BATCH_SIZE`, `__THREADS`, `QP_COMPONENTS__EMBEDDER__LOCAL__THREADS` (existing) | | 0 = calibrated | Calibration chooses | no |

### 1.6 Calibration

| Variable | Default | Why | CL |
|---|---|---|---|
| `QP_CALIBRATION__USERS` (existing) | 4 kept | The docs: size streams to the average requests in flight, not the peak. Measured: on 8 vCPUs calibration kept 2 streams for 4 and for 16 in flight; on 16 vCPUs more streams gained 2% and doubled a lone request (MS) | no |
| OVMS stream candidates (no variable of their own) | 1, 2, 4, ... up to the cores: `max(1, cpus // threads_per_core)` | The docs: never more streams than cores. Measured: 16 streams on 8 cores took 14.8 s and 29.4 s where 1 to 4 took 3.8 s and 14.4 s (MS). Calibration never kept the candidate dropped | no pinned value changes; a start tries one candidate fewer, and a 2-vCPU App none |

### 1.7 The workflow app's HTTP client

The top-level `http` block. The code holds one shared `httpx.AsyncClient` (`composition/http_resources.py`), so these four are the block's and no component overrides them.

| Variable | Becomes | Default | Why this default | CL |
|---|---|---|---|---|
| `QP_HTTP__MAX_CONNECTIONS` | `httpx.Limits.max_connections` | 100 | httpx's own; 32 users hold at most 32 calls in flight | no |
| `QP_HTTP__MAX_KEEPALIVE_CONNECTIONS` | `max_keepalive_connections` | 20 | httpx's own. A handshake costs milliseconds against calls of 0.3 to 20 s (MS: "the server around it costs milliseconds") | no |
| `QP_HTTP__KEEPALIVE_EXPIRY_S` | `keepalive_expiry` | 5.0 | httpx's own; the docs tie it to nothing | no |
| `QP_HTTP__HTTP2` | `http2` | `false` | Needs `h2`, not installed, and a server that speaks HTTP/2: Uvicorn's is experimental, OVMS has none. `true` without `h2` is refused, naming the `httpx[http2]` extra | no |

## 2. Example files, as they would change

Only added or changed lines are shown, and they state only what differs from a default. Nothing is written before the owner has seen these.

### 2.1 `query_processing_rag_usecase/.env.example`

After the `QP_HTTP__BACKOFF_S` line:

```sh
# The shared HTTP client's pool is httpx's own (100 connections, 20 kept alive, 5 s
# idle, HTTP/1.1): QP_HTTP__MAX_CONNECTIONS, QP_HTTP__MAX_KEEPALIVE_CONNECTIONS,
# QP_HTTP__KEEPALIVE_EXPIRY_S, QP_HTTP__HTTP2. Nothing here differs from it.
```

After the `QP_LOCAL_MODELS__CPUS` line:

```sh
# QP_LOCAL_MODELS__THREADS_PER_CORE=1           # 2 by default; 1 on a machine with hyper-threading off: OVMS's calibration tries no more streams than cores
# OpenVINO's CPU properties of the in-process models (README, "Variables"): only
# what a site changes, such as full precision on a CPU with AMX, to compare answers:
# QP_LOCAL_MODELS__OPENVINO__INFERENCE_PRECISION_HINT=f32
# QP_LOCAL_MODELS__OPENVINO__PROPERTIES='{"SCHEDULING_CORE_TYPE": "PCORE_ONLY"}'   # any other property; one a named setting gives is refused here
```

After the "host/port are qp-serve flags" line:

```sh
# What a process of qp-serve or qp-model-server runs as, each at Uvicorn's own
# default (README, "Variables"): QP_UVICORN_WORKERS, QP_UVICORN_LIMIT_CONCURRENCY,
# QP_UVICORN_BACKLOG, QP_UVICORN_TIMEOUT_KEEP_ALIVE_S. On Domino, replicas, not workers.
# QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2   # preloaded by one restart; the environment must hold the library
# QP_PROCESS_OMP_PROC_BIND=close                # torch only; OpenVINO pins its own threads
```

### 2.2 `query_processing_rag_usecase/deploy.example.yaml`

The alias list gains one line:

```yaml
  # The LLM served too, by OVMS or by vLLM in an App of its own (apps.llm below,
  # and its model under overrides.components.llm); its run label ends +llm=<kind>:
  # split-all-llm: {embedder: served, reranker: served, llm: served}
```

`defaults.server`'s comment names the third kind:

```yaml
  # own: fastapi (qp-model-server) | ovms (OpenVINO Model Server) | vllm. Each kind is
```

The `wait_timeout_s` comment:

```yaml
# A vLLM App compiles its model when it starts (17 minutes on the check of
# 16 November 2026): a deploy with one raises the wait, to 2400 for example.
# wait_timeout_s: 600
```

`apps.reranker` gains what an OVMS App can state beyond its two pins:

```yaml
    # Read OVMS's own metrics on this App (the time a request waits for an
    # inference request against the time it computes):
    # settings: {QP_OVMS_NUM_STREAMS: 2, QP_OVMS_THREADS: 8, QP_OVMS_METRICS: true}
```

`apps.llm` loses its two batching lines, which are the defaults now, and gains the vLLM form:

```yaml
  # llm:
  #   server: ovms
  #   hardware_tier_id: <a tier with 16 cores>
  #   settings:
  #     # A folder this App can write, where OVMS pulls the model on its first
  #     # start (2 GB):
  #     QP_OVMS_MODELS_DIR: /domino/datasets/local/<dataset>/ovms-llm
  #     # OVMS reads each prompt in one pass by default (--dynamic_split_fuse false
  #     # --max_num_batched_tokens 8192, better at 4 users on the check of
  #     # 16 November 2026). OVMS's own batching back:
  #     # QP_OVMS_DYNAMIC_SPLIT_FUSE: true
  #     # QP_OVMS_MAX_BATCHED_TOKENS: 0
  # The same App on vLLM. Its CPU wheel needs glibc 2.39 (Ubuntu 24.04), and
  # Domino 6.2's standard environment is Ubuntu 22.04: the App names an
  # environment of its own, with uv or with the vllm command. Its tier needs
  # 32 GB (it held 20 GB on the check).
  # llm:
  #   server: vllm
  #   environment_id: <an environment on Ubuntu 24.04>
  #   hardware_tier_id: <a tier with 16 cores and 32 GB>
  #   settings:
  #     QP_VLLM_MAX_MODEL_LEN: 4096
```

`overrides.components.llm` names the vLLM model beside the OVMS one:

```yaml
    #   source_id: OpenVINO/Qwen3.5-2B-int4-ov
    #   alias: qwen3.5-2b
    #   # On vLLM, the weights it loads as published, from a folder of a Dataset
    #   # (a Domino App cannot reach the Hub):
    #   # source_id: /domino/datasets/local/<dataset>/models/Qwen3.5-2B-int4-AutoRound
```

### 2.3 Files that do not change

`workflow.yaml`, `prompts.yaml`, `routes.yaml` and the use case's `Makefile` (nothing differs from a default), `query_processing/docker-compose.yml` (its two OVMS services serve the export as written), `tests/helpers/fixture_workflow.yaml`.

### 2.4 Other files with words to change (ticket 13)

`query_processing/domino/Dockerfile` (a header paragraph: neither image is for a vLLM App), `README.md` ("Variables", "OpenVINO Model Server", "In-process models on OpenVINO", "Deploying on Domino 6.2", the tree, rule 2's count of commands, rule 9's list), `CONTEXT.md` (Model server, Model-server kind, /v3 routes, Served model: the branch's wording), `docs/model-servers.md` (the vLLM section of the branch, and "How each is set up here"), `docs/server-settings-for-throughput-and-latency.md` (its "Here today" columns), `CLAUDE.md` (the fake-backend paragraph is untouched; the toolkit's line gains the vLLM kind).

## 3. Settings that get no variable

| Setting | Why not |
|---|---|
| `nireq` | It never reaches an embeddings or a rerank graph: their queue is OpenVINO's optimal request count (Doc, `src/sidepacket_servable.cpp`) |
| `--grpc_workers`, `grpc_max_threads` | An App starts OVMS with `--rest_port` alone: no gRPC port is open. `QP_OVMS_ARGS` reaches them if one day it is |
| `DYNAMIC_QUANTIZATION_GROUP_SIZE` (OVMS LLM) | Pull mode has no flag for it at 2026.4.1 (`docs/parameters.md`, "Text generation"), and the LLM's graph is OVMS's own, which this project does not rewrite. The docs give no CPU value (decision 17) |
| NUMA binding: `numactl`, one OVMS per NUMA node, `--cpuset-cpus` | Machines of 2 to 16 vCPUs have one NUMA node. "One instance per node behind a balancer" is what replicas already are, and where a replica lands is Kubernetes' choice (decision 8) |
| vLLM's Docker rights (`--cap-add SYS_NICE`, seccomp, `--shm-size`) | They are the container's, which Domino sets, not a value an App can state. Ticket 13 writes them in the deploy example and in `docs/model-servers.md` as what to ask an administrator |
| vLLM `--enforce-eager` | One variable for one choice: `QP_VLLM_OPTIMIZATION_LEVEL=0` is the same start with no compilation on a CPU (Doc, start-up table) |
| `VLLM_CPU_KVCACHE_SPACE` | The docs call it legacy; `QP_VLLM_KV_CACHE_GIB` sets the same value by the flag. Found in an App's environment beside the named setting, it is refused |
| The ONNX backend of Sentence Transformers | Its chart puts OpenVINO first on an Intel CPU for embedders and cross-encoders alike. A third runtime is an export, a folder rule and a calibration, not a variable (decision 13) |
| `torch_dtype` `float16` or `bfloat16` in-process | Sentence Transformers: "must be avoided altogether" on a CPU for cross-encoders, 0.17 times the speed |
| Uvicorn `loop`, `http` | Already the best ones: `uvloop` and `httptools` come with `uvicorn[standard]` and are picked by themselves |
| AnyIO's thread limit (40) | Model calls run on the project's own bounded pool (`model_pool.py`), not on AnyIO's |

## 4. How the tickets are run

One clean-context agent a ticket (Opus 5.5, medium effort, its own worktree, `git switch -c ticket-NN <sha>` from the work branch's tip at that moment). Fable 5.2 orchestrates, writes ticket 13 and runs ticket 14. At most two tickets at once, and only the pairs the table allows.

Each ticket agent: writes the failing tests first and sees them fail, builds, runs `make lint`, `ty check src`, `make complexity deadcode deps`, then `make test` in the background, alone, with `PYTEST_ADDOPTS=--basetemp=<a folder of its own>` and its output in a file. It does not run mypy. It does not edit `README.md`, `CONTEXT.md`, `CLAUDE.md`, `.env.example` or `deploy.example.yaml`: it lists the sentences it made false, for ticket 13. A ticket that adds a module fails `test_the_readme_tree_lists_every_module_and_nothing_else` until the orchestrator lists the module; its brief says so. A ticket that changes a settings class runs `tests/deploy` too.

The orchestrator, on each report: checks author and paths, cherry-picks, runs mypy on the changed files alone (after `rtk proxy ps` shows no other mypy), and keeps one numbered file of the points agents leave open.

| Order | Tickets at once | Why these two do not meet |
|---|---|---|
| 1 | 1 | A move: alone |
| 2 | 2, 3 | 2 writes `config/apps/flag_pass_through.py`; 3 writes the OpenVINO block, `shared_settings.py`, `local_model.py` |
| 3 | 4, 9 | 4 is OVMS's graph and calibration; 9 is the `http` block and `http_resources.py` |
| 4 | 5, 7 | 5 is OVMS's flags and `ovms.sh`; 7 is Uvicorn and the two commands' `main` |
| 5 | 6, 8 | 6 is two defaults of `ovms_settings.py`; 8 is the process settings and the two `main` again |
| 6 | 10, 11 | 10 is `deploy/`, `domain/` and `loadtest/`; 11 is `config/apps/vllm_settings.py` and a new command |
| 7 | 12 | The launch script; it needs 10 and 11 |
| 8 | 13, then 14 | The orchestrator's |

Dependencies: 1 → 2, 3 → 4 → 5 → 6; 3 → 9 only by the shared file; 7 → 8; 2 → 11; 10, 11 → 12; all → 13 → 14.

## 5. The tickets

Paths are relative to `query_processing/` unless they start with another top-level folder. `Q` is `src/query_processing`.

### Ticket 1: the App-own settings get a sub-package

A move: no behaviour, no test written first. The count of tests before and after is equal.

**Files:**
- Move: `Q/config/ovms_settings.py` → `Q/config/apps/ovms_settings.py`; `tests/config/test_ovms_settings.py` → `tests/config/apps/test_ovms_settings.py`
- Create: `Q/config/apps/__init__.py` (one docstring line: the settings one App owns, no value of the workflow file), `tests/config/apps/__init__.py` if the tests folders hold one

- [ ] Record `make test`'s count. `git mv` both files.
- [ ] Rewrite the imports with a script over the exact dotted name `query_processing.config.ovms_settings` and its relative forms; read the diff. No diff line that is not an import or a path in a docstring.
- [ ] `tests/structure/` green; the same count. Commit.

Why: tickets 3, 5, 7, 8 and 11 add five settings modules and their tests, and `tests/config/` holds 8 modules of the 12 rule 9 allows.

### Ticket 2: one pass-through helper for both servers

**Files:**
- Create: `Q/config/apps/flag_pass_through.py`
- Test: `tests/config/apps/test_flag_pass_through.py`

**Interfaces (produces):**
- `split_pass_through(words: str, *, variable: str) -> list[str]`: `shlex.split`; an unbalanced quote raises `ValueError` naming `variable` and the text.
- `names_flag(word: str, flag: str) -> bool`: true for the flag itself and for `flag=value`; `_` and `-` read alike.
- `refuse_given_twice(named: Mapping[str, str], words: Sequence[str], *, pass_through: str) -> None`: `named` maps each flag a named setting gives to that setting's variable with its value (`"QP_OVMS_REST_WORKERS=4"`); raises `ValueError` holding both, and "state it once".

- [ ] **Failing tests**

```python
@pytest.mark.parametrize("word", ["--rest_workers", "--rest-workers", "--rest_workers=8"])
def test_a_flag_is_recognised_under_each_spelling(word):
    assert names_flag(word, "--rest_workers")

def test_another_flag_with_the_same_start_is_not_it():
    assert not names_flag("--rest_workers_extra", "--rest_workers")

def test_an_unbalanced_quote_names_the_variable():
    with pytest.raises(ValueError, match="QP_OVMS_ARGS"):
        split_pass_through("--log_level 'DEBUG", variable="QP_OVMS_ARGS")

def test_a_flag_given_both_ways_is_refused_naming_both():
    with pytest.raises(ValueError, match=r"QP_OVMS_REST_WORKERS=4.*QP_OVMS_ARGS"):
        refuse_given_twice({"--rest_workers": "QP_OVMS_REST_WORKERS=4"},
                           ["--rest-workers=8"], pass_through="QP_OVMS_ARGS")
```

- [ ] See them fail, build, see them pass, gate, commit.

### Ticket 3: the OpenVINO CPU block, and the in-process model reads it

**Files:**
- Create: `Q/config/openvino_cpu_settings.py`; `tests/config/test_openvino_cpu_settings.py`
- Modify: `Q/infrastructure/openvino_cpu_properties.py`, `Q/infrastructure/local_models/local_model.py`, `Q/infrastructure/ovms/ovms_graph.py` (signature only), `Q/config/shared_settings.py` (`LocalModelsSettings`), `Q/config/apps/ovms_settings.py` (one field, the nested delimiter), `Q/composition/local_model_resources.py`, `Q/composition/ovms_stages.py` (hands the mapping on)
- Test: `tests/infrastructure/test_openvino_cpu_properties.py` (new or existing), `tests/infrastructure/local_models/test_local_model.py`, `tests/config/test_shared_settings.py`

**Interfaces (produces):**
- `class OpenVinoCpuSettings(SettingsBlock)` with `performance_hint: Literal["", "LATENCY", "THROUGHPUT"] = ""`, `performance_hint_num_requests: int = 0`, `hyper_threading: bool = True`, `cpu_pinning: bool | None = None`, `cpu_reservation: bool | None = None`, `inference_precision_hint: Literal["", "f32", "bf16"] = ""`, `execution_mode_hint: Literal["", "PERFORMANCE", "ACCURACY"] = ""`, `properties: dict[str, str] = {}`.
- `OpenVinoCpuSettings.chosen() -> dict[str, str]`: OpenVINO's names and spellings (`YES`/`NO` for a bool), named ones first, only what is set; `{"ENABLE_HYPER_THREADING": "YES"}` at the defaults. The comment on `hyper_threading` carries the measurement (2.94 s against 3.38 s, 15 November).
- `chosen()` writes a named field's property unless `properties` gives it and the named field was not set (`model_fields_set`): the pass-through is then the one statement. A validator refuses: a key of `properties` whose named field is also set, naming both variables; `NUM_STREAMS` or `INFERENCE_NUM_THREADS` in `properties` (calibration's); `performance_hint_num_requests` above 0 without `THROUGHPUT`.
- `openvino_cpu_properties(threads: int, chosen: Mapping[str, str]) -> dict[str, str]`: `{"INFERENCE_NUM_THREADS": str(threads), **chosen}`.
- `LocalModelsSettings.openvino: OpenVinoCpuSettings` and `LocalModelsSettings.threads_per_core: int = Field(default=2, ge=1)`; `OvmsSettings.openvino: OpenVinoCpuSettings` with `env_nested_delimiter="__"`.
- `LocalModel` gains `openvino_properties: Mapping[str, str]`; `local_model(..., openvino_properties=)`; `ovms_plugin_config(tuning: StageTuning, chosen: Mapping[str, str])`.

- [ ] **Failing tests**

```python
def test_the_defaults_write_what_runs_today():
    assert openvino_cpu_properties(4, OpenVinoCpuSettings().chosen()) == {
        "INFERENCE_NUM_THREADS": "4", "ENABLE_HYPER_THREADING": "YES"}

@pytest.mark.parametrize(("named", "passed"), [
    ({"performance_hint": "THROUGHPUT"}, {"PERFORMANCE_HINT": "THROUGHPUT"}),
    ({"cpu_pinning": False}, {"ENABLE_CPU_PINNING": "NO"}),
    ({"cpu_reservation": True}, {"ENABLE_CPU_RESERVATION": "YES"}),
    ({"inference_precision_hint": "f32"}, {"INFERENCE_PRECISION_HINT": "f32"}),
    ({"execution_mode_hint": "ACCURACY"}, {"EXECUTION_MODE_HINT": "ACCURACY"}),
])
def test_a_property_named_or_passed_through_gives_the_same_properties(named, passed):
    assert OpenVinoCpuSettings(**named).chosen() == OpenVinoCpuSettings(properties=passed).chosen()

def test_hyper_threading_passed_through_replaces_the_default():
    assert OpenVinoCpuSettings(properties={"ENABLE_HYPER_THREADING": "NO"}).chosen() == {
        "ENABLE_HYPER_THREADING": "NO"}

def test_a_property_with_no_named_setting_is_written_as_it_is():
    chosen = OpenVinoCpuSettings(properties={"SCHEDULING_CORE_TYPE": "PCORE_ONLY"}).chosen()
    assert chosen["SCHEDULING_CORE_TYPE"] == "PCORE_ONLY"

def test_a_property_given_both_ways_is_refused_naming_both():
    with pytest.raises(ValueError, match=r"hyper_threading.*ENABLE_HYPER_THREADING"):
        OpenVinoCpuSettings(hyper_threading=True, properties={"ENABLE_HYPER_THREADING": "NO"})

@pytest.mark.parametrize("calibrated", ["NUM_STREAMS", "INFERENCE_NUM_THREADS"])
def test_a_property_calibration_owns_is_refused_in_the_pass_through(calibrated):
    with pytest.raises(ValueError, match=calibrated):
        OpenVinoCpuSettings(properties={calibrated: "1"})

@pytest.mark.parametrize("variable", ["QP_LOCAL_MODELS__OPENVINO__HYPER_THREADING",
                                      "QP_OVMS_OPENVINO__HYPER_THREADING"])
def test_the_block_is_read_from_both_homes(variable, monkeypatch): ...  # false there -> "NO" in chosen()

def test_an_in_process_model_loads_with_the_blocks_properties():
    model = LocalModel("m", "m", "openvino", {"ENABLE_HYPER_THREADING": "YES", "PERFORMANCE_HINT": "LATENCY"})
    assert model.load_arguments(2)["model_kwargs"]["ov_config"]["PERFORMANCE_HINT"] == "LATENCY"
```

- [ ] See them fail, build, see them pass. `tests/deploy` too (a new key under `local_models` is checked by `check_overrides`). Gate, commit.

### Ticket 4: OVMS's graph takes the hint, the pool and the embeddings options; streams stop at the cores

**Files:**
- Modify: `Q/config/apps/ovms_settings.py`, `Q/infrastructure/ovms/ovms_graph.py`, `Q/infrastructure/ovms/ovms_serving_folder.py`, `Q/composition/ovms_stages.py`, `Q/composition/ovms_calibration.py`, `Q/entrypoints/qp_ovms_config.py` (the refusals by type)
- Test: `tests/infrastructure/ovms/test_ovms_graph.py`, `tests/infrastructure/ovms/test_ovms_serving_folder.py`, `tests/composition/test_ovms_calibration.py`, `tests/config/apps/test_ovms_settings.py`, `tests/entrypoints/test_qp_ovms_config.py`

**Interfaces:**
- Consumes: `OvmsSettings.openvino.chosen()`, `LocalModelsSettings.threads_per_core`, `ovms_plugin_config(tuning, chosen)`.
- Produces: `OvmsSettings.graph_queue_max_size: str = ""` (`""`, `AUTO`, or digits), `embeddings_truncate: bool | None = None`, `embeddings_max_length: int = 0`; `OvmsSettings.require_no_embeddings_option(component: str)` and `require_no_graph_option(component: str)` (an LLM's App: the OpenVINO block, the queue size, the embeddings options), built on `_refuse_set`.
- `ovms_plugin_config`: when `chosen` holds `PERFORMANCE_HINT`, `NUM_STREAMS` is left out.
- `graph_with_queue_size(graph_text: str, size: str, *, source: str) -> str`: replaces the `# OVMS_GRAPH_QUEUE_MAX_SIZE: <value>` line, or puts it first when the export has none.
- `graph_with_embeddings_options(graph_text: str, *, truncate: bool | None, max_length: int, source: str) -> str`: after the `plugin_config` line, `truncate: true,` and `max_length: N,` as the exporter's template writes them; an existing line is replaced, `truncate=False` removes it.
- `write_ovms_serving_folder(export, serving_dir, plugin_config, graph_options: OvmsGraphOptions)` where `OvmsGraphOptions` is a frozen dataclass of the three values, in `ovms_graph.py`.
- `ovms_candidates(cpus: int, *, streams: int, threads: int, cores: int, hinted: bool) -> Candidates`: unpinned streams go 1, 2, 4, ... up to `cores`; with `hinted`, streams are `(1,)` and are not searched. A validator on `OvmsSettings` refuses a hint beside a pinned `num_streams`.

- [ ] **Failing tests**

```python
def test_a_performance_hint_replaces_num_streams():
    config = ovms_plugin_config(StageTuning(threads=4, batch_size=1, streams=1),
                                {"PERFORMANCE_HINT": "THROUGHPUT", "ENABLE_HYPER_THREADING": "YES"})
    assert "NUM_STREAMS" not in config and config["INFERENCE_NUM_THREADS"] == "4"

def test_a_hint_beside_a_pinned_stream_count_is_refused():
    with pytest.raises(ValueError, match=r"QP_OVMS_OPENVINO__PERFORMANCE_HINT.*QP_OVMS_NUM_STREAMS=2"):
        OvmsSettings(num_streams=2, openvino={"performance_hint": "THROUGHPUT"})

@pytest.mark.parametrize(("graph", "expected_first_line"), [
    ("# OVMS_GRAPH_QUEUE_MAX_SIZE: AUTO\ninput_stream: ...", "# OVMS_GRAPH_QUEUE_MAX_SIZE: 4"),
    ("input_stream: ...", "# OVMS_GRAPH_QUEUE_MAX_SIZE: 4"),
])
def test_the_queue_size_is_replaced_or_added_once(graph, expected_first_line): ...

def test_truncate_is_written_once_on_a_graph_that_already_has_it(): ...   # count of "truncate:" lines == 1
def test_the_default_options_leave_the_exports_graph_text_but_its_plugin_config(): ...

def test_stream_candidates_stop_at_the_cores():
    assert ovms_candidates(16, streams=0, threads=0, cores=8, hinted=False) == ((1, 2, 4, 8), (16, 8, 4, 2, 1))
    assert ovms_candidates(2, streams=0, threads=0, cores=1, hinted=False) == ((1,), (2, 1))

def test_a_pinned_stream_count_above_the_cores_is_still_tried():   # a pin is the owner's choice
    assert ovms_candidates(4, streams=4, threads=0, cores=2, hinted=False)[0] == (4,)

@pytest.mark.parametrize("variable", ["QP_OVMS_EMBEDDINGS_TRUNCATE", "QP_OVMS_EMBEDDINGS_MAX_LENGTH"])
def test_a_rerankers_app_refuses_an_embeddings_option(variable): ...      # qp-ovms-config --serve reranker exits 1, one line
def test_an_llms_app_refuses_the_openvino_block_and_the_queue_size(): ... # qp-ovms-config --serve-llm llm exits 1
```

- [ ] See them fail, build, see them pass, gate, commit.

### Ticket 5: OVMS's server flags and the LLM's remaining flags

**Files:**
- Modify: `Q/config/apps/ovms_settings.py`, `Q/entrypoints/qp_ovms_config.py`, `Q/infrastructure/ovms/ovms_process.py`, `Q/composition/ovms_stages.py` (`OvmsServer` carries the flags), `domino/model_servers/ovms.sh`
- Test: `tests/config/apps/test_ovms_settings.py`, `tests/entrypoints/test_qp_ovms_config.py`, `tests/infrastructure/ovms/test_ovms_process.py`, `tests/scripts/test_domino_app_sh_ovms_flags.py` (new), `tests/scripts/test_domino_app_sh_llm.py`

**Interfaces:**
- Consumes: ticket 2's three functions.
- Produces, on `OvmsSettings`: `rest_workers: int = 0`, `metrics: bool = False`, `log_level: Literal["", "DEBUG", "INFO", "ERROR"] = ""`, `cache_dir: str = ""`, `args: str = ""`; an LLM's: `max_num_seqs: int = 0`, `cache_size_gb: int = 0`, `prefix_caching: bool | None = None`, `kv_cache_precision: Literal["", "u8"] = ""`, `model_distribution_policy: Literal["", "TENSOR_PARALLEL", "PIPELINE_PARALLEL"] = ""`.
- `OvmsSettings.server_flags() -> list[str]`: the named flags set, then the pass-through; `[]` at the defaults. `OvmsSettings.llm_flags() -> list[str]`: the LLM's named flags (today's two first), without the server flags. A validator refuses a flag given both ways (`args` against every named flag of both lists, and against `--rest_port`, `--rest_bind_address`, `--config_path`, `--source_model`, `--model_name`, `--model_repository_path`, `--task`, `--target_device`, which the launch script owns), and `max_num_seqs` above `max_batched_tokens` when both are above 0.
- `require_no_llm_option` covers the five new LLM fields too.
- `qp-ovms-config --server-flags NAME`: prints `shlex.join(ovms.server_flags())` for any served component. `--serve-llm NAME` prints the LLM's flags followed by the server flags.
- `OvmsProcess(..., server_flags: Sequence[str] = ())`: appended to the command, so a candidate is timed as it will be served.
- `ovms.sh`: `ovms_run --config_path "$config"` gains the words of `qp-ovms-config --server-flags "$component"`. Without `qp-ovms-config` on PATH, a set `QP_OVMS_REST_WORKERS`, `QP_OVMS_METRICS`, `QP_OVMS_LOG_LEVEL`, `QP_OVMS_CACHE_DIR`, `QP_OVMS_ARGS`, `QP_OVMS_GRAPH_QUEUE_MAX_SIZE`, `QP_OVMS_EMBEDDINGS_*` or `QP_OVMS_OPENVINO__*` stops the App, as a pinned stream count does today (one loop over `compgen -v QP_OVMS_` minus the three the script itself reads, not a list written twice).

- [ ] **Failing tests**

```python
def test_the_defaults_add_no_server_flag():
    assert OvmsSettings().server_flags() == []

@pytest.mark.parametrize(("named", "passed"), [
    ({"rest_workers": 4}, "--rest_workers 4"),
    ({"metrics": True}, "--metrics_enable"),
    ({"log_level": "DEBUG"}, "--log_level DEBUG"),
    ({"cache_dir": "/cache"}, "--cache_dir /cache"),
])
def test_a_server_flag_named_or_passed_through_gives_the_same_command_line(named, passed):
    assert OvmsSettings(**named).server_flags() == OvmsSettings(args=passed).server_flags()

@pytest.mark.parametrize(("named", "passed"), [
    ({"rest_workers": 4}, "--rest-workers=8"), ({"log_level": "INFO"}, "--log_level DEBUG"),
    ({}, "--rest_port 9000"), ({}, "--config_path /x.json"),
])
def test_a_flag_given_both_ways_or_owned_by_the_script_is_refused(named, passed):
    with pytest.raises(ValueError, match="QP_OVMS_ARGS"):
        OvmsSettings(**named, args=passed)

@pytest.mark.parametrize(("named", "flags"), [
    ({"max_num_seqs": 64}, ["--max_num_seqs", "64"]),
    ({"cache_size_gb": 8}, ["--cache_size", "8"]),
    ({"prefix_caching": False}, ["--enable_prefix_caching", "false"]),
    ({"kv_cache_precision": "u8"}, ["--kv_cache_precision", "u8"]),
    ({"model_distribution_policy": "TENSOR_PARALLEL"}, ["--model_distribution_policy", "TENSOR_PARALLEL"]),
])
def test_an_llm_option_becomes_its_flag(named, flags):
    assert OvmsSettings(dynamic_split_fuse=True, max_batched_tokens=0, **named).llm_flags() == flags

def test_more_sequences_than_batched_tokens_is_refused(): ...   # max_num_seqs=512, max_batched_tokens=256
```

Script tests, with the `ovms` stand-in that records its arguments:

```python
def test_a_default_rerankers_app_runs_todays_command_line(): ...   # argv == ["--rest_port", "8888", "--config_path", <config>]
def test_rest_workers_reaches_ovms_on_a_rerankers_app(): ...       # QP_OVMS_REST_WORKERS=4 -> argv ends "--rest_workers", "4"
def test_the_pass_through_reaches_ovms_on_an_llms_app(): ...        # QP_OVMS_ARGS="--log_level DEBUG"
def test_a_flag_setting_without_qp_ovms_config_stops_the_app(): ...
```

- [ ] See them fail, build, see them pass, gate (`shellcheck` is in `make lint`), commit.

### Ticket 6: an LLM on OVMS reads each prompt in one pass by default

**Files:**
- Modify: `Q/config/apps/ovms_settings.py` (two defaults, the comment above them, two docstring examples), `domino/model_servers/ovms.sh` (the header's paragraph)
- Test: `tests/config/apps/test_ovms_settings.py`, `tests/entrypoints/test_qp_ovms_config.py`, `tests/scripts/test_domino_app_sh_llm.py`, `query_processing_rag_usecase/tests/test_deploy_dry_run.py` if it pins the LLM App's settings

`dynamic_split_fuse: bool = False`, `max_batched_tokens: int = 8192`. `llm_flags()` keeps its rule, a flag only away from OVMS's own value, so it now writes both by default; `true` and `0` leave them out. The comment cites G16: 20.4 tokens a second against 11.6 at 4 users, prompts of 750 to 920 tokens, and says that 16 and 32 users are not measured.

- [ ] **Failing tests**

```python
def test_an_llms_default_command_line_reads_each_prompt_in_one_pass():
    assert OvmsSettings().llm_flags() == ["--dynamic_split_fuse", "false",
                                          "--max_num_batched_tokens", "8192"]

def test_ovms_own_batching_is_one_setting_each_away():
    assert OvmsSettings(dynamic_split_fuse=True, max_batched_tokens=0).llm_flags() == []

def test_a_rerankers_app_still_starts_at_the_defaults():      # require_no_llm_option reads model_fields_set
    OvmsSettings().require_no_llm_option("reranker")
```

Script test: a default LLM App's recorded `ovms` arguments end with those four words.

- [ ] See them fail, change the two defaults, fix the tests that pinned the old ones, gate, commit. The commit message says the command line of an LLM's App changes and that a sitting has to confirm it (section 7).

### Ticket 7: Uvicorn's four settings

**Files:**
- Create: `Q/config/apps/uvicorn_settings.py`, `Q/infrastructure/http/uvicorn_run.py`; `tests/config/apps/test_uvicorn_settings.py`, `tests/infrastructure/http/test_uvicorn_run.py`
- Modify: `Q/entrypoints/qp_serve.py` and `Q/entrypoints/qp_model_server.py` (`main`, and one factory each), `Q/composition/local_model_resources.py` (`_work_out_unpinned_budget`), `Q/composition/composition_root.py` only if it is what hands the worker count on
- Test: `tests/entrypoints/qp_serve/test_qp_serve_uvicorn.py`, `tests/entrypoints/test_qp_model_server.py`, `tests/composition/test_calibrated_stages.py` or the budget's own test file

**Interfaces (produces):**
- `class UvicornSettings(BaseSettings)`, prefix `QP_UVICORN_`, `extra="ignore"`, `.env` read: `workers: int = Field(1, ge=1)`, `limit_concurrency: int = Field(0, ge=0)`, `backlog: int = Field(2048, ge=1)`, `timeout_keep_alive_s: int = Field(5, ge=0)`; `run_arguments() -> dict[str, int | None]` with Uvicorn's keyword names (`limit_concurrency` 0 becomes `None`); `variable_of(field)` as `OvmsSettings` has.
- `run_uvicorn(app: ASGIApp, factory: str, *, host: str, port: int, arguments: Mapping[str, int | None], run: Callable[..., None] = uvicorn.run) -> None`: `workers == 1` passes `app`, as today; above 1 passes `factory` (an import string) with `factory=True`.
- `qp_serve.workflow_app_from_environment() -> FastAPI` and `qp_model_server.model_server_from_environment() -> FastAPI`: build the settings and the app again in a worker. `main` exports what a worker cannot get from its arguments before it calls `run_uvicorn`: `QP_SERVE_DOMINO=1` for `--domino`, and `QP_MODEL_SERVER_COMPONENT` for `--serve` (the variable `qp-deploy` already sets).
- The thread budget of a process is worked out on `cpus / UvicornSettings().workers`.

- [ ] **Failing tests**

```python
def test_the_defaults_are_uvicorns_own():
    config = uvicorn.Config("x:y")
    assert UvicornSettings().run_arguments() == {
        "workers": 1, "limit_concurrency": config.limit_concurrency,
        "backlog": config.backlog, "timeout_keep_alive": config.timeout_keep_alive}

def test_one_worker_runs_the_app_object_as_today():
    calls = []
    run_uvicorn(app, "m:f", host="0.0.0.0", port=8888,
                arguments=UvicornSettings().run_arguments(), run=lambda *a, **k: calls.append((a, k)))
    assert calls[0][0] == (app,) and "factory" not in calls[0][1]

def test_several_workers_run_the_import_string():
    ...  # workers=2 -> args == ("m:f",), kwargs["factory"] is True and kwargs["workers"] == 2

def test_web_concurrency_in_the_environment_changes_nothing(monkeypatch):
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    assert UvicornSettings().run_arguments()["workers"] == 1

@pytest.mark.parametrize("build", [create_app_as_main_does, workflow_app_from_environment])
def test_the_app_object_and_the_factory_answer_healthz_alike(build): ...   # fake backends

def test_two_workers_share_the_cpus(): ...   # 8 CPUs, QP_UVICORN_WORKERS=2, openvino -> reranker_threads == 4
```

- [ ] See them fail, build, see them pass, gate, commit.

### Ticket 8: the process environment (allocator, OpenMP)

**Files:**
- Create: `Q/config/apps/process_environment_settings.py`, `Q/infrastructure/process_environment.py`; `tests/config/apps/test_process_environment_settings.py`, `tests/infrastructure/test_process_environment.py`
- Modify: `Q/entrypoints/qp_serve.py`, `Q/entrypoints/qp_model_server.py` (two lines each in `main`, before the settings are read)

**Interfaces (produces):**
- `class ProcessEnvironmentSettings(BaseSettings)`, prefix `QP_PROCESS_`: `memory_allocator_library: str = ""`, `omp_proc_bind: str = ""`, `kmp_affinity: str = ""`; `wanted_environment() -> dict[str, str]` (`LD_PRELOAD`, `OMP_PROC_BIND`, `KMP_AFFINITY`, only what is set). A validator refuses a library path that is no file, naming the variable and the path.
- `environment_to_restart_with(wanted: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, str] | None`: `None` when `environ` already holds every wanted value (`LD_PRELOAD` holds the library among its entries); else `environ` with them, the library first in `LD_PRELOAD`. Raises `ValueError` when `environ` holds `OMP_PROC_BIND` or `KMP_AFFINITY` at another value than the named setting: given both ways.
- `restart_with(environment: Mapping[str, str], argv: Sequence[str], execve: Callable[..., None] = os.execve) -> None`.

- [ ] **Failing tests**

```python
def test_nothing_set_restarts_nothing():
    assert environment_to_restart_with({}, {"PATH": "/bin"}) is None

def test_the_allocator_goes_first_in_ld_preload():
    env = environment_to_restart_with({"LD_PRELOAD": "/l/libjemalloc.so.2"}, {"LD_PRELOAD": "/l/a.so"})
    assert env["LD_PRELOAD"] == "/l/libjemalloc.so.2:/l/a.so"

def test_a_process_already_in_its_environment_does_not_restart_again():
    assert environment_to_restart_with({"LD_PRELOAD": "/l/j.so"}, {"LD_PRELOAD": "/l/j.so:/l/a.so"}) is None

def test_openmp_binding_given_both_ways_is_refused():
    with pytest.raises(ValueError, match=r"QP_PROCESS_OMP_PROC_BIND.*OMP_PROC_BIND=spread"):
        environment_to_restart_with({"OMP_PROC_BIND": "close"}, {"OMP_PROC_BIND": "spread"})

def test_a_library_that_is_no_file_is_refused_before_any_restart(tmp_path):
    with pytest.raises(ValueError, match="QP_PROCESS_MEMORY_ALLOCATOR_LIBRARY"):
        ProcessEnvironmentSettings(memory_allocator_library=str(tmp_path / "missing.so"))

def test_main_restarts_once_with_the_same_arguments(): ...   # a recording execve; argv kept
```

- [ ] See them fail, build, see them pass, gate, commit.

### Ticket 9: the HTTP client's pool

**Files:**
- Modify: `Q/config/shared_settings.py` (`DefaultHttpSettings`), `Q/composition/http_resources.py` (`_start_client`)
- Test: `tests/config/test_shared_settings.py`, `tests/composition/test_http_resources.py`

**Interfaces (produces):** `DefaultHttpSettings.max_connections: int = Field(100, ge=1)`, `max_keepalive_connections: int = Field(20, ge=0)`, `keepalive_expiry_s: float = Field(5.0, ge=0)`, `http2: bool = False`. `ComponentHttpSettings` does not change: one client is shared. `_start_client` builds `httpx.AsyncClient(timeout=..., limits=httpx.Limits(...), http2=..., transport=...)`. With `http2` on and `h2` not importable, `HttpResources` raises `ValueError` at construction naming `QP_HTTP__HTTP2` and `httpx[http2]`.

- [ ] **Failing tests**

```python
def test_the_defaults_are_httpx_own():
    http = DefaultHttpSettings()
    assert httpx.Limits(max_connections=http.max_connections,
                        max_keepalive_connections=http.max_keepalive_connections,
                        keepalive_expiry=http.keepalive_expiry_s) == httpx.Limits(
        max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0)

async def test_the_client_is_built_with_the_blocks_limits(): ...    # a recording client factory; max_connections=8 arrives
def test_http2_without_h2_is_refused_naming_the_extra(monkeypatch): ...
def test_a_components_http_block_refuses_a_pool_value(): ...         # components.llm.http.max_connections: unknown key
```

- [ ] See them fail, build, see them pass, `tests/deploy` too, gate, commit.

### Ticket 10: the `vllm` model-server kind, its routes, and a load test straight at it

**Files:**
- Create: `Q/deploy/model_server_kinds/vllm_kind.py`
- Modify: `Q/domain/endpoints/api_endpoint.py` (three constants), `Q/loadtest/loadtest_settings.py`, `Q/loadtest/request_plans.py`
- Test: `tests/deploy/model_server_kinds/test_model_server_kinds.py`, `tests/domain/` (the constants' test), `tests/loadtest/test_request_plans.py`, `tests/loadtest/test_loadtest_settings.py`

**Interfaces (produces):**
- `VLLM_EMBEDDINGS_PATH = "/v1/embeddings"`, `VLLM_RERANK_PATH = "/v1/rerank"`, `VLLM_CHAT_PATH = "/v1/chat/completions"`.
- `@MODEL_SERVER_KINDS.register("vllm") class VllmModelServerKind(BaseModelServerKind)`: `routes` for `embedder`, `reranker` and `llm`; `probe_route = "/v1/models"` (vLLM lists its served models there and has no route for one); no launch export, no refusal: it needs no bundle.
- `QP_LOADTEST_MODEL_SERVER_PATH: str = ""`: the path a model-server target posts to; `""` is the /v3 path of the target, as today. A vLLM run sets `/v1/rerank` or `/v1/embeddings` (decision 12).

- [ ] **Failing tests**

```python
def test_vllm_serves_the_three_model_types_on_v1():
    kind = model_server_kind("vllm")
    assert [kind.route_of(t) for t in ("embedder", "reranker", "llm")] == [
        "/v1/embeddings", "/v1/rerank", "/v1/chat/completions"]

def test_vllm_serves_no_registered_type():
    assert not model_server_kind("vllm").serves("shouter")

def test_a_rerank_target_posts_to_the_path_the_settings_name(): ...   # QP_LOADTEST_MODEL_SERVER_PATH=/v1/rerank
def test_without_it_a_rerank_target_posts_to_v3(): ...
```

`tests/deploy/model_server_kinds/test_a_new_model_server_kind.py` already holds that a kind is one class and one script: until ticket 12 writes the script, a test that pairs every kind with a script fails. This ticket's brief says so, and tickets 10 and 12 are joined before the full gate.

- [ ] See them fail, build, see them pass, gate, commit.

### Ticket 11: vLLM's settings and `qp-vllm-config`

**Files:**
- Create: `Q/config/apps/vllm_settings.py`, `Q/entrypoints/qp_vllm_config.py`; `tests/config/apps/test_vllm_settings.py`, `tests/entrypoints/test_qp_vllm_config.py`
- Modify: `pyproject.toml` (`[project.scripts]`: `qp-vllm-config`), `vulture_whitelist.py` if the gate asks

**Interfaces:**
- Consumes: ticket 2's three functions.
- Produces: `class VllmSettings(BaseSettings)`, prefix `QP_VLLM_`, with the fields of table 1.3 (`max_model_len: int = 0`, `optimization_level: int | None = None`, `max_num_seqs: int = 0`, `max_num_batched_tokens: int = 0`, `performance_mode: Literal["", "balanced", "interactivity", "throughput"] = ""`, `prefix_caching: bool | None = None`, `kv_cache_gib: float = Field(8.0, ge=0)`, `kv_cache_dtype: str = ""`, `dtype: str = ""`, `tensor_parallel_size: int = 0`, `stream_interval: int = 0`, `cpu_omp_threads_bind: str = ""`, `cpu_reserved_cpus: int | None = None`, `args: str = ""`).
- `VllmSettings.serve_flags() -> list[str]`: the named flags set, then the pass-through. At the defaults: `["--kv-cache-memory-bytes", "8589934592"]`.
- `VllmSettings.exported_environment(environ: Mapping[str, str]) -> dict[str, str]`: `VLLM_CPU_OMP_THREADS_BIND` and `VLLM_CPU_NUM_OF_RESERVED_CPU` when set. Raises `ValueError` when `environ` holds either at another value, or holds `VLLM_CPU_KVCACHE_SPACE` while `kv_cache_gib` is above 0.
- A validator refuses a flag given both ways, the branch's rule: `--max-model-len`, `--optimization-level` with `-O`, `-O3`, `-O=3` and `--enforce-eager`, every other named flag, `-tp` with `--tensor-parallel-size`, and the flags the launch script owns (`--host`, `--port`, `--root-path`, `--served-model-name`).
- `qp-vllm-config --serve NAME`: one line of shell words: the component's source id, `--served-model-name`, its alias, then `serve_flags()`. `qp-vllm-config --environment NAME`: one line of `NAME=value` words. It refuses, in one line and exit code 1, an empty source id; a name that is no model component of the workflow file (`settings.model_components()`) is a usage error. It holds no list of types: `entrypoints` may not import `deploy`, and the deploy tool has already refused a type the `vllm` kind cannot serve.

- [ ] **Failing tests**

```python
def test_the_default_command_line_bounds_the_kv_cache_and_nothing_else():
    assert VllmSettings().serve_flags() == ["--kv-cache-memory-bytes", str(8 * 2**30)]

@pytest.mark.parametrize(("named", "passed"), [
    ({"max_model_len": 4096}, "--max-model-len 4096"),
    ({"optimization_level": 0}, "--optimization-level 0"),
    ({"max_num_seqs": 32}, "--max-num-seqs 32"),
    ({"max_num_batched_tokens": 8192}, "--max-num-batched-tokens 8192"),
    ({"performance_mode": "throughput"}, "--performance-mode throughput"),
    ({"kv_cache_dtype": "fp8"}, "--kv-cache-dtype fp8"),
    ({"dtype": "bfloat16"}, "--dtype bfloat16"),
    ({"tensor_parallel_size": 2}, "--tensor-parallel-size 2"),
    ({"stream_interval": 10}, "--stream-interval 10"),
])
def test_a_flag_named_or_passed_through_gives_the_same_command_line(named, passed):
    assert VllmSettings(**named).serve_flags() == VllmSettings(args=passed).serve_flags()

@pytest.mark.parametrize(("named", "passed"), [
    ({"max_model_len": 4096}, "--max_model_len=8192"),
    ({"optimization_level": 0}, "-O3"), ({"optimization_level": 2}, "--enforce-eager"),
    ({"tensor_parallel_size": 2}, "-tp 4"), ({}, "--port 9000"),
    ({}, "--kv-cache-memory-bytes 1"),
])
def test_a_flag_given_both_ways_is_refused(named, passed):
    with pytest.raises(ValueError, match="QP_VLLM_ARGS"):
        VllmSettings(**named, args=passed)

def test_the_cache_passed_through_needs_the_named_one_at_zero():
    assert VllmSettings(kv_cache_gib=0, args="--kv-cache-memory-bytes 1").serve_flags() == [
        "--kv-cache-memory-bytes", "1"]

@pytest.mark.parametrize(("named", "environ"), [
    ({"cpu_omp_threads_bind": "0-7"}, {"VLLM_CPU_OMP_THREADS_BIND": "auto"}),
    ({}, {"VLLM_CPU_KVCACHE_SPACE": "8"}),
])
def test_a_vllm_variable_given_both_ways_is_refused(named, environ): ...

def test_qp_vllm_config_prints_the_source_the_alias_and_the_flags(): ...   # fixture workflow, one line, shlex-splittable
def test_qp_vllm_config_refuses_an_empty_source_id_in_one_line(): ...
```

- [ ] See them fail, build, see them pass, gate, commit.

### Ticket 12: the `vllm` launch script

**Files:**
- Create: `domino/model_servers/vllm.sh`; `tests/scripts/test_domino_app_sh_vllm.py` (the branch's 354-line test carried to the toolkit's script layout and names)
- Modify: `pyproject.toml` (`[dependency-groups] vllm`, the wheel URL of the branch), `domino/Dockerfile` (the header paragraph of the branch), `tests/helpers/launch_script_stand_ins.py` (a `vllm` stand-in that records its arguments and environment), `tests/deploy/test_entry_scripts_through_app_sh.py`, `query_processing_rag_usecase/tests/test_deploy_dry_run.py`

**Interfaces:**
- Consumes: `qp-vllm-config --serve NAME` and `--environment NAME` (ticket 11); the `vllm` kind (ticket 10); `install_project`, `install_this_project`, `die`, `TOOLKIT`, `PYTHON_VERSION_FILE` of `domino/app.sh`.
- Produces: `bootstrap_model_server` (nothing when `vllm` is on PATH; else, with `uv`, the `vllm` group alone into `$HOME/vllm-venv` with `--torch-backend cpu`, a failure naming glibc 2.39 and what `ldd --version` prints; Intel OpenMP preloaded when the virtualenv holds `lib/libiomp5.so`) and `serve_model_server <component>` (installs the project into `$HOME/qp-venv` when `qp-vllm-config` is missing, exports the environment line, then `exec vllm serve <words> --host 0.0.0.0 --port 8888`, with `--root-path` from `DOMINO_RUN_HOST_PATH` without its trailing slash). No nginx, no export, no calibration.

- [ ] **Failing tests** (each through `domino/app.sh` with stand-ins; none reaches a network)

```python
def test_a_default_vllm_app_runs_the_one_default_command_line(): ...
    # argv == ["serve", <source id>, "--served-model-name", <alias>,
    #          "--kv-cache-memory-bytes", "8589934592", "--host", "0.0.0.0", "--port", "8888"]
def test_a_named_setting_reaches_vllm(): ...          # QP_VLLM_MAX_MODEL_LEN=4096
def test_the_pass_through_reaches_vllm(): ...          # QP_VLLM_ARGS="--dtype bfloat16"
def test_a_named_vllm_variable_is_exported_to_vllm(): ...   # QP_VLLM_CPU_OMP_THREADS_BIND=0-7 in the stand-in's environment
def test_dominos_path_prefix_becomes_root_path(): ...
def test_without_vllm_and_without_uv_the_app_stops_naming_ubuntu_24_04(): ...
def test_a_failed_install_names_the_glibc_found(): ...
def test_qp_deploy_writes_a_vllm_apps_entry_script(): ...   # dry run: QP_MODEL_SERVER_KIND=vllm, the served URL ends /v1/chat/completions
```

- [ ] See them fail, write the script, see them pass, `make lint` (shellcheck), gate, commit.

### Ticket 13: the documents and the example files (orchestrator)

After the owner has approved section 2. The orchestrator writes, from the list of sentences each agent reported false:

- [ ] `query_processing_rag_usecase/.env.example` and `deploy.example.yaml` as previewed; the use case's tests that read them (`test_deploy_dry_run.py`, `test_image_files.py`) stay green.
- [ ] `README.md`: the "Variables" table gains every row of section 1; "OpenVINO Model Server" (the flags, the graph options, the stream candidates, the new LLM default); "In-process models on OpenVINO" (the block); "Deploying on Domino 6.2" (a vLLM section, from the branch); the tree; rule 2's count of commands (nine); rule 9 unchanged.
- [ ] `CONTEXT.md`: the four entries of the branch, in the toolkit's words.
- [ ] `docs/model-servers.md`: the vLLM section of the branch; "How each is set up here" gains the rows of this plan; the sentence on stream candidates above the cores is corrected.
- [ ] `docs/server-settings-for-throughput-and-latency.md`: each "Here today" cell names its variable (decision 14).
- [ ] `CLAUDE.md`: the toolkit's line names the three kinds.
- [ ] `make check`, commit.

### Ticket 14: integration and whole-branch review (Fable 5.2)

- [ ] mypy on the joined tree, alone.
- [ ] `make check` from the root, in the background, its output in a file.
- [ ] The non-regression check of section 6.4.
- [ ] A whole-branch review split in three: OVMS (tickets 3 to 6), the two apps (7 to 9), vLLM (10 to 12). One reviewer starts an App of each kind from the README alone, on stand-ins. Each gets the open points of the agents and the six lines of "Review focus". Scripts go in one named folder; each finding becomes a fix ticket whose first test is the script.
- [ ] The command-line audit: for each row of section 1 marked "no", one test named in section 6.3 shows the default command line equal to today's. For the two rows marked "yes", the commit of ticket 6 is the only place the line changes.
- [ ] Tell the owner how many agent worktrees are left on disk.

## 6. The tests

### 6.1 A unit test for each setting

Each field of section 1 has a test that it is read from its variable and a test of the value it becomes: tickets 3 (the OpenVINO block, both homes), 4 (graph options, candidates), 5 (flags), 6 (the two defaults), 7 (Uvicorn), 8 (process), 9 (HTTP), 11 (vLLM). Each refusal has a test whose `match` holds the variable's name and the offending value.

### 6.2 Named against pass-through, parametrized

One parametrized test a server, each case a named setting and the same thing written in the pass-through, asserting equal results: `test_a_property_named_or_passed_through_gives_the_same_properties` (ticket 3), `test_a_server_flag_named_or_passed_through_gives_the_same_command_line` (ticket 5), `test_a_flag_named_or_passed_through_gives_the_same_command_line` (ticket 11). Beside each, the parametrized refusal of a value given both ways (tickets 3, 5, 8, 11).

### 6.3 A default reaches the command line or the graph file

| What | Test | Ticket |
|---|---|---|
| The graph's `plugin_config` at the defaults is today's three properties | `test_the_defaults_write_what_runs_today` | 3 |
| The serving folder's graph differs from the export's by its `plugin_config` only | `test_the_default_options_leave_the_exports_graph_text_but_its_plugin_config` | 4 |
| A default reranker's App runs `ovms --rest_port 8888 --config_path <config>` | `test_a_default_rerankers_app_runs_todays_command_line` | 5 |
| A default LLM's App ends `--dynamic_split_fuse false --max_num_batched_tokens 8192` | the script test of ticket 6 | 6 |
| Uvicorn gets the app object and its own four values | `test_one_worker_runs_the_app_object_as_today`, `test_the_defaults_are_uvicorns_own` | 7 |
| No restart at the defaults | `test_nothing_set_restarts_nothing` | 8 |
| The client's limits are httpx's | `test_the_defaults_are_httpx_own` | 9 |
| An in-process model's `ov_config` is today's two properties | ticket 3's `load_arguments` test at the defaults | 3 |
| A default vLLM App's whole command line | `test_a_default_vllm_app_runs_the_one_default_command_line` | 12 |

### 6.4 Answers on the fake backends, before and after

Before ticket 1, on `d37e46b`: from `query_processing_rag_usecase/`, serve with every component on its fake backend (`QP_COMPONENTS__<NAME>__BACKEND=fake` for `EMBEDDER`, `ROUTER_EMBEDDER`, `ROUTER`, `RERANKER`, `LLM`, `VECTOR_STORE`), then `make record-answers RUN_NAME=before HOST=http://127.0.0.1:8888`. After ticket 13, on the branch: the same with `RUN_NAME=after`, then `make compare-answers BASELINE=before`. Expected: every record equal, `/query` and `/retrieve`. A second "after" with `QP_UVICORN_WORKERS=2` must be equal too: it is the one run of the import-string path against a real process.

## 7. Verification on machines

Nothing below runs without the owner's approval of the machines, given on a hardware table first. Models come from the cache bucket; the vLLM model has to be put there once. Each sitting is registered in `experiments/`, and `make experiment-charts` follows.

| # | What a sitting has to show | Runs | Machines to ask for |
|---|---|---|---|
| S1 | The new LLM default, through the deploy tool and the emulated Domino: `split-all-llm` on OVMS | Default against `QP_OVMS_DYNAMIC_SPLIT_FUSE=true` with `QP_OVMS_MAX_BATCHED_TOKENS=0`, at 1, 4, 16 and 32 users, 3 minutes each | The layout of 15 November with the LLM's App on one 16-vCPU machine (the owner's exception for a scaled part) |
| S2 | Calibration with streams stopped at the cores chooses as before | One OVMS reranker App start on 8 and on 16 vCPUs; read `ovms_serving_config`; compare with the choices of 15 November (2 streams; 1 and 4) | One reranker machine of each size |
| S3 | A vLLM App starts from `vllm.sh` and holds its memory bound | First start on Ubuntu 24.04 with `uv`; memory held with the 8 GiB default; 1 and 4 users. Then `QP_VLLM_MAX_NUM_BATCHED_TOKENS=8192`, and `QP_VLLM_OPTIMIZATION_LEVEL=0` for the start time | One 16-vCPU machine with 62 GB, on an Ubuntu 24.04 image |
| S4 | The three open conditions | (a) answer records of `/retrieve` on `fp32` and on `int8`: order and scores of the 10 passages; (b) the same with `INFERENCE_PRECISION_HINT=f32` on a machine with AMX, if a tier has it; (c) the FastAPI reranker with and without jemalloc, under `openvino` and `torch` | The 4 + 1 layout of 2 CPUs each for (a) and (c); (b) needs a C3 or other AMX machine |

Until S1 has run, the two rows marked "yes" in 1.2 are a default measured by hand at 4 users on one machine, not a verified one; the README says so where it gives the default. `--metrics_enable` as a bare flag is also first run in S1 or S2 (`QP_OVMS_METRICS=true` on one App).

## 8. Decisions for the owner

1. **The LLM's two batching defaults change** (`false`, `8192`). Measured at 4 users only; OVMS's docs warn that large batches cost interactivity at higher concurrency. Recommended: change now, run S1 at 16 and 32 users, go back if it serves less there.
2. **`int8` weights.** Faster by about 1.5 times; ranking quality unchecked. Recommended: keep `fp32`, run S4 (a), switch the default in a ticket of its own if the 10 passages keep their order.
3. **Precision on a CPU with AMX.** With the hint unset, OpenVINO picks `bf16` by itself there, so answers on such a Domino tier may differ in the last digits from those recorded on the emulation. Recommended: leave it unset; find out whether a Domino tier has AMX; if one has, run S4 (b) before comparing answer records across tiers.
4. **A smaller KV cache** (OVMS `u8`, vLLM `fp8`). Recommended: unset until an answers comparison (ROUGE-L, `qp-compare-answers`) says the text holds.
5. **vLLM's KV cache at 8 GiB by default**, the one default that is not the server's own. Recommended: yes; left out, vLLM sizes it from the host's memory, not the container's.
6. **vLLM on Domino 6.2** needs an environment on Ubuntu 24.04 and its model on a Dataset. Recommended: build the kind now; the owner provides the environment before S3 or a real deploy.
7. **The memory allocator** takes a library path and installs nothing. Recommended: keep it so; decide on adding `libjemalloc2` to the Domino environment after S4 (c).
8. **No NUMA variable.** Recommended: none while every tier is one NUMA node; `QP_OVMS_ARGS` and the App's own environment remain.
9. **Cores as CPUs divided by `threads_per_core` (2)**, a setting, rather than reading the machine's topology. Recommended: the setting; under a Kubernetes CPU quota the node's topology says nothing of the container's share.
10. **Uvicorn workers above 1 are possible** and divide the CPUs. Recommended: keep 1 on Domino, as FastAPI's docs advise for a cluster.
11. **`QP_UVICORN_LIMIT_CONCURRENCY` sheds load**, which `CLAUDE.md` says to add only when asked. It is in the owner's list, so it is planned, off. Recommended: keep it off; say so if it should be dropped instead.
12. **A load test straight at a vLLM server names a path** (`QP_LOADTEST_MODEL_SERVER_PATH`), where the branch named a kind: `loadtest` may import the domain only, and the kinds are registered in `deploy/`. Recommended: the path.
13. **No ONNX runtime.** Recommended: none.
14. **`docs/server-settings-for-throughput-and-latency.md` is not committed.** This plan cites it as its spec. Recommended: commit it with ticket 13.
15. **Keep-alive.** Uvicorn closes an idle connection after 5 s and httpx keeps one for 5 s; no source ties either to speed, and Domino's proxy sits between them. Recommended: keep both; read the transport retries in S1's logs before touching either.
16. **Flat `QP_VLLM_*`, one App one component**, where the branch had `RAG_VLLM_<COMPONENT>_*`. Recommended: flat, as `QP_OVMS_*`; the LLM's 4096 moves to the use case's deploy file.
17. **`DYNAMIC_QUANTIZATION_GROUP_SIZE` and `--grpc_workers` get no variable** (section 3). Recommended: accept; the first needs OVMS's LLM graph rewritten, the second a gRPC port.
