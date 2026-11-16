# A generator on 16 vCPUs: OVMS, PyTorch and vLLM on Qwen3.5 with 2 B parameters

Sitting of 16 November 2026, 06:26 to 08:25 UTC. Registered on 16 November 2026 from
the logs and the statistics files of the session that ran it.

This sitting is a check made before a generator server is built into the product. It
ran on one machine, by hand, outside the deploy tool and the emulated Domino. Read its
figures as a first look: "Limits" says why.

## Question

Which server answers fastest when the generation stage runs a language model on a CPU:
OVMS, PyTorch behind a FastAPI server, or vLLM? And what in each decides its speed?

The model is Qwen3.5 with 2 B parameters, with 4-bit weights. The embedder, the
reranker and the vector store are the fake ones, so the generator is the only real
model on the path of `/query`.

## Hardware

| Role | Machine type | vCPUs | Count |
|---|---|---|---|
| Generator, workflow app and Locust, on demand | `n2-standard-16` (8 cores, 62 GB) | 16 | 1 |

An Intel Xeon 2.80 GHz with AVX-512 VNNI, Ubuntu 22.04. No Workspace, no emulated
Domino, no proxy, no Elasticsearch.

## Settings common to the sitting

- Code: `main` at `2209c12`. The copy of the workflow app installed on the machine was
  patched to send `max_tokens` with every chat request and to note the token counts of
  every answer. Neither is in the product yet.
- Models: `OpenVINO/Qwen3.5-2B-int4-ov` for OVMS (OpenVINO's conversion, 4-bit
  asymmetric weights); `Intel/Qwen3.5-2B-int4-AutoRound` for PyTorch and vLLM (4-bit
  symmetric weights). Both have a group size of 128. They are two 4-bit versions of one
  model, not one file.
- Servers: OVMS 2026.4.1; PyTorch 2.15.0 with transformers 5.23.0 and auto-round 0.16.1;
  vLLM 0.32.0, its CPU image.
- The workflow app: `monolith`, the fake embedder, reranker and vector store,
  generation on the `api` backend at the server under test. Every answer is capped at
  128 tokens, at temperature 0.
- The load: Locust on `/query`, 1 user then 4, 3 minutes a run. The load test's 50
  synthetic questions are 14 distinct ones, and with the fake store each retrieves the
  same 10 passages every time: a run sends 14 distinct prompts of 747 to 922 tokens.
- Run names: `g16-check-<server>-u<users>`. An id that starts with `g16-probe-` is a
  measurement that is not a Locust run: requests sent straight to a server by a script.

## Experiments

### Locust runs

The route's row, then the tokens. No request failed in any run.

| Id | Server | Users | Requests/s | p50 | p95 | Tokens per answer | Tokens served per second |
|---|---|---|---|---|---|---|---|
| `g16-check-ovms` | OVMS as it ships | 1 | 0.124 | 7.9 s | 8.2 s | 114 | 14.1 |
| | | 4 | 0.092 | 42 s | 49 s | 122 | 11.6 |
| `g16-check-ovms-tuned` | OVMS reading each prompt in one pass | 1 | 0.124 | 7.9 s | 8.5 s | 118 | 14.4 |
| | | 4 | 0.170 | 23 s | 31 s | 118 | 20.4 |
| `g16-check-torch` | PyTorch behind a stand-in FastAPI server, one generation at a time | 1 | 0.076 | 13 s | 13 s | 128 | 9.3 |
| | | 4 | 0.075 | 51 s | 56 s | 128 | 10.0 |
| `g16-check-torch-first-script` | The stand-in's first version | 1 | 0.075 | 13 s | 13 s | 128 | 9.3 |
| | | 4 | 0.076 | 50 s | 58 s | 128 | 10.0 |
| `g16-check-vllm` | vLLM's CPU image | 1 | 0.061 | 19 s | 20 s | 113 | 7.0 |
| | | 4 | 0.148 | 25 s | 50 s | 105 | 15.8 |

### Measurements that are not runs

| Id | What was measured | Headline |
|---|---|---|
| `g16-probe-ovms-first-requests` | OVMS's first requests, its prefix cache on then off | Pulls and serves in 60 s; 8.3 s a request with the cache, 8.0 s without |
| `g16-probe-ovms-settings` | Four OVMS settings: reading, one request, four at once | Reading stays at 4.6 to 4.9 s; four at once take 22 s in one pass against 33 s as shipped |
| `g16-probe-pytorch-threads` | PyTorch called from a script, on 16, 8 and 4 threads | 9.6 s a request on 8 threads, 11.4 s on 16; reading takes 1.1 s |
| `g16-probe-pytorch-in-the-project-environment` | Whether PyTorch loads the model in the project's virtualenv | It does not, or loads it wrongly |
| `g16-probe-vllm` | vLLM's start and its first requests | 17 minutes to start; 20 s a request; 20 GB of memory |

Settings in full: [experiments.yaml](experiments.yaml). Every row of every run:
[results.csv](results.csv).

## What it showed

- **One user.** OVMS answered in 7.9 s, the PyTorch stand-in in 13 s, vLLM in 19 s
  (p50). OVMS served 14 tokens a second, the stand-in 9, vLLM 7.
- **Four users, and one OVMS setting.** OVMS as it ships served less at 4 users than at
  1: 0.092 requests a second against 0.124, with a p50 of 42 s. Told to read each
  prompt in one pass (`--dynamic_split_fuse false --max_num_batched_tokens 8192`), it
  served 0.170 requests a second at 4 users with a p50 of 23 s, the best of the
  sitting, and nothing changed at 1 user.
- **Batching is what gains under load.** From 1 user to 4, tokens served a second went
  from 14.4 to 20.4 on OVMS in one pass and from 7.0 to 15.8 on vLLM, which both run
  several requests in one batch. The stand-in runs one generation at a time and stayed
  at 9 to 10.
- **Reading against writing.** A request reads its prompt, then writes its answer.
  PyTorch read a 781-token prompt in 1.1 s, OVMS in 3.9 s, vLLM in 7.9 s. For a
  922-token prompt and 128 tokens written, OVMS took 8.6 s, PyTorch 9.6 s (called from a
  script), vLLM 19.9 s. So OVMS writes about twice as fast as PyTorch and reads more
  than three times slower. The prompt of a RAG request is long and its answer short,
  which is why the two end close.
- **No OVMS setting changed its reading.** It stayed between 4.6 and 4.9 s for 922
  tokens under the four settings tried. The model's multi-token prediction did not
  help: 8.9 s a request, and 38 s for four at once.
- **OVMS's prefix cache changed nothing** for this model: a prompt sent again took
  8.3 s with the cache and 8.0 s without. OVMS serves this model with its
  vision-language pipeline.
- **The answers are not the same length.** Capped at 128 tokens, the PyTorch
  stand-in's answers all filled the cap; OVMS's held 114 to 122 tokens on average and
  vLLM's 105 to 113. Requests a second alone would favour the server that writes less.
- **Threads.** PyTorch was fastest on 8 threads, the machine's cores: 9.6 s, against
  11.4 s on 16 and 12.0 s on 4.
- **4-bit weights are worth 2.6 times under PyTorch.** The original model in floating
  point took 24.6 s for the request the 4-bit one answered in 9.6 s.
- **PyTorch cannot share the project's virtualenv for this model.** As installed, the
  project resolves transformers 5.0.0, which does not know the architecture. Asked for
  5.2 or later it gets 5.5.4, the newest that `optimum-intel` accepts, and Intel's model
  then loads with no 4-bit layer and writes unreadable text, with no error. In a
  virtualenv without the project, on transformers 5.23.0, it loads and answers.
- **vLLM starts slowly and holds much more memory.** It compiles the model when it
  starts: 17 minutes the first time, and a second start was still compiling after 19
  minutes. It held 20 GB, against 3.2 GB for OVMS and 3.7 GB for PyTorch.
- **OVMS 2026.4.1 serves the model unaided.** It pulled the files and served in 60 s,
  capped an answer when asked, reported the tokens of each answer, and answered 404 for
  an unknown model with the message the project's own model server copies.

## Limits

- One machine held the generator, the workflow app and Locust. The last two use little
  CPU, but in a sitting on the emulation they have machines of their own.
- One 3-minute run per point, of 11 to 30 requests. A measurement that is not a run is
  one request, after one warm-up.
- The PyTorch server is a stand-in of about 60 lines, not the product's model server.
  It took 13 s a request where the same model called from a script took 9.6 s. Setting
  the thread count in the generating thread (`g16-check-torch`) changed nothing, and
  the cause was not established. A PyTorch server without that loss would serve about
  0.10 requests a second at 1 user, still below OVMS.
- The two 4-bit versions of the model are different files, made by different tools.
  Speeds compare; the answers are not the same text.
- On the same weights as the stand-in, vLLM's answers stop sooner on some questions.
  Why was not established.
- The 14 prompts repeat during a run. That would favour a cache of prompts already
  read; OVMS's changed nothing here, and no other server had one turned on.
- Answer quality was not looked at.
- vLLM ran with its defaults. Nothing was tried to shorten its start or its reading.
- Nothing went through the deploy tool, the emulated Domino or its proxy.
