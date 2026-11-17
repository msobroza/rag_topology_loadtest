# A served LLM is asked for as any served component is, by a topology alias's row and its App's kind

ADR 0003 refused a topology alias that serves an LLM, since no model-server kind could serve one. The `ovms` kind now serves an LLM on OpenVINO Model Server's chat completions route. It is asked for as any served component is: a row of a topology alias says `llm: served`, and the LLM's App block in the deploy settings says `server: ovms`. Nothing else names it: no key of its own and no Role of its own. The model OVMS serves is the LLM component's `source_id`, which OVMS pulls on the App's first start, served under the component's `alias`, which the workflow App asks for. Both are set in one place, the LLM's block of the workflow file or of the deploy settings' `overrides`, and every App reads them from there.

## Considered options

- **A key of its own in the deploy settings**, `generator_server` before the toolkit, with a Role of its own and a make target. Rejected: it would be a second way to say where a component runs, beside the topology alias, and a second way to publish an App, beside the `model-server` Role told its component and its kind. The order of Apps, the URL handed to the workflow App, the run label and the Domino run's token already come with a served component.
- **The model and its served name in the LLM App's own variables**, as the repository read them before the toolkit. Rejected: the workflow App and the LLM's App must send and serve the same name, and a model is named in one place. On the component, both Apps read it from the same block.
- **The LLM on the `fastapi` kind.** Rejected: the FastAPI model server serves no LLM, and a sitting by hand on 16 November 2026 answered one user soonest on OVMS.

## Consequences

- ADR 0003's last consequence becomes narrower: a topology alias that puts an LLM on a kind that cannot serve it, `fastapi`, is refused before anything is published, naming `ovms`.
- `qp-deploy` publishes the LLM's App (`<app_name_prefix>-llm`) after the other model servers and before the workflow App, hands the workflow App its URL plus `/v3/chat/completions`, and adds `+llm=ovms` to the run label.
- An LLM's App runs on the generic environment with `QP_BOOTSTRAP=1` and the OVMS bundle, not on the prebuilt `ovms` image, which holds neither `uv` nor the toolkit to read the component's `source_id` and `alias`.
- OVMS serves an LLM as it pulls it: nothing is exported or calibrated, and a pinned stream or thread count on an LLM's App is refused. Its two batching options, `QP_OVMS_DYNAMIC_SPLIT_FUSE` and `QP_OVMS_MAX_BATCHED_TOKENS`, are that App's settings, and are refused on an embedder's or a reranker's App.
- The workflow App's `/readyz` does not check a served LLM, so it can report ready while OVMS still pulls the model.
- Not verified: an LLM's App on OVMS has not run on a machine, with the toolkit or before it. What has run is the bare `ovms` command, by hand, on 16 November 2026. Whether OVMS honours `response_format`, which the RAG's rewrite sends its schema by, is unknown.
