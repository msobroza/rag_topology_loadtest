"""The settings the components share: four top-level blocks of the workflow file.

A component's own settings sit in its entry
(``components/component_settings.py``). What several components share is
grouped by what it is about, each group a top-level key of the workflow file,
so each value has the variable its path names, such as
``QP_LOCAL_MODELS__CPUS`` or ``QP_HTTP__RETRIES``:

- ``local_models``: what the in-process models share, those of the components
  on the ``local`` backend: the runtime that runs them, where the ``openvino``
  runtime keeps their files and what it stores their weights as (the weight
  format, which OVMS's export reads too), and the workers and CPUs of the
  thread budget;
- ``calibration``: the budget and the rounds every calibration runs under, and
  the requests in flight a model server is calibrated for;
- ``http``: the default of each value a component's own ``http`` block leaves
  out, and the policy of the calls that are no component's;
- ``domino``: the Domino run token's URL and lifetime, and Domino tracing.

Domino's own run variables (``DOMINO_API_PROXY``, ...) are not settings of
ours: ``WorkflowSettings`` reads them under the names Domino gives them.

Example:
    >>> LocalModelsSettings(runtime="torch").workers
    4
"""

from __future__ import annotations

from pydantic import Field

from ..domain.runtime.backends import LocalModelRuntime, OpenVinoWeightFormat
from .components.component_settings import SettingsBlock


class LocalModelsSettings(SettingsBlock):
    """The ``local_models`` block: the runtime, its files and the thread budget.

    The weight format is read by both OpenVINO exports, so that servers are
    compared on the same weights: the in-process one and OVMS's
    (``ovms/export_models.sh``, through ``qp-served-models``).

    Example:
        >>> LocalModelsSettings().openvino_dir
        'models/openvino'
    """

    runtime: LocalModelRuntime = "openvino"  # openvino | torch
    # Where the openvino runtime keeps its models, one folder each. A model
    # without a folder there is downloaded and converted into it the first time
    # it loads; qp-export-openvino fills it ahead of time.
    openvino_dir: str = "models/openvino"
    # What an OpenVINO export stores its weights as (fp32 | int8). In-process
    # (local_models/openvino_export.py) an int8 model gets its own folder,
    # "<model>-int8"; OVMS's export reads it too. fp32 is what the in-process
    # models have always run; ranking quality on int8 is unchecked.
    # domino/model_servers/ovms.sh repeats this default in bash; a test keeps
    # the two equal.
    openvino_weight_format: OpenVinoWeightFormat = "fp32"
    # Model calls running at once, max, under torch only; openvino runs one call
    # at a time on every CPU (domain/runtime/thread_budget.py).
    workers: int = Field(default=4, ge=1)
    # Virtual CPUs the model calls share (domain/runtime/thread_budget.py): 0 = find
    # out; > 0 wins over every other count.
    cpus: float = Field(default=0, ge=0)


class CalibrationSettings(SettingsBlock):
    """The ``calibration`` block: what every calibration runs under.

    The budget and the rounds apply to an in-process model and to a model
    server alike; ``users`` only to a model server.

    Example:
        >>> CalibrationSettings().budget_s
        90.0
    """

    # Per CPU-bound stage (stage_calibration.py): no new candidate is started
    # once this many seconds have passed; a model server's trial still running
    # then is abandoned (served_stage_calibration.py).
    budget_s: float = Field(default=90.0, gt=0)
    # Timed requests per candidate, after one warm-up.
    rounds: int = Field(default=1, ge=1)
    # Requests in flight while a model server's candidate is timed: the load its
    # stream count is chosen for (one request is always fastest on one stream).
    # In-process stages are timed on one request: under openvino they run one
    # call at a time, so n requests take n times as long under any tuning.
    users: int = Field(default=4, ge=1)


class DefaultHttpSettings(SettingsBlock):
    """The top-level ``http`` block: the default of every call out.

    A component's own ``http`` block overrides each value for its calls
    (``components.<name>.http``); the calls that are no component's, such as
    the Domino hardware-tier lookup, take these as they are.

    Example:
        >>> DefaultHttpSettings().retries
        2
    """

    timeout_s: float = Field(default=30.0, gt=0)  # of one attempt
    retries: int = Field(default=2, ge=0)  # on transport errors, 5xx and 429
    backoff_s: float = Field(default=0.2, ge=0)  # before the 1st retry; doubles


class DominoSettings(SettingsBlock):
    """The ``domino`` block: the run token's URL and lifetime, and tracing.

    Example:
        >>> DominoSettings().token_ttl_s
        240.0
    """

    # Where a Domino run serves its token, for a served component under
    # qp-serve --domino and for an api key set to "domino".
    token_url: str = "http://localhost:8899/access-token"
    token_ttl_s: float = Field(default=240.0, gt=0)  # tokens live ~5 min
    tracing: bool = False  # wrap the workflow run with Domino's add_tracing
