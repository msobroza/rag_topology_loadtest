"""A new model-server kind is one registered class and one launch script.

The package is copied with one module added to ``deploy/model_server_kinds/``,
and ``domino/`` with one script added to ``domino/model_servers/``; nothing else
of either copy is touched, the deploy tool's flow and app.sh's dispatch
included. The new kind serves rerankers only, on a route of its own, and needs
one variable of its own. A driver run against the copy writes the entry scripts
of a split-all whose reranker the new kind serves, and asks for the embedder to
be served by it too; the reranker's script is then run through the copy's
app.sh.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from launch_script_stand_ins import log, make_stand_ins, stand_in_env
from rag_graphs import FIXTURE_WORKFLOW
from repo_paths import TOOLKIT_ROOT

# Dropped into deploy/model_server_kinds/ of a copy of the package: the only change.
RERANK_ONLY_KIND = '''"""A model-server kind of the tests: rerankers only, on a route of its own."""

from __future__ import annotations

from collections.abc import Mapping

from .base_model_server_kind import MODEL_SERVER_KINDS, BaseModelServerKind


@MODEL_SERVER_KINDS.register("rerank-only")
class RerankOnlyKind(BaseModelServerKind):
    @property
    def routes(self) -> Mapping[str, str]:
        return {"reranker": "/v1/rerank"}

    @property
    def probe_route(self) -> str:
        return "/v1/models/{name}"

    def launch_exports(self, settings) -> dict[str, str]:
        return {"QP_RERANK_ONLY_THREADS": "4"}
'''
# Dropped into domino/model_servers/ of a copy of domino/: its launch script.
RERANK_ONLY_SCRIPT = """# shellcheck shell=bash
bootstrap_model_server() {
  echo "bootstrap" >> "$HOME/rerank-only.log"
}

serve_model_server() {
  echo "serve $1 threads=$QP_RERANK_ONLY_THREADS" >> "$HOME/rerank-only.log"
}
"""
# Writes the copy's entry scripts and its refusal, as JSON.
DRIVER = """
import json
import sys

import query_processing
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.publish import LaunchPaths, entry_script
from query_processing.deploy.topology_alias import resolve_topology

URLS = {"embedder": "https://domino.test/e1", "reranker": "https://domino.test/r1"}


def settings(apps):
    return DeploySettings(
        app_name_prefix="rag",
        topology="split-all",
        topology_alias={"split-all": {"embedder": "served", "reranker": "served"}},
        defaults={"environment_id": "env-1"},
        apps=apps,
    )


chosen = settings({"reranker": {"server": "rerank-only"}})
topology = resolve_topology(chosen)
scripts = {
    part: entry_script(
        chosen.for_app(part), chosen, topology,
        URLS if part == "workflow" else {}, LaunchPaths(sys.argv[1], sys.argv[2]),
    )
    for part in topology.app_parts()
}
try:
    resolve_topology(settings({"embedder": {"server": "rerank-only"}}))
    refusal = ""
except DominoDeployError as exc:
    refusal = str(exc)
print(json.dumps(
    {"package": query_processing.__file__, "scripts": scripts, "refusal": refusal}
))
"""


def copies_with_a_new_kind(root: Path) -> tuple[Path, Path]:
    """Copy the package and domino/ under ``root``, each with the new kind added."""
    package = root / "src" / "query_processing"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(TOOLKIT_ROOT / "src" / "query_processing", package, ignore=ignore)
    kinds = package / "deploy" / "model_server_kinds"
    (kinds / "rerank_only_kind.py").write_text(RERANK_ONLY_KIND, encoding="utf-8")
    shutil.copytree(TOOLKIT_ROOT / "domino", root / "domino")
    script = root / "domino" / "model_servers" / "rerank-only.sh"
    script.write_text(RERANK_ONLY_SCRIPT, encoding="utf-8")
    return package, root / "domino" / "app.sh"


def run_driver(root: Path, app_sh: Path) -> dict[str, object]:
    """Run the driver in ``root``, a use case's directory holding the tests' file."""
    (root / "driver.py").write_text(DRIVER, encoding="utf-8")
    shutil.copy(FIXTURE_WORKFLOW, root / "workflow.yaml")
    # No QP_* variable of this shell, and the copy ahead of any installed package.
    env = {k: v for k, v in os.environ.items() if not k.startswith("QP_")}
    env["PYTHONPATH"] = str(root / "src")
    run = subprocess.run(
        [sys.executable, str(root / "driver.py"), str(app_sh), str(root)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=root,
        env=env,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout.splitlines()[-1])


def test_a_new_kind_is_one_class_and_one_script(tmp_path: Path) -> None:
    root = tmp_path / "copy"
    package, app_sh = copies_with_a_new_kind(root)

    written = run_driver(root, app_sh)

    assert Path(str(written["package"])).parent == package  # the copy
    scripts = written["scripts"]
    assert isinstance(scripts, dict)
    # The workflow App calls the reranker on the new kind's own route.
    workflow = scripts["workflow"]
    assert (
        "export QP_COMPONENTS__EMBEDDER__SERVED__URL="
        "https://domino.test/e1/v3/embeddings\n"
    ) in workflow
    assert (
        "export QP_COMPONENTS__RERANKER__SERVED__URL=https://domino.test/r1/v1/rerank\n"
    ) in workflow
    label = "export QP_RUN_LABEL=split-all+embedder=fastapi+reranker=rerank-only\n"
    assert label in workflow
    # app.sh starts the new kind's script, with the variable the class adds.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    env = stand_in_env(run_dir, make_stand_ins(run_dir))
    started = subprocess.run(
        ["bash", "-c", scripts["reranker"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert started.returncode == 0, started.stderr
    assert (run_dir / "home" / "rerank-only.log").read_text().splitlines() == [
        "bootstrap",
        "serve reranker threads=4",
    ]
    assert log(run_dir, "qp-model-server") == ""


def test_a_kind_that_cannot_serve_a_type_is_refused_naming_the_kinds_that_can(
    tmp_path: Path,
) -> None:
    root = tmp_path / "copy"
    written = run_driver(root, copies_with_a_new_kind(root)[1])

    assert written["refusal"] == (
        "topology alias 'split-all' serves component 'embedder', of type "
        "'embedder', with the model-server kind 'rerank-only' (apps.embedder.server, "
        "else defaults.server, in deploy.yaml), which cannot serve it: the kinds "
        "that can are 'fastapi', 'ovms', 'vllm'"
    )
