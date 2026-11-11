"""The start script a placed App runs: Domino's variables, the project's, then its own."""

from __future__ import annotations

import subprocess

from fake_domino.emulation.start_script import DominoRunVariables, compose_start_script

RUN = DominoRunVariables(
    hardware_tier_id="cpu-8",
    project_id="proj-1",
    app_id="app-1",
    app_version_id="app-1-v2",
    run_host_path="/fake-owner/fake-project/r/notebookSession/app-1-v2-i/",
)


def test_the_script_exports_in_order_then_runs_the_entry_script_unchanged() -> None:
    entry = "export RAG_ROLE=workflow\nexec bash domino/app.sh"
    script = compose_start_script(
        RUN, "http://localhost:8899", {"RAG_ES_URL": "http://es:9200"}, entry
    )
    assert script == (
        "export DOMINO_API_PROXY=http://localhost:8899\n"
        "export DOMINO_HARDWARE_TIER_ID=cpu-8\n"
        "export DOMINO_PROJECT_ID=proj-1\n"
        "export DOMINO_APP_ID=app-1\n"
        "export DOMINO_APP_VERSION_ID=app-1-v2\n"
        "export DOMINO_RUN_HOST_PATH=/fake-owner/fake-project/r/notebookSession/app-1-v2-i/\n"
        "export RAG_ES_URL=http://es:9200\n"
        "export RAG_ROLE=workflow\n"
        "exec bash domino/app.sh\n"
    )


def test_values_are_shell_quoted_and_the_entry_scripts_exports_win() -> None:
    awkward = 'it\'s $HOME `x` "q" ; rm -rf /'
    project_env = {"SECRET": awkward, "RAG_TOPOLOGY": "monolith"}
    entry = 'export RAG_TOPOLOGY=split-all\nprintf "%s\\n%s" "$SECRET" "$RAG_TOPOLOGY"'
    script = compose_start_script(RUN, "http://localhost:8899", project_env, entry)
    ran = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    assert ran.stdout == f"{awkward}\nsplit-all"
