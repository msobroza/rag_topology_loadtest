"""End-to-end tests of ``qp-deploy tiers`` and ``qp-deploy environments``.

``main`` over the in-process fake Domino, which has three tiers (cpu-8 the
default) and one environment, qp-rag, with the revisions 7 (latest), 6 (active)
and 5. Under pytest the output is no terminal, so the table is the default look.
The fixtures (run_main) are in domino_deploy_fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from domino_deploy_helpers import RunMain
from fake_domino.server import create_fake_domino
from fake_domino.stand_ins import fake_environment, fake_tier
from fake_domino.state import FakeDominoApps

pytest_plugins = ["domino_deploy_fixtures"]  # domino_env, no_real_http, run_main

BUILT_7 = "2026-11-12T09:41:00Z"
BUILT_6 = "2026-11-03T16:02:00Z"
TIERS = [
    fake_tier("small-k8s", "cpu-8", 8, 32, is_default=True),
    fake_tier("large-k8s", "cpu-16", 16, 64),
    fake_tier("gpu-k8s", "gpu-1", 8, 32, gpus=1),
]
# Not ENV_1: the helpers' ENV_1 is the text of a deploy file, which run_main writes.
FAKE_ENV_1 = fake_environment(
    "env-1",
    (7, "rev-7", BUILT_7),
    (6, "rev-6", BUILT_6),
    (5, "rev-5", BUILT_6),
    name="qp-rag",
    active=6,
)
INTERNAL_CALL = ("GET", "/v4/environments/env-1/paginatedRevisionData")
TIERS_TABLE = (
    "NAME    CORES  MEMORY  GPUS  DEFAULT  ID\n"
    "cpu-8       8  32 GiB     -  yes      small-k8s\n"
    "cpu-16     16  64 GiB     -           large-k8s\n"
    "gpu-1       8  32 GiB     1           gpu-k8s\n"
)


def fake_apps() -> FakeDominoApps:
    return FakeDominoApps(tiers=TIERS, environments=[FAKE_ENV_1])


def internal_calls(apps: FakeDominoApps) -> list[tuple[str, str]]:
    return [call for call in apps.received if call[1].endswith("RevisionData")]


def test_tiers_prints_its_table_where_there_is_no_terminal(run_main: RunMain) -> None:
    assert run_main(["tiers"], create_fake_domino(fake_apps())) == (0, TIERS_TABLE, "")


def test_tiers_named_with_details_prints_yaml_of_that_tier_alone(
    run_main: RunMain,
) -> None:
    code, out, _ = run_main(
        ["tiers", "cpu-16", "--details"], create_fake_domino(fake_apps())
    )
    [shown] = yaml.safe_load(out)
    assert (code, shown["id"], shown["resources"]["cores"]) == (0, "large-k8s", 16.0)
    assert next(iter(shown)) == "name"


def test_environments_prints_its_table(run_main: RunMain) -> None:
    apps = fake_apps()
    assert run_main(["environments"], create_fake_domino(apps)) == (
        0,
        "NAME    VISIBILITY  OWNER       ACTIVE  LATEST  ID\n"
        "qp-rag  private     fake-owner       6       7  env-1\n",
        "",
    )
    assert INTERNAL_CALL not in apps.received


def test_a_named_environment_prints_its_revisions(run_main: RunMain) -> None:
    apps = fake_apps()
    code, out, _ = run_main(
        ["environments", "qp-rag", "--table"], create_fake_domino(apps)
    )
    assert (code, out) == (
        0,
        "qp-rag   id env-1   private, owner fake-owner\n"
        "\n"
        "REVISION  BUILT (UTC)       STATUS     ID\n"
        "7 latest  2026-11-12 09:41  succeeded  rev-7\n"
        "6 active  2026-11-03 16:02  succeeded  rev-6\n"
        "5         2026-11-03 16:02             rev-5\n",
    )
    assert internal_calls(apps) == [INTERNAL_CALL]


def test_a_named_environment_with_details_holds_its_revisions(
    run_main: RunMain,
) -> None:
    apps = fake_apps()
    code, out, _ = run_main(
        ["environments", "qp-rag", "--details"], create_fake_domino(apps)
    )
    [shown] = yaml.safe_load(out)
    assert code == 0
    assert [revision["number"] for revision in shown["revisions"]] == [7, 6, 5]
    assert internal_calls(apps) == [INTERNAL_CALL]


def test_details_alone_never_call_the_internal_route(run_main: RunMain) -> None:
    apps = fake_apps()
    code, out, _ = run_main(["environments", "--details"], create_fake_domino(apps))
    [shown] = yaml.safe_load(out)
    assert code == 0
    assert "latestRevision" in shown
    assert "revisions" not in shown
    assert INTERNAL_CALL not in apps.received


def test_an_unknown_name_is_refused_with_the_names_domino_has(
    run_main: RunMain,
) -> None:
    code, _, err = run_main(["tiers", "cpu-61"], create_fake_domino(fake_apps()))
    assert code == 2
    assert err.endswith(
        "qp-deploy: error: no hardware tier of Domino is named 'cpu-61'; it has: "
        "cpu-8, cpu-16, gpu-1\n"
    )


def test_the_two_options_exclude_each_other(run_main: RunMain) -> None:
    code, _, err = run_main(
        ["tiers", "--table", "--details"], create_fake_domino(fake_apps())
    )
    assert code == 2
    assert "not allowed with argument" in err


@pytest.mark.parametrize("command", ["tiers", "environments"])
def test_a_listing_is_refused_in_a_dry_run(run_main: RunMain, command: str) -> None:
    code, _, err = run_main(["--dry-run", command], None)
    assert code == 2
    assert err.endswith(
        f"qp-deploy: error: qp-deploy {command} asks Domino what it has, and a dry "
        "run asks Domino nothing: run it without --dry-run\n"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["tiers", "--table"],
        ["tiers", "cpu-8", "--details"],
        ["environments", "--table"],
        ["environments", "--details"],
        ["environments", "qp-rag", "--table"],
        ["environments", "qp-rag", "--details"],
    ],
)
def test_a_listing_sends_nothing_but_get(run_main: RunMain, argv: list[str]) -> None:
    apps = fake_apps()
    assert run_main(argv, create_fake_domino(apps))[0] == 0
    assert apps.received
    assert {method for method, _ in apps.received} == {"GET"}


def test_a_listing_needs_no_valid_app_block(run_main: RunMain) -> None:
    Path("deploy.yaml").write_text('defaults: {environment_id: "<environment id>"}\n')
    code, out, _ = run_main(["tiers"], create_fake_domino(fake_apps()))
    assert (code, out) == (0, TIERS_TABLE)
