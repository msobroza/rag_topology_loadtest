"""Tests of the two ways a block of the deploy file writes one setting.

The hardware tier by its name (``hardware_tier``) or by its id
(``hardware_tier_id``), the environment revision by its number
(``environment_revision``) or by its id (``environment_revision_id``): one way
a block, the App's own winning over the default whichever way each is written.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from domino_deploy_helpers import LAUNCH, make_client, make_settings, no_sleep
from fake_domino.state import FakeDominoApps
from query_processing.deploy.apps_client import DominoDeployError
from query_processing.deploy.deploy_refusals import refuse_placeholders
from query_processing.deploy.deploy_settings import DeploySettings
from query_processing.deploy.publish import deploy

TIER_WAYS = [
    ({"hardware_tier": "cpu-16"}, ("cpu-16", "")),
    ({"hardware_tier_id": "large-k8s"}, ("", "large-k8s")),
]
REVISION_WAYS = [
    ({"environment_revision": 7}, (7, "")),
    ({"environment_revision_id": "rev-7"}, (None, "rev-7")),
]
BLOCKS = ["defaults", "apps.reranker"]


def tier_of(settings: DeploySettings, part: str) -> tuple[str, str]:
    app = settings.for_app(part)
    return app.hardware_tier, app.hardware_tier_id


def revision_of(settings: DeploySettings, part: str) -> tuple[int | None, str]:
    app = settings.for_app(part)
    return app.environment_revision, app.environment_revision_id


def settings_with(block: str, values: dict[str, object]) -> DeploySettings:
    """Settings whose ``block`` (``defaults`` or ``apps.reranker``) holds ``values``."""
    if block == "defaults":
        return make_settings(defaults={"environment_id": "env-1", **values})
    return make_settings(apps={"reranker": values})


@pytest.mark.parametrize(("own", "expected"), TIER_WAYS)
@pytest.mark.parametrize(
    "default", [{"hardware_tier": "cpu-8"}, {"hardware_tier_id": "small-k8s"}]
)
def test_an_apps_own_tier_replaces_the_default_whichever_way_each_is_written(
    own: dict[str, str], expected: tuple[str, str], default: dict[str, str]
) -> None:
    settings = make_settings(
        defaults={"environment_id": "env-1", **default}, apps={"reranker": own}
    )
    assert tier_of(settings, "reranker") == expected
    assert tier_of(settings, "embedder") == (
        default.get("hardware_tier", ""),
        default.get("hardware_tier_id", ""),
    )


@pytest.mark.parametrize(("own", "expected"), REVISION_WAYS)
def test_the_default_revision_does_not_follow_an_app_to_its_own_environment(
    own: dict[str, object], expected: tuple[int | None, str]
) -> None:
    settings = make_settings(
        defaults={"environment_id": "env-1", **own},
        apps={"reranker": {"environment_id": "env-2"}},
    )
    assert revision_of(settings, "embedder") == expected
    assert revision_of(settings, "reranker") == (None, "")


@pytest.mark.parametrize(("own", "expected"), REVISION_WAYS)
@pytest.mark.parametrize(("default", "_"), REVISION_WAYS)
def test_an_apps_own_revision_replaces_the_default_either_way(
    own: dict[str, object],
    expected: tuple[int | None, str],
    default: dict[str, object],
    _: tuple[int | None, str],
) -> None:
    settings = make_settings(
        defaults={"environment_id": "env-1", **default}, apps={"reranker": own}
    )
    assert revision_of(settings, "reranker") == expected


@pytest.mark.parametrize("block", BLOCKS)
def test_both_keys_of_the_tier_in_one_block_are_refused(block: str) -> None:
    with pytest.raises(ValidationError) as caught:
        settings_with(
            block, {"hardware_tier": "cpu-16", "hardware_tier_id": "large-k8s"}
        )
    assert block in str(caught.value)
    assert (
        "hardware_tier 'cpu-16' and hardware_tier_id 'large-k8s' are both set: they "
        "are two ways to write one setting, keep one"
    ) in str(caught.value)


@pytest.mark.parametrize("block", BLOCKS)
def test_both_keys_of_the_revision_in_one_block_are_refused(block: str) -> None:
    values = {"environment_revision": 7, "environment_revision_id": "rev-7"}
    with pytest.raises(ValidationError) as caught:
        settings_with(block, values)
    assert block in str(caught.value)
    assert (
        "environment_revision 7 and environment_revision_id 'rev-7' are both set: "
        "they are two ways to write one setting, keep one"
    ) in str(caught.value)


def test_an_empty_id_stands_beside_the_other_key() -> None:
    settings = make_settings(
        defaults={
            "environment_id": "env-1",
            "hardware_tier_id": "",
            "hardware_tier": "cpu-8",
            "environment_revision_id": "",
            "environment_revision": 7,
        }
    )
    assert tier_of(settings, "embedder") == ("cpu-8", "")
    assert revision_of(settings, "embedder") == (7, "")


def test_an_apps_empty_tier_does_not_replace_the_default() -> None:
    settings = make_settings(
        defaults={"environment_id": "env-1", "hardware_tier": "cpu-8"},
        apps={"reranker": {"hardware_tier": "", "hardware_tier_id": ""}},
    )
    assert tier_of(settings, "reranker") == ("cpu-8", "")


@pytest.mark.parametrize("revision", [0, -1, 7.5, "7", "v7", True])
def test_a_revision_that_is_no_whole_number_from_one_is_refused(
    revision: object,
) -> None:
    with pytest.raises(ValidationError) as caught:
        make_settings(
            defaults={"environment_id": "env-1", "environment_revision": revision}
        )
    assert "defaults.environment_revision" in str(caught.value)
    assert f"input_value={revision!r}" in str(caught.value)
    # The key is known and its value refused, not the key taken for a misspelt one.
    assert "Extra inputs" not in str(caught.value)


def test_a_placeholder_left_in_the_tier_name_is_refused() -> None:
    settings = make_settings(
        apps={"reranker": {"hardware_tier": "<a tier with 16 cores>"}}
    )
    with pytest.raises(DominoDeployError) as caught:
        refuse_placeholders(settings, ["reranker"])
    assert str(caught.value) == (
        "hardware_tier in apps.reranker of deploy.yaml still holds the example "
        "placeholder '<a tier with 16 cores>': replace it with a real value"
    )


def test_a_default_revision_number_without_a_default_environment_is_refused() -> None:
    apps = FakeDominoApps(["Running"])
    settings = make_settings(defaults={"environment_revision": 7})
    with pytest.raises(DominoDeployError) as caught:
        deploy(make_client(apps), settings, "all", launch=LAUNCH, sleep=no_sleep)
    assert str(caught.value).startswith(
        "defaults.environment_revision 7 in deploy.yaml has no "
        "defaults.environment_id beside it"
    )
    assert apps.received == []
