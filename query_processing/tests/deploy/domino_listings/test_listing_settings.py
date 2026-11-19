"""Tests of what a listing reads of deploy.yaml: how to reach Domino, nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest

from query_processing.deploy.domino_listings.listing_settings import ListingSettings

# Keys an App needs, left invalid: a listing reads none of them.
NO_VALID_APP_BLOCK = (
    'apps: {nothing: {valid: here}}\ndefaults: {environment_id: "<environment id>"}\n'
)


@pytest.mark.parametrize("written_in", ["the file", "a variable"])
def test_how_to_reach_domino_is_read_from_the_file_or_a_variable_alike(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, written_in: str
) -> None:
    deploy_yaml = tmp_path / "deploy.yaml"
    reach = "http_timeout_s: 5\nhttp_retries: 4\n"
    deploy_yaml.write_text(
        NO_VALID_APP_BLOCK + (reach if written_in == "the file" else "")
    )
    if written_in == "a variable":
        monkeypatch.setenv("QP_DEPLOY_HTTP_TIMEOUT_S", "5")
        monkeypatch.setenv("QP_DEPLOY_HTTP_RETRIES", "4")
    monkeypatch.setenv("QP_DEPLOY_FILE", str(deploy_yaml))
    monkeypatch.setenv("DOMINO_PROJECT_ID", "proj-1")
    settings = ListingSettings()
    assert (settings.http_timeout_s, settings.http_retries) == (5.0, 4)
    assert settings.project_id == "proj-1"
