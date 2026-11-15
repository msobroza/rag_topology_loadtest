"""The placement file Terraform writes, and the project env file: parsing and refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fake_domino.emulation.placement_settings import (
    HardwareTierPool,
    PlacementSettings,
    SshSettings,
    load_placement,
)
from fake_domino.emulation.project_env import load_project_env

DOCUMENTED = {
    "ssh": {"user": "ubuntu", "key": "/home/ubuntu/.ssh/id_emulation"},
    "app_port": 8888,
    "tiers": [
        {
            "id": "cpu-8",
            "cores": 8,
            "memory_gib": 32,
            "default": True,
            "machines": ["10.10.0.21", "10.10.0.22"],
        },
        {"id": "cpu-16", "cores": 16, "memory_gib": 64, "machines": ["10.10.0.31"]},
    ],
}


def written(tmp_path: Path, content: object) -> Path:
    path = tmp_path / "placement.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def test_the_documented_shape_loads_with_the_defaults(tmp_path: Path) -> None:
    settings = load_placement(written(tmp_path, DOCUMENTED))
    assert settings == PlacementSettings(
        ssh=SshSettings("ubuntu", "/home/ubuntu/.ssh/id_emulation", 5.0, 20.0),
        app_port=8888,
        tiers=(
            HardwareTierPool("cpu-8", 8, 32, ("10.10.0.21", "10.10.0.22"), True),
            HardwareTierPool("cpu-16", 16, 64, ("10.10.0.31",), False),
        ),
        unit="rag-app",
        stop_timeout_seconds=10.0,
        probe_timeout_seconds=1.0,
        api_proxy_url="http://localhost:8899",
    )


def test_tiers_are_found_by_id_and_an_empty_id_is_the_default(tmp_path: Path) -> None:
    settings = load_placement(written(tmp_path, DOCUMENTED))
    assert settings.default_tier.id == "cpu-8"
    assert [settings.tier(i) for i in ("", None, "cpu-16", "gpu")] == [
        settings.tiers[0],
        settings.tiers[0],
        settings.tiers[1],
        None,
    ]


def tiers_with(**changes: object) -> dict[str, object]:
    first = {**DOCUMENTED["tiers"][0], **changes}  # type: ignore[index, dict-item]
    return {**DOCUMENTED, "tiers": [first, DOCUMENTED["tiers"][1]]}  # type: ignore[index]


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("{not json", "Invalid JSON"),
        ({**DOCUMENTED, "app_prot": 1}, "app_prot: Unexpected keyword argument"),
        ({k: v for k, v in DOCUMENTED.items() if k != "ssh"}, "ssh: Field required"),
        (
            {**DOCUMENTED, "ssh": {"user": "ubuntu"}},
            "ssh.key: Field required",
        ),
        (tiers_with(cores="8"), "tiers.0.cores: Input should be a valid number"),
        (tiers_with(machines="10.10.0.21"), "tiers.0.machines: Input should be"),
        ({**DOCUMENTED, "app_port": 70000}, "app_port: 70000 is not a TCP port"),
        (
            tiers_with(default=False),
            'tiers: exactly one needs "default": true, found []',
        ),
        (
            {**DOCUMENTED, "tiers": [DOCUMENTED["tiers"][0]] * 2},  # type: ignore[index]
            "found ['cpu-8', 'cpu-8']",
        ),
        (
            tiers_with(id="cpu-16"),
            "tiers: tier id listed more than once: ['cpu-16']",
        ),
        (
            tiers_with(machines=["10.10.0.31"]),
            "tiers: machine listed more than once: ['10.10.0.31']",
        ),
    ],
)
def test_a_bad_file_is_refused_naming_the_file_and_the_key(
    tmp_path: Path, content: object, problem: str
) -> None:
    path = written(tmp_path, content)
    with pytest.raises(ValueError, match=f"^placement file {path}: ") as caught:
        load_placement(path)
    assert problem in str(caught.value)


def test_a_missing_placement_file_is_refused_naming_it(tmp_path: Path) -> None:
    path = tmp_path / "absent.json"
    with pytest.raises(ValueError, match=f"placement file {path}: cannot be read"):
        load_placement(path)


# --- the project env file ---


def env_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "project.env"
    path.write_text(text)
    return path


def test_project_env_lines_are_read_in_order_without_comments(tmp_path: Path) -> None:
    text = "# secrets\nRAG_ES_PASSWORD=s3 cr=t\n\n  # indented\nB=\nA_1='q'\nB=2\n"
    assert load_project_env(env_file(tmp_path, text)) == {
        "RAG_ES_PASSWORD": "s3 cr=t",
        "B": "2",
        "A_1": "'q'",
    }


@pytest.mark.parametrize(
    "line", ["NO_EQUALS", "1ABC=x", "export A=x", " A=x", "A B=x", "=x"]
)
def test_a_malformed_project_env_line_names_file_and_line_not_its_text(
    tmp_path: Path, line: str
) -> None:
    path = env_file(tmp_path, f"OK=1\n# c\n{line}\n")
    with pytest.raises(
        ValueError, match=f"^project env file {path}, line 3: "
    ) as caught:
        load_project_env(path)
    assert line not in str(caught.value)


def test_a_missing_project_env_file_is_refused_naming_it(tmp_path: Path) -> None:
    path = tmp_path / "absent.env"
    with pytest.raises(ValueError, match=f"project env file {path}: cannot be read"):
        load_project_env(path)
