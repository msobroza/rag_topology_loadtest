"""A workflow file names the packages it loads under ``extensions``.

Every module of each package is imported before the components are read, so a
component type an extension defines is read like the toolkit's own: its
backend's block with the settings class its implementation names.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from query_processing.config.files.workflow_file import WORKFLOW_FILE_VARIABLE
from query_processing.config.workflow_settings import WorkflowSettings
from shout_components import SHOUT_EXTENSION, write_shout_workflow
from shout_extension.shouting import UpperShouterSettings


def settings_from(
    tmp_path: Path, extensions: list[str], components: dict[str, Any]
) -> WorkflowSettings:
    path = write_shout_workflow(tmp_path / "workflow.yaml", extensions, components)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(WORKFLOW_FILE_VARIABLE, str(path))
        return WorkflowSettings(_env_file=None)


def write_package(root: Path, files: dict[str, str]) -> None:
    """Write a package's files under ``root``, each path relative to it."""
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


def test_a_workflow_file_loads_the_components_of_its_extensions(
    tmp_path: Path,
) -> None:
    settings = settings_from(
        tmp_path,
        [SHOUT_EXTENSION],
        {"shouter": {"backend": "local", "local": {"suffix": "!"}}},
    )

    assert settings.extensions == [SHOUT_EXTENSION]
    assert settings.components["shouter"].backend_block() == UpperShouterSettings(
        suffix="!"
    )


def test_every_module_of_an_extension_is_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_package(
        tmp_path,
        {
            "nested_ext/__init__.py": "",
            "nested_ext/inner/__init__.py": "",
            "nested_ext/inner/deep.py": "LOADED = True\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    settings_from(tmp_path, ["nested_ext"], {})

    assert sys.modules["nested_ext.inner.deep"].LOADED


def test_the_block_of_the_backend_in_use_gets_its_defaults(tmp_path: Path) -> None:
    settings = settings_from(
        tmp_path, [SHOUT_EXTENSION], {"shouter": {"backend": "local"}}
    )

    assert settings.components["shouter"].backend_block() == UpperShouterSettings()


def test_an_unknown_key_of_a_backend_block_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"sufix.*expected.*suffix"):
        settings_from(
            tmp_path,
            [SHOUT_EXTENSION],
            {"shouter": {"backend": "local", "local": {"sufix": "!"}}},
        )


def test_a_key_the_component_type_does_not_have_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match=r"'shouter' component type has no 'loud'.*fake, local"
    ):
        settings_from(
            tmp_path, [SHOUT_EXTENSION], {"shouter": {"backend": "local", "loud": True}}
        )


def test_an_extension_that_cannot_be_imported_names_the_package(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"extension 'no_such_package' cannot be imported: ModuleNotFoundError",
    ):
        settings_from(tmp_path, ["no_such_package"], {})


def test_an_extension_that_names_a_backend_twice_names_both_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_package(
        tmp_path,
        {
            "twice_ext/__init__.py": "",
            "twice_ext/second.py": (
                "from shout_extension.shouting import BaseShouter\n"
                "class SecondShouter(BaseShouter, backend='local'):\n"
                "    def invoke(self, request):\n"
                "        return request\n"
            ),
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(
        ValueError,
        match=r"extension 'twice_ext' cannot be imported: TypeError: shouter "
        r"backend 'local' is already UpperShouter, so SecondShouter",
    ):
        settings_from(tmp_path, ["twice_ext", SHOUT_EXTENSION], {})
