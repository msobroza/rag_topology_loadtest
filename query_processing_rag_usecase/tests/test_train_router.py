"""``qp-rag-train-router``: trains the router on the fake embedder, writes its file and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from query_processing_rag_usecase.entrypoints.train_router import main, train_router
from query_processing_rag_usecase.infrastructure.pipeline_artifact import manifest_path
from router_workflow import PIPELINE_FILE, settings_on_fakes


async def test_training_on_the_fake_embedder_writes_a_file_its_manifest_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = await train_router(
        settings_on_fakes(tmp_path, monkeypatch), "router", allow_fake_embedder=True
    )

    assert manifest.routes == ["leans_on_history", "self_contained"]
    assert manifest.n_neighbors == 3
    assert manifest.metric == "cosine"
    assert 0.0 <= manifest.leave_one_out_accuracy <= 1.0
    assert manifest.vector_size == 16  # the fake embedder's
    written = json.loads(manifest_path(tmp_path / PIPELINE_FILE).read_text("utf-8"))
    assert written == manifest.model_dump()


async def test_the_manifest_records_the_embedders_alias_not_its_source_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = await train_router(
        settings_on_fakes(tmp_path, monkeypatch), "router", allow_fake_embedder=True
    )

    assert manifest.embedder_alias == "Solon-embeddings-large-0.1"
    assert manifest.embedder_query_prefix == "query : "
    assert "OrdalieTech" not in manifest_path(tmp_path / PIPELINE_FILE).read_text()


async def test_a_name_that_is_no_router_is_refused_with_the_routers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_on_fakes(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=r"'router_embedder'.*expected.*'router'"):
        await train_router(settings, "router_embedder", allow_fake_embedder=True)


def test_the_command_reports_the_questions_it_routes_as_self_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings_on_fakes(tmp_path, monkeypatch)
    questions = tmp_path / "questions.jsonl"
    lines = [
        {"question": "Que couvre la garantie de responsabilité civile ?"},
        {"question": "Et pour un véhicule loué ?"},
        {"question": "Pourquoi est-ce obligatoire ?"},
    ]
    questions.write_text("\n".join(json.dumps(line) for line in lines), "utf-8")

    assert (
        main(
            [
                "--router",
                "router",
                "--allow-fake-embedder",
                "--check-questions",
                str(questions),
            ]
        )
        == 0
    )

    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["questions"] == 3
    assert report["routed"] == {"self_contained": 1, "leans_on_history": 2}
    assert (tmp_path / PIPELINE_FILE).is_file()


async def test_training_on_the_fake_embedder_is_refused_unless_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fake embedder's vectors are not the real one's: a pipeline trained on
    # them would carry the real embedder's alias in its manifest all the same.
    settings = settings_on_fakes(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="fake") as refused:
        await train_router(settings, "router")

    message = str(refused.value)
    assert "'router_embedder'" in message
    assert "QP_COMPONENTS__ROUTER_EMBEDDER__BACKEND" in message
    assert "--allow-fake-embedder" in message
    assert not (tmp_path / PIPELINE_FILE).exists()


def test_the_command_on_the_fake_embedder_is_a_usage_error_without_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings_on_fakes(tmp_path, monkeypatch)

    with pytest.raises(SystemExit) as stop:
        main(["--router", "router"])

    assert stop.value.code == 2
    assert "unset QP_COMPONENTS__ROUTER_EMBEDDER__BACKEND" in capsys.readouterr().err


def test_a_questions_file_that_is_not_one_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings_on_fakes(tmp_path, monkeypatch)
    questions = tmp_path / "questions.jsonl"
    questions.write_text("not json\n", "utf-8")

    with pytest.raises(SystemExit) as stop:
        main(["--allow-fake-embedder", "--check-questions", str(questions)])

    assert stop.value.code == 2
    assert "line 1 is 'not json'" in capsys.readouterr().err
