"""``qp-rag-train-router``: trains the kNN router on its routes' utterances.

It embeds each utterance of the router's routes through the ``ainvoke`` of the
embedder component the router names, so the embedder's query prefix is
applied as at query time; fits a normalizer and a k-nearest-neighbours
classifier with the values of the router's ``local`` block; scores it by
leave-one-out; and writes the pipeline file with its manifest (spec 9.3).
Run from the use case's directory, where it finds ``workflow.yaml`` and
``routes.yaml``. With ``--check-questions`` it then routes the questions of a
file, one ``{"question"}`` per line, and reports how many go to each route:
a check run by hand on questions that are never committed.

Example:
    $ qp-rag-train-router --router router --check-questions data/quebec/questions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel
from query_processing.extension_api import (
    BaseComponent,
    CompositionRoot,
    EmbedAnswer,
    EmbedRequest,
    Route,
    RouterSettings,
    WorkflowSettings,
)

from ..config.knn_router_settings import KnnRouterSettings
from ..infrastructure.knn_pipeline import fit_pipeline
from ..infrastructure.pipeline_artifact import (
    PipelineManifest,
    expected_manifest_values,
    load_pipeline,
    save_pipeline,
)

# The backend whose block holds the pipeline file and the training values.
_LOCAL = "local"


class RouterParts(NamedTuple):
    """What training and routing read of one router: all four travel together."""

    router: RouterSettings
    knn: KnnRouterSettings  # its local block
    routes: list[Route]  # in the order it lists them
    embedder: BaseModel  # the settings of the embedder it reads


def router_parts(settings: WorkflowSettings, name: str) -> RouterParts:
    """Return a router's settings, local block, routes and embedder's settings.

    Args:
        settings: The workflow settings, with the routes of ``routes.yaml``.
        name: A router component of the workflow file.

    Returns:
        The four, read from the settings.

    Example:
        >>> router_parts(settings, "router").knn.n_neighbors
        3
    """
    router = router_of(settings, name)
    return RouterParts(
        router,
        knn_block_of(settings, name),
        routes_of(settings, router),
        settings.components[router.embedder],
    )


async def train_router(
    settings: WorkflowSettings, router_name: str
) -> PipelineManifest:
    """Train the router's pipeline and write it with its manifest.

    Args:
        settings: The workflow settings, with the routes of ``routes.yaml``.
        router_name: The router component to train.

    Returns:
        The manifest written beside the pipeline file.

    Example:
        >>> manifest = await train_router(WorkflowSettings(), "router")
        >>> manifest.routes
        ['leans_on_history', 'self_contained']
    """
    router, knn, routes, embedder = router_parts(settings, router_name)
    texts = [u for route in routes for u in route.utterances]
    labels = [route.name for route in routes for _ in route.utterances]
    vectors = await embed_texts(settings, router.embedder, texts)
    fitted = fit_pipeline(
        vectors, labels, n_neighbors=knn.n_neighbors, metric=knn.metric
    )
    manifest = PipelineManifest.model_validate(
        {
            **expected_manifest_values(knn, embedder, routes),
            "pipeline_sha256": "",  # filled once the file is written
            "vector_size": len(vectors[0]),
            "leave_one_out_accuracy": fitted.leave_one_out_accuracy,
        }
    )
    return save_pipeline(fitted.pipeline, Path(knn.pipeline_file), manifest)


async def route_questions(
    settings: WorkflowSettings, router_name: str, questions_file: Path
) -> dict[str, int]:
    """Route the questions of a file with the router's trained pipeline.

    The pipeline is loaded as start-up loads it, so its manifest is checked.

    Args:
        settings: The workflow settings, with the routes of ``routes.yaml``.
        router_name: The router component whose pipeline routes them.
        questions_file: One ``{"question": ...}`` object per line.

    Returns:
        How many questions go to each of the router's routes, in its order.

    Example:
        >>> await route_questions(settings, "router", Path("questions.jsonl"))
        {'self_contained': 61, 'leans_on_history': 21}
    """
    router, knn, routes, embedder = router_parts(settings, router_name)
    pipeline = load_pipeline(knn, embedder, routes)
    vectors = await embed_texts(
        settings, router.embedder, read_questions(questions_file)
    )
    counts = dict.fromkeys(router.routes, 0)
    for name in pipeline.predict(vectors) if vectors else []:
        counts[str(name)] += 1
    return counts


def router_of(settings: WorkflowSettings, name: str) -> RouterSettings:
    """Return the settings of the router component ``name``.

    Args:
        settings: The workflow settings.
        name: A component of the workflow file.

    Returns:
        Its settings.

    Raises:
        ValueError: If it is no router; the message names the routers declared.

    Example:
        >>> router_of(settings, "router").embedder
        'router_embedder'
    """
    component = settings.components.get(name)
    if isinstance(component, RouterSettings):
        return component
    routers = [
        n for n, c in settings.components.items() if isinstance(c, RouterSettings)
    ]
    raise ValueError(
        f"component {name!r} is no router of the workflow file; expected one of "
        f"its routers: {', '.join(map(repr, routers)) or 'none'}"
    )


def knn_block_of(settings: WorkflowSettings, name: str) -> KnnRouterSettings:
    """Return the router's ``local`` block, read as a kNN router's settings.

    Args:
        settings: The workflow settings.
        name: A router component of the workflow file.

    Returns:
        Its pipeline file and training values.

    Raises:
        ValueError: If the router states no ``local`` block; the message names
            the keys expected.

    Example:
        >>> knn_block_of(settings, "router").n_neighbors
        3
    """
    block = (router_of(settings, name).model_extra or {}).get(_LOCAL)
    if isinstance(block, KnnRouterSettings):
        return block
    raise ValueError(
        f"router {name!r} has the local block {block!r}; expected one with "
        "pipeline_file, n_neighbors and metric, which training reads whatever "
        "backend the router is on"
    )


def routes_of(settings: WorkflowSettings, router: RouterSettings) -> list[Route]:
    """Return the routes a router lists, from the catalogue of ``routes.yaml``.

    Args:
        settings: The workflow settings; they refuse a route the file lacks.
        router: The router's settings.

    Returns:
        Its routes, in the order it lists them.

    Example:
        >>> [r.name for r in routes_of(settings, router_of(settings, "router"))]
        ['self_contained', 'leans_on_history']
    """
    catalogue = {route.name: route for route in settings.routes}
    return [catalogue[name] for name in router.routes]


async def embed_texts(
    settings: WorkflowSettings, embedder_name: str, texts: Sequence[str]
) -> list[list[float]]:
    """Embed texts as queries, through the ``ainvoke`` of one embedder component.

    Only that component opens: the composition is given a graph of one stage
    that calls it, so that neither the router, whose pipeline may not exist
    yet, nor any other component of the workflow file is opened.

    Args:
        settings: The workflow settings.
        embedder_name: The embedder component to embed with.
        texts: The texts to embed.

    Returns:
        One vector per text, in the same order.

    Raises:
        TypeError: If the component opens as no component, or answers with no
            vector; the message shows what it gave.

    Example:
        >>> await embed_texts(settings, "router_embedder", ["Et sinon ?"])
        [[0.12, ...]]
    """
    stage = {"call": embedder_name, "reads": [], "writes": []}
    graph = type(settings.graph).model_validate({"stages": {embedder_name: stage}})
    composition = CompositionRoot(settings.model_copy(update={"graph": graph}))
    async with composition:
        opened = (await composition.workflow_dependencies()).components()
        embedder = opened[embedder_name]
        if not isinstance(embedder, BaseComponent):
            raise TypeError(
                f"{embedder_name!r} opened as {embedder!r}; expected a component"
            )
        answers = [await embedder.ainvoke(EmbedRequest(text=text)) for text in texts]
    return [_vector_of(answer) for answer in answers]


def read_questions(questions_file: Path) -> list[str]:
    """Read the questions of a file, one ``{"question": ...}`` object per line.

    Args:
        questions_file: The file, blank lines skipped.

    Returns:
        The questions, in file order.

    Raises:
        ValueError: If a line is no such object; the message names the line.

    Example:
        >>> read_questions(Path("questions.jsonl"))[0]
        'Quelle est la franchise ?'
    """
    lines = questions_file.read_text(encoding="utf-8").splitlines()
    return [
        _question(questions_file, number, line)
        for number, line in enumerate(lines, start=1)
        if line.strip()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``qp-rag-train-router``: train, then print the manifest as one JSON line.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        The exit status, 0.

    Example:
        >>> main(["--router", "router"])
        {"event": "router_trained", ...}
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        settings = WorkflowSettings()
        manifest = asyncio.run(train_router(settings, args.router))
        print(json.dumps({"event": "router_trained", **manifest.model_dump()}))
        if args.check_questions is not None:
            _report_routed(settings, args.router, args.check_questions)
    except ValueError as exc:  # a settings or file mistake is a usage error
        parser.error(str(exc))
    return 0


def _report_routed(settings: WorkflowSettings, router: str, questions: Path) -> None:
    routed = asyncio.run(route_questions(settings, router, questions))
    report = {"questions": sum(routed.values()), "routed": routed}
    print(json.dumps({"event": "questions_routed", **report}, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qp-rag-train-router",
        description="Train the kNN router on its routes' utterances.",
    )
    parser.add_argument(
        "--router", default="router", metavar="NAME", help="the router component"
    )
    parser.add_argument(
        "--check-questions",
        type=Path,
        metavar="FILE",
        help="then report how many of this file's questions go to each route",
    )
    return parser


def _vector_of(answer: object) -> list[float]:
    if not isinstance(answer, EmbedAnswer):
        raise TypeError(f"the embedder answered {answer!r}; expected an EmbedAnswer")
    return answer.embedding


def _question(path: Path, number: int, line: str) -> str:
    try:
        question = json.loads(line)["question"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"{path}: line {number} is {line[:80]!r}; expected an object "
            '{"question": "..."}'
        ) from exc
    return str(question)
