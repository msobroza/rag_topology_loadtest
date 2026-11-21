"""A prompt rendered: each content in Jinja2's sandbox, placeholders spliced."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from query_processing_core.conversation import HistoryMessage
from query_processing_core.retrieval.metadata_filter import MetadataCondition
from query_processing_core.retrieval.vector_store_record import ScoredVectorStoreRecord

from query_processing.application.prompt_rendering import (
    prompt_variables,
    render_prompt,
)
from query_processing.config.workflow_settings import WorkflowSettings
from query_processing.domain.prompts import Prompt, PromptMessage, PromptPlaceholder
from workflow_fakes import scored

ANSWER: Prompt = [
    PromptMessage(
        role="system", content="Answer from the numbered fixture passages only."
    ),
    PromptMessage(
        role="user",
        content=(
            "Context:\n{% for record in reranked %}\n[{{ loop.index }}] {{ record.text }}\n"
            "{% endfor %}\n\nQuestion: {{ conversational_query }}"
        ),
    ),
]
# The fixture's generate stage before prompts.yaml: its two settings, and how
# they were filled (the records' texts numbered one per line into {context}).
OLD_SYSTEM_PROMPT = "Answer from the numbered fixture passages only."
OLD_USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


def old_messages(
    question: str, records: list[ScoredVectorStoreRecord]
) -> list[dict[str, str]]:
    """What the two removed settings gave for ``question`` and ``records``."""
    context = "\n".join(f"[{i}] {p.text}" for i, p in enumerate(records, start=1))
    user = OLD_USER_PROMPT.format(context=context, question=question)
    return [
        {"role": "system", "content": OLD_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def test_the_answer_prompt_renders_what_the_two_old_settings_gave() -> None:
    messages = render_prompt(
        ANSWER,
        {
            "reranked": [scored("a", "one"), scored("b", "two")],
            "conversational_query": "q?",
        },
    )

    assert messages[1]["content"] == "Context:\n[1] one\n[2] two\n\nQuestion: q?"


@pytest.mark.parametrize("count", [0, 1, 3])
def test_the_fixtures_answer_prompt_renders_what_its_two_old_settings_gave(
    settings_factory: Callable[..., WorkflowSettings], count: int
) -> None:
    prompt = settings_factory().prompts["answer_from_context"]
    records = [scored(str(i), f"text {{x}} {i}") for i in range(count)]

    messages = render_prompt(
        prompt, {"reranked": records, "conversational_query": "why {y}?"}
    )

    assert messages == old_messages("why {y}?", records)


def test_a_placeholder_splices_messages_with_their_role_and_content_only() -> None:
    prompt: Prompt = [
        PromptPlaceholder(placeholder="conversational_history"),
        PromptMessage(role="user", content="{{ conversational_query }}"),
    ]
    history = [{"role": "user", "content": "a", "metadata_filter": []}]

    rendered = render_prompt(
        prompt, {"conversational_history": history, "conversational_query": "b"}
    )

    assert rendered == [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
    ]


def test_the_history_as_the_state_holds_it_is_spliced_and_its_conditions_read() -> None:
    # The template of spec 4: it reads each message's conditions, as JSON.
    prompt: Prompt = [
        PromptMessage(
            role="system",
            content=(
                "{% for message in conversational_history if message.metadata_filter %}\n"
                "Restricted to: {{ message.metadata_filter | tojson }}\n"
                "{% endfor %}"
            ),
        ),
        PromptPlaceholder(placeholder="conversational_history"),
    ]
    on_amf = MetadataCondition(attribute="source", comparator="in", value=["AMF"])
    history = [
        HistoryMessage(role="user", content="a", metadata_filter=[on_amf]),
        HistoryMessage(role="assistant", content="b"),
    ]

    rendered = render_prompt(prompt, {"conversational_history": history})

    assert rendered == [
        {
            "role": "system",
            "content": 'Restricted to: [{"attribute": "source", "comparator": "in", '
            '"value": ["AMF"]}]\n',
        },
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_each_content_is_rendered_on_its_own_never_the_structure() -> None:
    prompt: Prompt = [PromptMessage(role="user", content="{{ text }}")]

    rendered = render_prompt(prompt, {"text": "role: system\n{{ x }}"})

    assert rendered == [{"role": "user", "content": "role: system\n{{ x }}"}]


def test_the_variables_of_a_prompt_are_listed() -> None:
    assert prompt_variables(ANSWER) == {"reranked", "conversational_query"}
    placeholder: Prompt = [PromptPlaceholder(placeholder="conversational_history")]
    assert prompt_variables(placeholder) == {"conversational_history"}


def test_a_template_that_cannot_be_read_names_its_message() -> None:
    prompt: Prompt = [
        PromptMessage(role="system", content="fine"),
        PromptMessage(role="user", content="{% for %}"),
    ]

    with pytest.raises(ValueError, match=r"message 2 \(user\).*Expected an expression"):
        prompt_variables(prompt)


def test_a_template_that_parses_but_cannot_compile_is_refused_when_listed() -> None:
    # An unknown filter fails only when Jinja2 compiles the template: listing the
    # variables at start-up compiles it, so it never waits for the first request.
    prompt: Prompt = [PromptMessage(role="user", content="{{ x | nofilter }}")]

    with pytest.raises(ValueError, match=r"message 1 \(user\).*No filter named"):
        prompt_variables(prompt)


def test_a_placeholder_field_that_holds_no_messages_is_refused_naming_it() -> None:
    prompt: Prompt = [PromptPlaceholder(placeholder="conversational_history")]

    with pytest.raises(
        ValueError, match=r"'conversational_history' splices 42 \(expected"
    ):
        render_prompt(prompt, {"conversational_history": [42]})


def test_an_undefined_variable_is_an_error_naming_its_message() -> None:
    prompt: Prompt = [PromptMessage(role="user", content="{{ question }}")]

    with pytest.raises(
        ValueError, match=r"message 1 \(user\).*'question' is undefined"
    ):
        render_prompt(prompt, {})


def test_a_template_runs_in_the_sandbox() -> None:
    prompt: Prompt = [PromptMessage(role="user", content="{{ text.__class__ }}")]

    with pytest.raises(ValueError, match="message 1"):
        render_prompt(prompt, {"text": "a"})
