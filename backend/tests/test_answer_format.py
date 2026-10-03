from types import SimpleNamespace

import pytest

from scholargraph import agent as agent_module
from scholargraph.agent import (
    SYNTHESIS_MAX_CLAIMS,
    SYNTHESIS_MIN_CLAIMS,
    GroundedClaim,
    GroundedResponse,
    GroundingValidationError,
    ScholarGraphAgent,
    format_grounded_answer,
)

RAG_ID = "W4401857375"
COLBERT_ID = "W3021397474"

EVIDENCE = [
    {
        "paper_id": RAG_ID,
        "title": "Retrieval-Augmented Large Language Models: A Survey",
        "authors": ["Alice Writer", "Bob Reader", "Carol Third"],
        "published": "2024-08-03",
    },
    {
        "paper_id": COLBERT_ID,
        "title": "ColBERT: Efficient Retrieval via Late Interaction",
        "authors": ["Omar Khattab", "Matei Zaharia"],
        "published": "2020-07-01",
    },
    {
        "paper_id": "internal-7",
        "title": "",
        "authors": [],
        "published": None,
    },
]


def test_answer_uses_numbered_citations_and_a_sources_list() -> None:
    claims = [
        GroundedClaim(statement="Overview of the topic.", source_ids=[RAG_ID]),
        GroundedClaim(
            statement="A detailed point about retrieval.",
            source_ids=[COLBERT_ID, RAG_ID],
        ),
    ]

    answer = format_grounded_answer(claims, EVIDENCE)

    assert answer == (
        "Overview of the topic. [1]\n\n"
        "- A detailed point about retrieval. [2] [1]\n\n"
        "**Sources**\n\n"
        "1. Retrieval-Augmented Large Language Models: A Survey "
        "(Alice Writer et al., 2024) — [OpenAlex](https://openalex.org/W4401857375)\n"
        "2. ColBERT: Efficient Retrieval via Late Interaction "
        "(Omar Khattab and Matei Zaharia, 2020) — "
        "[OpenAlex](https://openalex.org/W3021397474)"
    )


def test_raw_paper_ids_are_not_shown_in_the_answer_text() -> None:
    claims = [GroundedClaim(statement="Overview.", source_ids=[RAG_ID])]

    body, _, sources = format_grounded_answer(claims, EVIDENCE).partition("**Sources**")

    assert RAG_ID not in body
    assert f"[{RAG_ID}]" not in sources  # only appears inside the OpenAlex link URL


def test_sources_list_only_contains_cited_papers_in_citation_order() -> None:
    claims = [GroundedClaim(statement="Only one.", source_ids=[COLBERT_ID])]

    answer = format_grounded_answer(claims, EVIDENCE)

    assert "ColBERT" in answer
    assert "Retrieval-Augmented" not in answer
    assert answer.count("\n1. ") == 1


def test_repeated_ids_in_one_claim_are_cited_once() -> None:
    claims = [GroundedClaim(statement="Repeat.", source_ids=[RAG_ID, RAG_ID])]

    assert format_grounded_answer(claims, EVIDENCE).startswith("Repeat. [1]\n\n")


def test_non_openalex_ids_and_missing_details_degrade_gracefully() -> None:
    claims = [GroundedClaim(statement="Internal.", source_ids=["internal-7"])]

    answer = format_grounded_answer(claims, EVIDENCE)

    assert answer.endswith("**Sources**\n\n1. Untitled paper")
    assert "openalex.org" not in answer


def _synthesis_state() -> dict:
    return {
        "user_query": "How do retrieval-augmented models work?",
        "conversation_history": [],
        "reranked_evidence": EVIDENCE[:2],
        "source_ids": {RAG_ID, COLBERT_ID},
        "retrieval": {"graph_result_count": 0, "vector_hit_count": 2},
    }


def _agent_returning(claims: list[GroundedClaim], captured: dict) -> ScholarGraphAgent:
    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                text=GroundedResponse(claims=claims).model_dump_json()
            )

    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    agent.client = SimpleNamespace(models=Models())
    return agent


def test_synthesis_asks_for_a_detailed_answer_and_no_ids_in_the_text() -> None:
    captured: dict = {}
    agent = _agent_returning(
        [GroundedClaim(statement="Overview.", source_ids=[RAG_ID])], captured
    )

    agent._node_synthesis(_synthesis_state())

    prompt = captured["contents"]
    assert f"between {SYNTHESIS_MIN_CLAIMS} and {SYNTHESIS_MAX_CLAIMS} claims" in prompt
    assert "2-4 sentences" in prompt
    assert "Never write IDs" in prompt
    assert captured["config"]["response_schema"] is GroundedResponse


def test_claim_schema_tells_the_model_to_write_substantive_statements() -> None:
    schema = GroundedResponse.model_json_schema()
    statement = schema["$defs"]["GroundedClaim"]["properties"]["statement"]

    assert "2-4 sentences" in statement["description"]


def test_synthesis_returns_the_formatted_answer() -> None:
    claims = [
        GroundedClaim(statement="Overview.", source_ids=[RAG_ID]),
        GroundedClaim(statement="Detail.", source_ids=[COLBERT_ID]),
    ]
    agent = _agent_returning(claims, {})

    result = agent._node_synthesis(_synthesis_state())["result"]

    assert result.answer.startswith("Overview. [1]\n\n- Detail. [2]\n\n**Sources**")
    assert result.retrieval == _synthesis_state()["retrieval"]


def test_synthesis_still_rejects_citations_outside_the_evidence() -> None:
    agent = _agent_returning(
        [GroundedClaim(statement="Made up.", source_ids=["W-fake"])], {}
    )

    with pytest.raises(GroundingValidationError):
        agent._node_synthesis(_synthesis_state())


def test_reranker_keeps_title_and_publication_date_for_the_sources_list(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        agent_module,
        "get_documents_by_ids",
        lambda _ids: {
            "W-graph": {
                "document": "Stored abstract.",
                "metadata": {"title": "Stored title", "published": "2023-05-01"},
            }
        },
    )
    agent = ScholarGraphAgent.__new__(ScholarGraphAgent)
    agent.reranker = SimpleNamespace(predict=lambda pairs: [0.5 for _ in pairs])

    result = agent._node_reranker(
        {
            "user_query": "anything",
            "fused_paper_ids": ["W-graph", "W-vector"],
            "vector_raw_results": [
                {
                    "paper_id": "W-vector",
                    "document": "Vector abstract.",
                    "metadata": {"title": "Vector title", "published": "2025-01-02"},
                }
            ],
            # Neo4j knows neither title, so ChromaDB metadata must fill the gap.
            "expanded_graph_context": {},
        }
    )

    by_id = {item["paper_id"]: item for item in result["reranked_evidence"]}
    assert by_id["W-graph"]["title"] == "Stored title"
    assert by_id["W-graph"]["published"] == "2023-05-01"
    assert by_id["W-vector"]["title"] == "Vector title"
    assert by_id["W-vector"]["published"] == "2025-01-02"
