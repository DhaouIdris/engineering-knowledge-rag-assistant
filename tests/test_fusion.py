from dataclasses import dataclass
from typing import Any

from app.evaluation.fusion import reciprocal_rank_fusion


@dataclass
class Chunk:
    page_content: str
    metadata: dict[str, Any]


def test_shared_result_ranks_above_results_from_only_one_retriever():
    dense_only = Chunk("dense only", {"source": "a.pdf", "page": 1})
    lexical_only = Chunk("lexical only", {"source": "a.pdf", "page": 2})
    shared = Chunk("shared", {"source": "a.pdf", "page": 3})

    assert reciprocal_rank_fusion(
        ([dense_only, shared], [lexical_only, shared]), k=2
    ) == [shared, dense_only]


def test_same_chunk_cannot_vote_twice_within_one_ranking():
    shared = Chunk("shared", {"source": "a.pdf", "page": 3})
    other = Chunk("other", {"source": "a.pdf", "page": 4})

    assert reciprocal_rank_fusion(([shared, shared], [other]), k=3) == [shared, other]
