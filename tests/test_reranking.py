from dataclasses import dataclass

import pytest

from app.evaluation.reranking import CrossEncoderReranker


@dataclass
class FakeDocument:
    page_content: str


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, *, batch_size, show_progress_bar):
        self.calls.append((pairs, batch_size, show_progress_bar))
        return self.scores


def test_cross_encoder_reranks_candidates_and_keeps_ties_stable():
    model = FakeCrossEncoder([0.2, 0.9, 0.9])
    reranker = CrossEncoderReranker("fake", batch_size=8, model=model)
    documents = [FakeDocument("A"), FakeDocument("B"), FakeDocument("C")]

    result = reranker.rerank("question", documents, k=2)

    assert [document.page_content for document in result] == ["B", "C"]
    assert model.calls == [
        ([("question", "A"), ("question", "B"), ("question", "C")], 8, False)
    ]


def test_cross_encoder_validates_parameters_and_empty_candidates():
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker("fake", batch_size=0, model=FakeCrossEncoder([]))

    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder([]))
    assert reranker.rerank("question", [], k=3) == []
    with pytest.raises(ValueError, match="k"):
        reranker.rerank("question", [FakeDocument("A")], k=0)


def test_cross_encoder_rejects_incomplete_scores():
    reranker = CrossEncoderReranker("fake", model=FakeCrossEncoder([0.5]))
    with pytest.raises(ValueError, match="one score per candidate"):
        reranker.rerank(
            "question", [FakeDocument("A"), FakeDocument("B")], k=1
        )
