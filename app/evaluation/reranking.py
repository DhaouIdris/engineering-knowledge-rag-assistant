"""Cross-encoder reranking for retrieval candidates."""

from __future__ import annotations

from typing import Any, Sequence


class CrossEncoderReranker:
    """Score question/chunk pairs jointly and return the best chunks."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 16,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")
        self.model_name = model_name
        self.batch_size = batch_size
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device=device)
        self.model = model

    def rerank(self, question: str, documents: Sequence[Any], k: int) -> list[Any]:
        if k <= 0:
            raise ValueError("k must be strictly positive.")
        candidates = list(documents)
        if not candidates:
            return []

        pairs = [(question, document.page_content) for document in candidates]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = [float(score) for score in raw_scores]
        if len(scores) != len(candidates):
            raise ValueError("The reranker returned one score per candidate.")

        ranked_indices = sorted(
            range(len(candidates)), key=lambda index: (-scores[index], index)
        )
        return [candidates[index] for index in ranked_indices[:k]]
