import pytest

pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_community")

from scripts.evaluate_financebench_generation import (  # noqa: E402
    append_checkpoint,
    load_checkpoint,
)


def test_generation_checkpoint_round_trip_and_signature_validation(tmp_path):
    checkpoint = tmp_path / "answers.jsonl"
    append_checkpoint(
        checkpoint, {"run_signature": "run-a", "id": "question-1", "answer": "A"}
    )
    append_checkpoint(
        checkpoint, {"run_signature": "run-a", "id": "question-2", "answer": "B"}
    )

    completed = load_checkpoint(checkpoint, "run-a")

    assert set(completed) == {"question-1", "question-2"}
    assert completed["question-2"]["answer"] == "B"
    with pytest.raises(ValueError, match="configuration differs"):
        load_checkpoint(checkpoint, "run-b")
