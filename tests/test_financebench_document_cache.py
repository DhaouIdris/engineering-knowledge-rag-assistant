from pathlib import Path

import pytest

pytest.importorskip("langchain_community")

from scripts import evaluate_financebench_retrieval as financebench_script


class _FakeStore:
    def save_local(self, target: str) -> None:
        Path(target).mkdir(parents=True, exist_ok=True)


class _FakeFAISS:
    built = 0
    loaded = 0

    @classmethod
    def from_documents(cls, documents, embeddings):
        cls.built += 1
        return _FakeStore()

    @classmethod
    def load_local(cls, target, embeddings, allow_dangerous_deserialization):
        cls.loaded += 1
        return _FakeStore()


def test_per_document_indexes_resume_from_individual_caches(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    for filename in ("A.pdf", "B.pdf"):
        (pdf_dir / filename).write_bytes(filename.encode())

    dataset = {
        "examples": [
            {"relevant_locations": {("A.pdf", 0)}},
            {"relevant_locations": {("B.pdf", 1)}},
        ]
    }
    loaded_documents = []

    def fake_load_documents(folder, filenames):
        filename = next(iter(filenames))
        loaded_documents.append(filename)
        return [f"page:{filename}"]

    _FakeFAISS.built = 0
    _FakeFAISS.loaded = 0
    monkeypatch.setattr(financebench_script, "FAISS", _FakeFAISS)
    monkeypatch.setattr(financebench_script, "load_documents", fake_load_documents)
    monkeypatch.setattr(
        financebench_script, "build_chunks", lambda pages, chunk_size, overlap: pages
    )

    arguments = {
        "dataset": dataset,
        "embeddings": object(),
        "model": "test/model",
        "pdf_dir": pdf_dir,
        "cache_dir": tmp_path / "cache",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "rebuild": False,
    }
    stores, cache_hits, chunks = financebench_script.load_or_build_document_stores(
        **arguments
    )

    assert set(stores) == {"A.pdf", "B.pdf"}
    assert cache_hits == 0
    assert chunks == 2
    assert _FakeFAISS.built == 2
    assert loaded_documents == ["A.pdf", "B.pdf"]

    loaded_documents.clear()
    stores, cache_hits, chunks = financebench_script.load_or_build_document_stores(
        **arguments
    )

    assert set(stores) == {"A.pdf", "B.pdf"}
    assert cache_hits == 2
    assert chunks == 2
    assert _FakeFAISS.built == 2
    assert _FakeFAISS.loaded == 2
    assert loaded_documents == []
