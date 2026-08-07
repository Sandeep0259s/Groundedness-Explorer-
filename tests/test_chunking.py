from pathlib import Path

from src.rag.ingest import chunk_text, infer_label


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=800, overlap=120) == []


def test_chunk_text_short_single_paragraph_is_one_chunk():
    chunks = chunk_text("This is a short document.", chunk_size=800, overlap=120)
    assert chunks == ["This is a short document."]


def test_chunk_text_merges_short_paragraphs():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=800, overlap=120)
    assert len(chunks) == 1
    assert "Paragraph one." in chunks[0]
    assert "Paragraph three." in chunks[0]


def test_chunk_text_splits_oversized_paragraph_by_sentence():
    # A paragraph bigger than chunk_size, but each sentence fits on its own.
    sentences = [f"This is sentence number {i} in a long paragraph." for i in range(60)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    # No sentence should be cut in half — each chunk should start and end on
    # a sentence boundary (i.e. end with a period).
    for chunk in chunks:
        assert chunk.strip().endswith(".")


def test_chunk_text_word_window_fallback_for_oversized_sentence():
    # A single "sentence" (no punctuation) far bigger than chunk_size must
    # still get split, even though there's no sentence boundary to use.
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_chunk_text_overlap_carries_context_across_boundary():
    sentences = [f"Sentence {i} has some unique content here." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=50, overlap=15)
    assert len(chunks) >= 2
    # The tail of chunk N should reappear at the head of chunk N+1.
    tail_words = chunks[0].split()[-5:]
    assert any(w in chunks[1] for w in tail_words)


def test_infer_label_root_file_is_general(tmp_path):
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    file_path = data_dir / "note.txt"
    file_path.write_text("hello")
    assert infer_label(file_path, str(data_dir)) == "general"


def test_infer_label_subfolder_is_that_label(tmp_path):
    data_dir = tmp_path / "data" / "raw"
    (data_dir / "resume").mkdir(parents=True)
    file_path = data_dir / "resume" / "cv.pdf"
    file_path.write_text("hello")
    assert infer_label(file_path, str(data_dir)) == "resume"
