import csv
import hashlib
import re
import shutil
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from .config import settings
from .labels import DEFAULT_LABEL
from .memory_guard import InsufficientMemoryError, ensure_headroom_for_file, wait_for_headroom
from .text_utils import split_sentences
from .vectorstore import VectorStore


def infer_label(path: Path, data_dir: str = settings.data_dir) -> str:
    """A file directly under data_dir is unlabeled ("general"); a file under
    data_dir/<label>/... (how uploads are organized) takes that label."""
    try:
        rel_parts = path.resolve().relative_to(Path(data_dir).resolve()).parts
    except ValueError:
        return DEFAULT_LABEL
    return rel_parts[0] if len(rel_parts) > 1 else DEFAULT_LABEL

MIN_EXTRACTED_CHARS = 40  # below this, a PDF is assumed scanned/image-based

# Windows' Tesseract installer doesn't reliably land on PATH for a process
# started before the install (or before a terminal restart) — fall back to
# the installer's default location if the bare command can't be found.
WINDOWS_TESSERACT_FALLBACK = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")


def _configure_tesseract(pytesseract) -> None:
    if not shutil.which(pytesseract.pytesseract.tesseract_cmd) and WINDOWS_TESSERACT_FALLBACK.exists():
        pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT_FALLBACK)


def _read_image(path: Path) -> str:
    """OCR for any visible text, plus a vision-model caption so images with
    no text at all (a photo, a diagram) still become searchable — captioning
    degrades gracefully to "skip it" if no vision-capable model is pulled,
    rather than failing the whole ingest."""
    import pytesseract
    from PIL import Image

    from .vision import VisionUnavailable, get_vision_model

    _configure_tesseract(pytesseract)
    with Image.open(path) as image:
        ocr_text = pytesseract.image_to_string(image)

    parts = []
    try:
        caption = get_vision_model().caption(path)
        if caption.strip():
            parts.append(f"Image description: {caption.strip()}")
    except VisionUnavailable:
        pass
    except Exception as exc:
        print(f"Vision captioning skipped for {path.name}: {exc}")

    if ocr_text.strip():
        parts.append(f"Text visible in image: {ocr_text.strip()}")

    return "\n\n".join(parts)


@lru_cache(maxsize=1)
def _get_whisper_model():
    from faster_whisper import WhisperModel

    from .device import resolve_device

    device = resolve_device()
    whisper_device = "cuda" if device == "cuda" else "cpu"  # ctranslate2 has no MPS backend
    compute_type = "float16" if whisper_device == "cuda" else "int8"
    return WhisperModel("base", device=whisper_device, compute_type=compute_type)


def _read_media(path: Path) -> str:
    """Transcribes speech from a video or audio file with faster-whisper."""
    model = _get_whisper_model()
    segments, _ = model.transcribe(str(path))
    return " ".join(segment.text.strip() for segment in segments)


def _ocr_pdf(path: Path) -> str:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    _configure_tesseract(pytesseract)

    pages_text = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            wait_for_headroom(context=f"OCR page {i + 1} of {path.name}")
            pix = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages_text.append(pytesseract.image_to_string(image))
    return "\n".join(pages_text)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(text.strip()) < MIN_EXTRACTED_CHARS:
        return _ocr_pdf(path)
    return text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_docx(path: Path) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _read_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _read_csv(path: Path) -> str:
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""

    header, body = rows[0], rows[1:]
    lines = []
    for row in body:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
        if pairs:
            lines.append(", ".join(pairs))
    return "\n".join(lines)


def _read_xlsx(path: Path) -> str:
    import openpyxl

    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    lines = []
    for sheet in workbook.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue
        header = [str(c) if c is not None else "" for c in rows[0]]
        for row in rows[1:]:
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v is not None]
            if pairs:
                lines.append(f"[{sheet.title}] " + ", ".join(pairs))
    return "\n".join(lines)


SUFFIX_READERS = {
    ".pdf": _read_pdf,
    ".txt": _read_text,
    ".md": _read_text,
    ".docx": _read_docx,
    ".html": _read_html,
    ".htm": _read_html,
    ".csv": _read_csv,
    ".xlsx": _read_xlsx,
    ".png": _read_image,
    ".jpg": _read_image,
    ".jpeg": _read_image,
    ".bmp": _read_image,
    ".tiff": _read_image,
    ".tif": _read_image,
    ".webp": _read_image,
    ".mp4": _read_media,
    ".mov": _read_media,
    ".avi": _read_media,
    ".mkv": _read_media,
    ".webm": _read_media,
    ".mp3": _read_media,
    ".wav": _read_media,
    ".m4a": _read_media,
    ".flac": _read_media,
    ".ogg": _read_media,
}


def load_documents(data_dir: str) -> list[tuple[str, str, str]]:
    """Returns (source_path, text, label) triples. Label is inferred from the
    file's immediate parent folder under data_dir (see infer_label)."""
    directory = Path(data_dir)
    documents = []
    for path in sorted(directory.rglob("*")):
        reader = SUFFIX_READERS.get(path.suffix.lower())
        if not reader:
            continue
        try:
            ensure_headroom_for_file(path.stat().st_size, context=f"reading {path.name}")
            documents.append((str(path), reader(path), infer_label(path, data_dir)))
        except InsufficientMemoryError as exc:
            # Don't let one oversized file block the rest of the batch.
            print(f"Skipping {path.name}: {exc}")
    return documents


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _word_window_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Last-resort splitter for a single unit (e.g. one sentence) too long to
    fit in a chunk on its own — the only case where a cut is mid-sentence."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_text(text: str, chunk_size: int = settings.chunk_size, overlap: int = settings.chunk_overlap) -> list[str]:
    """Recursive/semantic chunking: groups whole paragraphs — falling back to
    sentences, then a raw word window only if a single sentence is itself too
    long — up to chunk_size words, instead of cutting at an arbitrary word
    offset. Consecutive chunks overlap by carrying the tail of the previous
    chunk forward, for retrieval continuity across a chunk boundary."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    if not paragraphs:
        return []

    pieces = []
    for paragraph in paragraphs:
        if len(paragraph.split()) <= chunk_size:
            pieces.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence.split()) <= chunk_size:
                pieces.append(sentence)
            else:
                pieces.extend(_word_window_split(sentence, chunk_size, overlap))

    chunks = []
    current_words: list[str] = []
    for piece in pieces:
        piece_words = piece.split()
        if current_words and len(current_words) + len(piece_words) > chunk_size:
            chunks.append(" ".join(current_words))
            overlap_tail = current_words[-overlap:] if overlap else []
            # A piece from the word-window fallback can itself be close to
            # chunk_size — if the overlap tail would push it over, drop the
            # overlap for this one transition rather than exceed chunk_size.
            if len(overlap_tail) + len(piece_words) > chunk_size:
                overlap_tail = []
            current_words = overlap_tail + piece_words
        else:
            current_words.extend(piece_words)
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def ingest_document(source: str, text: str, store: VectorStore, label: str = DEFAULT_LABEL) -> int:
    chunks = chunk_text(text)
    if not chunks:
        return 0

    ids = [hashlib.sha1(f"{source}:{i}".encode()).hexdigest() for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk_index": i, "label": label} for i in range(len(chunks))]
    store.add_chunks(ids, chunks, metadatas)
    return len(chunks)


def ingest_path(path: Path, store: VectorStore) -> int:
    reader = SUFFIX_READERS.get(path.suffix.lower())
    if not reader:
        return 0
    ensure_headroom_for_file(path.stat().st_size, context=f"reading {path.name}")
    return ingest_document(str(path), reader(path), store, infer_label(path))


def ingest(data_dir: str = settings.data_dir, store: VectorStore | None = None) -> int:
    store = store or VectorStore()
    documents = load_documents(data_dir)
    if not documents:
        print(f"No supported files found in {data_dir}")
        return 0

    total_chunks = 0
    for source, text, label in documents:
        count = ingest_document(source, text, store, label)
        if count:
            total_chunks += count
            print(f"Ingested {count} chunks from {source} [{label}]")

    print(f"Done. {total_chunks} chunks in collection (total now {store.count()}).")
    return total_chunks


if __name__ == "__main__":
    ingest()
