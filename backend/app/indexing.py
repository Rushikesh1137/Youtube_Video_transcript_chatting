import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from google.genai.errors import APIError
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, PineconeException

from .transcripts import TranscriptResponse, TranscriptSegment

EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
UPSERT_BATCH_SIZE = 100
DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"


class IndexingConfigurationError(Exception):
    pass


class IndexingProviderError(Exception):
    pass


class PineconeIndex(Protocol):
    def upsert(
        self, vectors: list[dict[str, object]], namespace: str
    ) -> object: ...


class Embedder(Protocol):
    def embed_documents(
        self, texts: list[str], *, task_type: str
    ) -> list[list[float]]: ...

    def embed_query(
        self, text: str, *, task_type: str
    ) -> list[float]: ...


@dataclass(frozen=True)
class PineconeResources:
    index: PineconeIndex
    embedder: Embedder
    embedding_dimension: int


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_id: str
    text: str
    start: float
    end: float
    language_code: str
    is_translation: bool


@dataclass(frozen=True)
class IndexingResult:
    video_id: str
    chunks_indexed: int
    languages_indexed: list[str]
    translations_indexed: list[str] = field(default_factory=list)
    translation_providers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _split_long_segment(
    segment: TranscriptSegment, chunk_size: int
) -> list[TranscriptSegment]:
    text = segment.text.strip()
    pieces: list[TranscriptSegment] = []
    offset = 0

    while offset < len(text):
        end = min(offset + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", offset, end)
            if boundary > offset:
                end = boundary

        piece = text[offset:end].strip()
        if piece:
            start_ratio = offset / len(text)
            end_ratio = end / len(text)
            piece_start = segment.start + segment.duration * start_ratio
            piece_end = segment.start + segment.duration * end_ratio
            pieces.append(
                TranscriptSegment(
                    text=piece,
                    start=piece_start,
                    duration=piece_end - piece_start,
                )
            )
        offset = end
        while offset < len(text) and text[offset].isspace():
            offset += 1

    return pieces


def _chunk_language(
    video_id: str,
    segments: Iterable[TranscriptSegment],
    language_code: str,
    is_translation: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TranscriptChunk]:
    units = [
        piece
        for segment in segments
        for piece in _split_long_segment(segment, chunk_size)
    ]
    chunks: list[TranscriptChunk] = []
    current: list[TranscriptSegment] = []
    current_length = 0

    def save_chunk() -> None:
        if not current:
            return
        chunk_number = len(chunks)
        chunks.append(
            TranscriptChunk(
                chunk_id=(
                    f"{video_id}:{language_code}:"
                    f"{'translation' if is_translation else 'original'}:{chunk_number}"
                ),
                text=" ".join(segment.text for segment in current),
                start=min(segment.start for segment in current),
                end=max(segment.start + segment.duration for segment in current),
                language_code=language_code,
                is_translation=is_translation,
            )
        )

    for unit in units:
        separator_length = 1 if current else 0
        if current and current_length + separator_length + len(unit.text) > chunk_size:
            save_chunk()

            overlap: list[TranscriptSegment] = []
            overlap_length = 0
            for previous in reversed(current):
                added_length = len(previous.text) + (1 if overlap else 0)
                if overlap_length + added_length > chunk_overlap:
                    break
                overlap.insert(0, previous)
                overlap_length += added_length
            current = overlap
            current_length = overlap_length
            while current and current_length + 1 + len(unit.text) > chunk_size:
                removed = current.pop(0)
                current_length -= len(removed.text) + (1 if current else 0)

        separator_length = 1 if current else 0
        current.append(unit)
        current_length += separator_length + len(unit.text)

    save_chunk()
    return chunks


def chunk_transcript(
    transcript: TranscriptResponse,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TranscriptChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")

    chunks: list[TranscriptChunk] = []
    for track in transcript.tracks:
        chunks.extend(
            _chunk_language(
                video_id=transcript.video_id,
                segments=track.segments,
                language_code=track.language_code,
                is_translation=track.is_translation,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks


def pinecone_resources_from_environment() -> PineconeResources:
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
    required_variables = ("GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
    missing_variables = [
        variable for variable in required_variables if not os.getenv(variable)
    ]
    if missing_variables:
        raise IndexingConfigurationError(
            "Set the following environment variables to index or query transcripts: "
            + ", ".join(missing_variables)
        )

    try:
        embedding_dimension = int(
            os.getenv("PINECONE_EMBEDDING_DIMENSION", str(DEFAULT_EMBEDDING_DIMENSION))
        )
    except ValueError as error:
        raise IndexingConfigurationError(
            "PINECONE_EMBEDDING_DIMENSION must be an integer."
        ) from error
    if embedding_dimension <= 0:
        raise IndexingConfigurationError(
            "PINECONE_EMBEDDING_DIMENSION must be greater than zero."
        )

    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    try:
        description = pinecone.describe_index(index_name)
    except PineconeException as error:
        raise IndexingConfigurationError(
            f"Unable to access Pinecone index '{index_name}'. "
            "Check the index name, API key, and Pinecone service status."
        ) from error
    actual_dimension = description.dimension
    if actual_dimension != embedding_dimension:
        raise IndexingConfigurationError(
            f"Pinecone index '{index_name}' has dimension {actual_dimension}; "
            f"configure PINECONE_EMBEDDING_DIMENSION={actual_dimension} or "
            "create an index with the configured dimension."
        )

    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=embedding_dimension,
        api_key=os.environ["GOOGLE_API_KEY"],
    )
    return PineconeResources(
        index=pinecone.Index(host=description.host),
        embedder=embedder,
        embedding_dimension=embedding_dimension,
    )


class PineconeTranscriptIndexer:
    def __init__(
        self,
        index: PineconeIndex,
        embedder: Embedder,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self._index = index
        self._embedder = embedder
        self._embedding_dimension = embedding_dimension

    @classmethod
    def from_environment(cls) -> "PineconeTranscriptIndexer":
        resources = pinecone_resources_from_environment()
        return cls(
            index=resources.index,
            embedder=resources.embedder,
            embedding_dimension=resources.embedding_dimension,
        )

    def index_transcript(
        self, transcript: TranscriptResponse
    ) -> IndexingResult:
        chunks = chunk_transcript(transcript)

        for batch_start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + UPSERT_BATCH_SIZE]
            try:
                embeddings = self._embedder.embed_documents(
                    [chunk.text for chunk in batch],
                    task_type="RETRIEVAL_DOCUMENT",
                )
            except APIError as error:
                raise IndexingProviderError(
                    "Gemini failed to generate transcript embeddings."
                ) from error
            if len(embeddings) != len(batch):
                raise ValueError("The embedding provider returned an unexpected result count.")

            vectors: list[dict[str, object]] = []
            for chunk, values in zip(batch, embeddings, strict=True):
                if len(values) != self._embedding_dimension:
                    raise ValueError(
                        "The embedding provider returned "
                        f"{len(values)} dimensions; expected {self._embedding_dimension}."
                    )
                vectors.append(
                    {
                        "id": chunk.chunk_id,
                        "values": values,
                        "metadata": {
                            "video_id": transcript.video_id,
                            "text": chunk.text,
                            "start": chunk.start,
                            "end": chunk.end,
                            "language_code": chunk.language_code,
                            "is_translation": chunk.is_translation,
                        },
                    }
                )

            try:
                self._index.upsert(vectors=vectors, namespace=transcript.video_id)
            except PineconeException as error:
                raise IndexingProviderError(
                    "Pinecone failed to store transcript chunks."
                ) from error

        return IndexingResult(
            video_id=transcript.video_id,
            chunks_indexed=len(chunks),
            languages_indexed=list(dict.fromkeys(chunk.language_code for chunk in chunks)),
            translations_indexed=list(
                dict.fromkeys(
                    chunk.language_code for chunk in chunks if chunk.is_translation
                )
            ),
            translation_providers=list(
                dict.fromkeys(
                    track.translation_provider
                    for track in transcript.tracks
                    if track.translation_provider is not None
                )
            ),
            warnings=transcript.warnings,
        )
