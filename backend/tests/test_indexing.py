import pytest

from backend.app import indexing
from backend.app.indexing import (
    IndexingConfigurationError,
    PineconeTranscriptIndexer,
    chunk_transcript,
)
from backend.app.transcripts import (
    TranscriptResponse,
    TranscriptSegment,
    TranscriptTrack,
)


class FakeEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.documents: list[str] = []

    def embed_documents(
        self, texts: list[str], *, task_type: str
    ) -> list[list[float]]:
        assert task_type == "RETRIEVAL_DOCUMENT"
        self.documents.extend(texts)
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str, *, task_type: str) -> list[float]:
        assert task_type == "RETRIEVAL_QUERY"
        return [0.1] * self.dimension


class FakePineconeIndex:
    def __init__(self) -> None:
        self.upserts: list[tuple[list[dict[str, object]], str]] = []

    def upsert(
        self, vectors: list[dict[str, object]], namespace: str
    ) -> None:
        self.upserts.append((vectors, namespace))


def test_chunk_transcript_preserves_time_and_overlaps_text() -> None:
    transcript = TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            TranscriptTrack(
                language="English",
                language_code="en",
                is_generated=False,
                segments=[
                    TranscriptSegment(text="one two", start=0, duration=1),
                    TranscriptSegment(text="three", start=1, duration=1),
                    TranscriptSegment(text="four", start=2, duration=1),
                ],
            )
        ],
    )

    chunks = chunk_transcript(transcript, chunk_size=15, chunk_overlap=5)

    assert [chunk.text for chunk in chunks] == ["one two three", "three four"]
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0, 2), (1, 3)]
    assert [chunk.language_code for chunk in chunks] == ["en", "en"]


def test_chunk_transcript_indexes_original_and_translated_segments() -> None:
    transcript = TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            TranscriptTrack(
                language="Hindi",
                language_code="hi",
                is_generated=True,
                segments=[TranscriptSegment(text="नमस्ते दुनिया", start=0, duration=2)],
            ),
            TranscriptTrack(
                language="English (translated from Hindi)",
                language_code="en",
                is_generated=True,
                is_translation=True,
                translation_provider="gemini",
                segments=[TranscriptSegment(text="Hello world", start=0, duration=2)],
            ),
        ],
    )

    chunks = chunk_transcript(transcript)

    assert [(chunk.language_code, chunk.is_translation) for chunk in chunks] == [
        ("hi", False),
        ("en", True),
    ]


def test_chunk_transcript_splits_oversized_caption_with_timestamps() -> None:
    transcript = TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            TranscriptTrack(
                language="English",
                language_code="en",
                is_generated=False,
                segments=[
                    TranscriptSegment(
                        text="one two three four five six", start=0, duration=6
                    )
                ],
            )
        ],
    )

    chunks = chunk_transcript(transcript, chunk_size=10, chunk_overlap=2)

    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == pytest.approx(6)


def test_chunk_transcript_keeps_chunks_with_overlap_within_size_limit() -> None:
    transcript = TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            TranscriptTrack(
                language="English",
                language_code="en",
                is_generated=False,
                segments=[
                    TranscriptSegment(text="1234", start=0, duration=1),
                    TranscriptSegment(text="abcdef", start=1, duration=1),
                    TranscriptSegment(text="123456789012345", start=2, duration=1),
                ],
            )
        ],
    )

    chunks = chunk_transcript(transcript, chunk_size=15, chunk_overlap=7)

    assert all(len(chunk.text) <= 15 for chunk in chunks)
    assert [chunk.text for chunk in chunks] == ["1234 abcdef", "123456789012345"]


def test_indexer_embeds_and_upserts_to_video_namespace() -> None:
    transcript = TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            TranscriptTrack(
                language="English",
                language_code="en",
                is_generated=False,
                segments=[
                    TranscriptSegment(text="A sample caption.", start=3, duration=2)
                ],
            )
        ],
    )
    embedder = FakeEmbedder(dimension=3)
    index = FakePineconeIndex()
    indexer = PineconeTranscriptIndexer(index, embedder, embedding_dimension=3)

    result = indexer.index_transcript(transcript)

    assert result.video_id == transcript.video_id
    assert result.chunks_indexed == 1
    assert result.languages_indexed == ["en"]
    assert result.translations_indexed == []
    assert result.warnings == []
    assert embedder.documents == ["A sample caption."]
    vectors, namespace = index.upserts[0]
    assert namespace == transcript.video_id
    assert vectors[0]["metadata"] == {
        "video_id": transcript.video_id,
        "text": "A sample caption.",
        "start": 3,
        "end": 5,
        "language_code": "en",
        "is_translation": False,
    }


def test_indexer_reports_translated_track_provider() -> None:
    transcript = TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            TranscriptTrack(
                language="Hindi",
                language_code="hi",
                is_generated=True,
                segments=[TranscriptSegment(text="नमस्ते", start=0, duration=1)],
            ),
            TranscriptTrack(
                language="English (translated by Gemini)",
                language_code="en",
                is_generated=True,
                is_translation=True,
                translation_provider="gemini",
                segments=[TranscriptSegment(text="Hello", start=0, duration=1)],
            ),
        ],
    )
    indexer = PineconeTranscriptIndexer(
        FakePineconeIndex(), FakeEmbedder(dimension=3), embedding_dimension=3
    )

    result = indexer.index_transcript(transcript)

    assert result.languages_indexed == ["hi", "en"]
    assert result.translations_indexed == ["en"]
    assert result.translation_providers == ["gemini"]


def test_indexer_rejects_embedding_dimension_mismatch() -> None:
    transcript = TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            TranscriptTrack(
                language="English",
                language_code="en",
                is_generated=False,
                segments=[TranscriptSegment(text="Caption.", start=0, duration=1)],
            )
        ],
    )
    indexer = PineconeTranscriptIndexer(
        FakePineconeIndex(), FakeEmbedder(dimension=2), embedding_dimension=3
    )

    with pytest.raises(ValueError, match="expected 3"):
        indexer.index_transcript(transcript)


def test_indexer_requires_cloud_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(indexing, "DOTENV_PATH", tmp_path / ".env")
    for name in ("GOOGLE_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(IndexingConfigurationError, match="GOOGLE_API_KEY"):
        PineconeTranscriptIndexer.from_environment()


def test_indexer_loads_credentials_from_project_dotenv(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GOOGLE_API_KEY=test-google-key\n"
        "PINECONE_API_KEY=test-pinecone-key\n"
        "PINECONE_INDEX_NAME=test-index\n"
        "PINECONE_EMBEDDING_DIMENSION=3\n"
    )
    monkeypatch.setattr(indexing, "DOTENV_PATH", env_file)
    for name in (
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_EMBEDDING_DIMENSION",
    ):
        monkeypatch.delenv(name, raising=False)

    class FakeIndexDescription:
        dimension = 3
        host = "https://test-index.example"

    class FakePineconeClient:
        def __init__(self, api_key: str) -> None:
            assert api_key == "test-pinecone-key"

        def describe_index(self, name: str) -> FakeIndexDescription:
            assert name == "test-index"
            return FakeIndexDescription()

        def Index(self, host: str) -> FakePineconeIndex:
            assert host == "https://test-index.example"
            return FakePineconeIndex()

    def fake_embedder(**kwargs) -> FakeEmbedder:
        assert kwargs["api_key"] == "test-google-key"
        assert kwargs["output_dimensionality"] == 3
        return FakeEmbedder(dimension=3)

    monkeypatch.setattr(indexing, "Pinecone", FakePineconeClient)
    monkeypatch.setattr(indexing, "GoogleGenerativeAIEmbeddings", fake_embedder)

    indexer = PineconeTranscriptIndexer.from_environment()

    assert isinstance(indexer, PineconeTranscriptIndexer)
