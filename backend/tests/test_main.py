from fastapi.testclient import TestClient

from backend.app import main
from backend.app import transcripts
from backend.app.indexing import IndexingResult
from backend.app.retrieval import AskResponse, SourceCitation

client = TestClient(main.app)


class FakeSnippet:
    def __init__(self, text: str, start: float, duration: float) -> None:
        self.text = text
        self.start = start
        self.duration = duration


class FakeFetchedTranscript:
    def __init__(self, text: str) -> None:
        self.snippets = [FakeSnippet(text=text, start=1.5, duration=2.0)]


class FakeTranslationLanguage:
    def __init__(self, language_code: str) -> None:
        self.language_code = language_code


class FakeTranscript:
    language = "Hindi (auto-generated)"
    language_code = "hi"
    is_generated = True
    is_translatable = True
    translation_languages = [FakeTranslationLanguage("en")]

    def __init__(self) -> None:
        self.translated_to: str | None = None

    def fetch(self) -> FakeFetchedTranscript:
        return FakeFetchedTranscript("नमस्ते")

    def translate(self, language_code: str) -> "FakeTranscript":
        self.translated_to = language_code
        return TranslatedFakeTranscript()


class FakeEnglishTranscript(FakeTranscript):
    language = "English (auto-generated)"
    language_code = "en"

    def fetch(self) -> FakeFetchedTranscript:
        return FakeFetchedTranscript("Hello")


class TranslatedFakeTranscript(FakeTranscript):
    language = "English"
    language_code = "en"

    def fetch(self) -> FakeFetchedTranscript:
        return FakeFetchedTranscript("Hello")


class FakeTranscriptList:
    def __init__(self, tracks: list[FakeTranscript]) -> None:
        self.tracks = tracks

    def __iter__(self):
        return iter(self.tracks)


class FakeTranscriptApi:
    def __init__(self, transcript_list: FakeTranscriptList) -> None:
        self.transcript_list = transcript_list

    def list(self, video_id: str) -> FakeTranscriptList:
        assert video_id == "tCMzkJxBM5c"
        return self.transcript_list


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_video_id_accepts_watch_short_and_shorts_urls() -> None:
    assert (
        transcripts.extract_video_id("https://www.youtube.com/watch?v=tCMzkJxBM5c")
        == "tCMzkJxBM5c"
    )
    assert transcripts.extract_video_id("https://youtu.be/tCMzkJxBM5c") == "tCMzkJxBM5c"
    assert (
        transcripts.extract_video_id("https://youtube.com/shorts/tCMzkJxBM5c")
        == "tCMzkJxBM5c"
    )


def test_extract_video_id_rejects_non_youtube_url() -> None:
    try:
        transcripts.extract_video_id("https://example.com/watch?v=tCMzkJxBM5c")
    except ValueError as error:
        assert "valid YouTube" in str(error)
    else:
        raise AssertionError("Expected a non-YouTube URL to be rejected.")


def test_transcript_endpoint_preserves_hindi_and_optionally_translates(
    monkeypatch,
) -> None:
    transcript = FakeTranscript()
    transcript_list = FakeTranscriptList([transcript])
    monkeypatch.setattr(
        transcripts,
        "YouTubeTranscriptApi",
        lambda: FakeTranscriptApi(transcript_list),
    )

    response = client.post(
        "/transcripts",
        json={
            "video_url": "https://youtu.be/tCMzkJxBM5c",
            "languages": ["hi"],
            "translate_to_english": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["tracks"] == [
        {
            "language": "Hindi (auto-generated)",
            "language_code": "hi",
            "is_generated": True,
            "is_translation": False,
            "translation_provider": None,
            "segments": [{"text": "नमस्ते", "start": 1.5, "duration": 2.0}],
        },
        {
            "language": "English",
            "language_code": "en",
            "is_generated": True,
            "is_translation": True,
            "translation_provider": "youtube",
            "segments": [{"text": "Hello", "start": 1.5, "duration": 2.0}],
        },
    ]
    assert data["warnings"] == []


def test_transcript_endpoint_fetches_each_selected_language(monkeypatch) -> None:
    transcript_list = FakeTranscriptList(
        [FakeEnglishTranscript(), FakeTranscript()]
    )
    monkeypatch.setattr(
        transcripts,
        "YouTubeTranscriptApi",
        lambda: FakeTranscriptApi(transcript_list),
    )

    response = client.post(
        "/transcripts",
        json={
            "video_url": "https://youtu.be/tCMzkJxBM5c",
            "languages": ["en", "hi"],
            "translate_to_english": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert [track["language_code"] for track in data["tracks"]] == ["en", "hi"]
    assert data["warnings"] == []


def test_transcript_endpoint_warns_when_translation_is_unavailable(
    monkeypatch,
) -> None:
    transcript = FakeTranscript()
    transcript.translation_languages = []
    monkeypatch.setattr(
        transcripts,
        "translate_hindi_texts_to_english",
        lambda texts: ["Hello"],
    )
    monkeypatch.setattr(
        transcripts,
        "YouTubeTranscriptApi",
        lambda: FakeTranscriptApi(FakeTranscriptList([transcript])),
    )

    response = client.post(
        "/transcripts",
        json={
            "video_url": "https://youtu.be/tCMzkJxBM5c",
            "languages": ["hi"],
            "translate_to_english": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert [(track["language_code"], track["is_translation"]) for track in data["tracks"]] == [
        ("hi", False),
        ("en", True),
    ]
    assert data["tracks"][1]["translation_provider"] == "gemini"
    assert "does not offer Hindi-to-English translation" in data["warnings"][0]
    assert "Used Gemini" in data["warnings"][0]


def test_transcript_endpoint_keeps_hindi_if_gemini_fallback_fails(
    monkeypatch,
) -> None:
    transcript = FakeTranscript()
    transcript.translation_languages = []

    def failed_translation(texts: list[str]) -> list[str]:
        raise transcripts.CaptionTranslationError("Gemini is temporarily unavailable.")

    monkeypatch.setattr(
        transcripts, "translate_hindi_texts_to_english", failed_translation
    )
    monkeypatch.setattr(
        transcripts,
        "YouTubeTranscriptApi",
        lambda: FakeTranscriptApi(FakeTranscriptList([transcript])),
    )

    response = client.post(
        "/transcripts",
        json={
            "video_url": "https://youtu.be/tCMzkJxBM5c",
            "languages": ["hi"],
            "translate_to_english": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert [(track["language_code"], track["is_translation"]) for track in data["tracks"]] == [
        ("hi", False)
    ]
    assert "Gemini fallback failed" in data["warnings"][0]
    assert "original Hindi captions were kept" in data["warnings"][0]


def test_transcript_endpoint_rejects_invalid_url() -> None:
    response = client.post(
        "/transcripts",
        json={"video_url": "https://example.com/video"},
    )

    assert response.status_code == 422


def test_transcript_endpoint_reports_unavailable_captions(monkeypatch) -> None:
    def unavailable(**kwargs):
        assert kwargs["video_id"] == "tCMzkJxBM5c"
        raise transcripts.TranscriptUnavailableError(
            "No captions are available in the requested languages."
        )

    monkeypatch.setattr(main, "fetch_transcript", unavailable)
    response = client.post(
        "/transcripts",
        json={"video_url": "https://youtu.be/tCMzkJxBM5c"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "No captions are available in the requested languages."
    )


def test_video_index_endpoint_fetches_and_indexes_transcript(monkeypatch) -> None:
    transcript = transcripts.TranscriptResponse(
        video_id="tCMzkJxBM5c",
        tracks=[
            transcripts.TranscriptTrack(
                language="Hindi",
                language_code="hi",
                is_generated=True,
                segments=[
                    transcripts.TranscriptSegment(
                        text="नमस्ते", start=0, duration=1
                    )
                ],
            )
        ],
    )
    indexed: list[transcripts.TranscriptResponse] = []

    class FakeIndexer:
        @classmethod
        def from_environment(cls) -> "FakeIndexer":
            return cls()

        def index_transcript(
            self, value: transcripts.TranscriptResponse
        ) -> IndexingResult:
            indexed.append(value)
            return IndexingResult(
                video_id=value.video_id,
                chunks_indexed=1,
                languages_indexed=["hi"],
                translation_providers=[],
                warnings=value.warnings,
            )

    def fake_fetch_transcript(**kwargs):
        assert kwargs["video_id"] == "tCMzkJxBM5c"
        assert kwargs["languages"] == ["hi", "en"]
        return transcript

    monkeypatch.setattr(main, "fetch_transcript", fake_fetch_transcript)
    monkeypatch.setattr(main, "PineconeTranscriptIndexer", FakeIndexer)

    response = client.post(
        "/videos/index",
        json={
            "video_url": "https://youtu.be/tCMzkJxBM5c",
            "languages": ["hi", "en"],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "video_id": "tCMzkJxBM5c",
        "chunks_indexed": 1,
        "languages_indexed": ["hi"],
        "translations_indexed": [],
        "translation_providers": [],
        "warnings": [],
    }
    assert indexed == [transcript]


def test_video_ask_endpoint_returns_grounded_answer(monkeypatch) -> None:
    class FakeRetriever:
        @classmethod
        def from_environment(cls) -> "FakeRetriever":
            return cls()

        def ask(self, video_id: str, question: str, top_k: int) -> AskResponse:
            assert video_id == "tCMzkJxBM5c"
            assert question == "How much assets?"
            assert top_k == 3
            return AskResponse(
                video_id=video_id,
                answer="The transcript says over $15 trillion [1].",
                sources=[
                    SourceCitation(
                        chunk_id="chunk-1",
                        text="Over 15 trillion dollars in assets.",
                        start=62,
                        end=69,
                        language_code="en",
                        score=0.91,
                        url="https://www.youtube.com/watch?v=tCMzkJxBM5c&t=62s",
                    )
                ],
            )

    monkeypatch.setattr(main, "PineconeTranscriptRetriever", FakeRetriever)

    response = client.post(
        "/videos/ask",
        json={
            "video_url": "https://youtu.be/tCMzkJxBM5c",
            "question": "How much assets?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "The transcript says over $15 trillion [1]."
    assert response.json()["sources"][0]["url"].endswith("&t=62s")


def test_video_ask_endpoint_rejects_invalid_video_url() -> None:
    response = client.post(
        "/videos/ask",
        json={
            "video_url": "https://example.com/video",
            "question": "What is this video about?",
        },
    )

    assert response.status_code == 422
