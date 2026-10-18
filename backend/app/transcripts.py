import re
from collections.abc import Iterable
from typing import Literal, Protocol
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, Field, field_validator
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

from .translation import CaptionTranslationError, translate_hindi_texts_to_english

LanguageCode = Literal["en", "hi"]

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}


class CaptionSnippet(Protocol):
    text: str
    start: float
    duration: float


class FetchedCaptions(Protocol):
    snippets: Iterable[CaptionSnippet]


class TranslationLanguage(Protocol):
    language_code: str


class CaptionTrack(Protocol):
    language: str
    language_code: str
    is_generated: bool
    translation_languages: list[TranslationLanguage]

    def fetch(self) -> FetchedCaptions: ...

    def translate(self, language_code: str) -> "CaptionTrack": ...


class TranscriptRequest(BaseModel):
    video_url: str
    languages: list[LanguageCode] = Field(
        default_factory=lambda: ["en", "hi"], min_length=1
    )
    translate_to_english: bool = False

    @field_validator("languages")
    @classmethod
    def reject_duplicate_languages(
        cls, languages: list[LanguageCode]
    ) -> list[LanguageCode]:
        if len(languages) != len(set(languages)):
            raise ValueError("Languages must not contain duplicates.")
        return languages


class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float


class TranscriptTrack(BaseModel):
    language: str
    language_code: str
    is_generated: bool
    is_translation: bool = False
    translation_provider: Literal["youtube", "gemini"] | None = None
    segments: list[TranscriptSegment]


class TranscriptResponse(BaseModel):
    video_id: str
    tracks: list[TranscriptTrack]
    warnings: list[str] = Field(default_factory=list)


class TranscriptUnavailableError(Exception):
    pass


class TranscriptProviderError(Exception):
    pass


def extract_video_id(video_url: str) -> str:
    candidate = video_url.strip()
    if _VIDEO_ID_PATTERN.fullmatch(candidate):
        return candidate

    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in _YOUTUBE_HOSTS:
        raise ValueError("Provide a valid YouTube video URL or 11-character video ID.")

    if parsed.hostname == "youtu.be":
        video_id = parsed.path.strip("/").split("/", maxsplit=1)[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        path_parts = parsed.path.strip("/").split("/")
        video_id = (
            path_parts[1]
            if len(path_parts) >= 2 and path_parts[0] in {"embed", "live", "shorts"}
            else ""
        )

    if not _VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("The YouTube URL does not contain a valid video ID.")
    return video_id


def _to_segments(snippets: Iterable[CaptionSnippet]) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            text=snippet.text,
            start=snippet.start,
            duration=snippet.duration,
        )
        for snippet in snippets
    ]


def _youtube_block_warning(
    operation: Literal["list", "fetch", "translate"],
    error: YouTubeTranscriptApiException,
) -> str:
    gerunds = {"list": "listing", "fetch": "fetching", "translate": "translating"}
    if type(error).__name__ == "IpBlocked":
        return (
            "YouTube temporarily blocked this server while "
            f"{gerunds[operation]} captions. Try again later."
        )
    return f"YouTube could not {operation} captions. Try again later."


def fetch_transcript(
    video_id: str,
    languages: list[LanguageCode],
    translate_to_english: bool = False,
) -> TranscriptResponse:
    transcript_api = YouTubeTranscriptApi()

    try:
        available_tracks = list(transcript_api.list(video_id))
    except (TranscriptsDisabled, NoTranscriptFound) as error:
        raise TranscriptUnavailableError(
            "This video has no available captions."
        ) from error
    except YouTubeTranscriptApiException as error:
        raise TranscriptProviderError(
            _youtube_block_warning("list", error)
        ) from error

    selected_tracks: list[tuple[LanguageCode, CaptionTrack]] = []
    warnings: list[str] = []
    for language in languages:
        matches = [
            track
            for track in available_tracks
            if track.language_code.lower() == language
        ]
        if not matches:
            matches = [
                track
                for track in available_tracks
                if track.language_code.lower().split("-", maxsplit=1)[0] == language
            ]
        if matches:
            selected_tracks.append((language, matches[0]))
        else:
            label = "English" if language == "en" else "Hindi"
            warnings.append(f"No {label} captions are available for this video.")

    result_tracks: list[TranscriptTrack] = []
    fetch_failures: list[str] = []
    for language, track in selected_tracks:
        try:
            fetched = track.fetch()
        except YouTubeTranscriptApiException as error:
            label = "English" if language == "en" else "Hindi"
            fetch_failures.append(label)
            warnings.append(_youtube_block_warning("fetch", error))
            continue

        source_segments = _to_segments(fetched.snippets)
        result_tracks.append(
            TranscriptTrack(
                language=track.language,
                language_code=track.language_code,
                is_generated=track.is_generated,
                segments=source_segments,
            )
        )

        if not translate_to_english or language != "hi":
            continue
        english_targets = [
            target.language_code
            for target in track.translation_languages
            if target.language_code.lower().split("-", maxsplit=1)[0] == "en"
        ]
        native_translation_unavailable = ""
        if not english_targets:
            native_translation_unavailable = (
                "YouTube does not offer Hindi-to-English translation for this video."
            )

        translated_track_response: TranscriptTrack | None = None
        if english_targets:
            try:
                translation_track = track.translate(english_targets[0])
                translated = translation_track.fetch()
            except YouTubeTranscriptApiException as error:
                native_translation_unavailable = _youtube_block_warning(
                    "translate", error
                )
            else:
                translated_track_response = TranscriptTrack(
                    language=translation_track.language,
                    language_code=translation_track.language_code,
                    is_generated=translation_track.is_generated,
                    is_translation=True,
                    translation_provider="youtube",
                    segments=_to_segments(translated.snippets),
                )

        if translated_track_response is None:
            try:
                translated_texts = translate_hindi_texts_to_english(
                    [segment.text for segment in source_segments]
                )
            except CaptionTranslationError as error:
                warnings.append(
                    f"{native_translation_unavailable} Gemini fallback failed: "
                    f"{error} The original Hindi captions were kept."
                )
                continue

            translated_track_response = TranscriptTrack(
                language="English (translated by Gemini)",
                language_code="en",
                is_generated=True,
                is_translation=True,
                translation_provider="gemini",
                segments=[
                    TranscriptSegment(
                        text=text,
                        start=segment.start,
                        duration=segment.duration,
                    )
                    for segment, text in zip(
                        source_segments, translated_texts, strict=True
                    )
                ],
            )
            warnings.append(
                f"{native_translation_unavailable} Used Gemini for the English "
                "translation."
            )

        result_tracks.append(translated_track_response)

    if not result_tracks:
        if fetch_failures:
            raise TranscriptProviderError(
                " ".join(dict.fromkeys(warnings))
            )
        raise TranscriptUnavailableError(
            "No captions are available in the selected languages."
        )

    return TranscriptResponse(
        video_id=video_id,
        tracks=result_tracks,
        warnings=warnings,
    )
