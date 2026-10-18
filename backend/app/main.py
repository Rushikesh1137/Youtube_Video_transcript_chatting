from fastapi import FastAPI, HTTPException

from .indexing import (
    IndexingConfigurationError,
    IndexingProviderError,
    IndexingResult,
    PineconeTranscriptIndexer,
)
from .retrieval import (
    AskRequest,
    AskResponse,
    NoIndexedTranscriptError,
    PineconeTranscriptRetriever,
    RetrievalProviderError,
)
from .transcripts import (
    TranscriptProviderError,
    TranscriptRequest,
    TranscriptResponse,
    TranscriptUnavailableError,
    fetch_transcript,
    extract_video_id,
)

app = FastAPI(
    title="YouTube Chat API",
    description="Backend API for asking questions about YouTube videos.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcripts", response_model=TranscriptResponse, tags=["transcripts"])
def get_video_transcript(request: TranscriptRequest) -> TranscriptResponse:
    try:
        video_id = extract_video_id(request.video_url)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        return fetch_transcript(
            video_id=video_id,
            languages=request.languages,
            translate_to_english=request.translate_to_english,
        )
    except TranscriptUnavailableError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except TranscriptProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/videos/index", response_model=IndexingResult, tags=["indexing"])
def index_video(request: TranscriptRequest) -> IndexingResult:
    try:
        video_id = extract_video_id(request.video_url)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        indexer = PineconeTranscriptIndexer.from_environment()
        transcript = fetch_transcript(
            video_id=video_id,
            languages=request.languages,
            translate_to_english=request.translate_to_english,
        )
        return indexer.index_transcript(transcript)
    except TranscriptUnavailableError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except IndexingConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (TranscriptProviderError, IndexingProviderError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/videos/ask", response_model=AskResponse, tags=["retrieval"])
def ask_video(request: AskRequest) -> AskResponse:
    try:
        video_id = extract_video_id(request.video_url)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        retriever = PineconeTranscriptRetriever.from_environment()
        return retriever.ask(
            video_id=video_id,
            question=request.question,
            top_k=request.top_k,
        )
    except IndexingConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except NoIndexedTranscriptError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RetrievalProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
