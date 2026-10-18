# YouTube Chat

Ask questions about a YouTube video's captions and get answers grounded in
timestamped transcript sources. The project currently includes a FastAPI
backend and a Streamlit web interface. It supports English and Hindi captions,
cloud vector storage with Pinecone, Gemini-powered retrieval, and optional
Hindi-to-English translation.

> **Current status:** The FastAPI backend and Streamlit UI are implemented.
> A Chrome extension is planned, but is not part of the current project yet.
> YouTube may block caption requests from some networks or IP addresses; see
> [Troubleshooting](#troubleshooting).

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the application](#run-the-application)
- [Use the Streamlit interface](#use-the-streamlit-interface)
- [Use the API directly](#use-the-api-directly)
- [Run tests](#run-tests)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)

## Features

- Fetch available English and/or Hindi YouTube caption tracks.
- Index all selected caption languages available for a video.
- Keep the original Hindi captions and optionally create an English translation:
  YouTube translation is preferred when available; Gemini is used as a fallback.
- Split captions into overlapping, timestamped chunks.
- Store embeddings and caption metadata in a Pinecone namespace for each video.
- Answer questions with a retrieval-augmented generation (RAG) pipeline:
  - LLM query rewriting and multi-query generation.
  - Domain-aware routing for fact lookup, comparison, timeline, and overview.
  - Dense semantic retrieval combined with BM25 keyword retrieval.
  - Reciprocal-rank fusion, Gemini reranking, and maximal marginal relevance
    (MMR).
  - Contextual compression to select exact transcript evidence for the answer.
  - Grounded answer generation and clickable timestamp citations.
- Configure caption language preferences, translation, and source count from
  the Streamlit sidebar.

## How it works

1. Enter a YouTube URL or video ID in the Streamlit UI.
2. Choose English and/or Hindi captions and whether to request an English
   translation of Hindi captions.
3. The backend asks YouTube for the selected caption tracks. If Hindi captions
   need translation, it tries YouTube's translation first and then uses Gemini
   if that translation is not available.
4. The backend chunks the captions, creates Gemini embeddings, and stores them
   in Pinecone under a namespace named for the video ID.
5. Ask a question about the indexed video. The retrieval pipeline searches that
   video's namespace, prepares evidence, and returns an answer with timestamped
   source links.

Each video must be indexed before you can ask questions about it. Re-index a
video after changing its selected captions or translation setting.

## Requirements

- Python 3.12 recommended.
- A Google AI Studio API key for Gemini embeddings, translation, and answer
  generation.
- A Pinecone account, API key, and dense vector index.
- A network connection that can access YouTube captions. Caption access can
  vary by video, network, and IP address.

## Setup

1. Create and activate a virtual environment

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install backend and frontend dependencies

Run these commands from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install -r frontend/requirements.txt
```

The frontend dependency file includes Streamlit, Requests, and pytest.

### 3. Create a Pinecone index

Create a **dense** Pinecone index with:

- **Dimension:** `1024` (must match `PINECONE_EMBEDDING_DIMENSION` below).
- **Metric:** cosine.

The index name is your choice; use that exact name in the environment file.

### 4. Configure environment variables

Create a file named `.env` in the repository root:

```dotenv
GOOGLE_API_KEY=your-google-ai-studio-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-pinecone-index-name
PINECONE_EMBEDDING_DIMENSION=1024
```

The backend loads this file automatically. Variables already exported in the
shell take precedence. `.env` is excluded by `.gitignore`; do not commit API
keys or share them publicly.

The optional `GEMINI_CHAT_MODEL` setting selects the Gemini chat model used for
query planning, reranking, contextual compression, and answer generation. The
current default is `gemini-3.5-flash-lite`. The same setting is used for the
Gemini translation fallback.

## Run the application

Start the backend and frontend in **two separate terminals**, from the
repository root. Activate the virtual environment in both terminals first.

### Terminal 1: FastAPI

```bash
uvicorn app.main:app --app-dir backend --reload
```

The backend runs at <http://127.0.0.1:8000>. Interactive API documentation is
available at <http://127.0.0.1:8000/docs>.

### Terminal 2: Streamlit

```bash
streamlit run frontend/streamlit_app.py
```

Open the local URL printed by Streamlit (usually
<http://localhost:8501>). The UI's backend URL defaults to
`http://127.0.0.1:8000`; set `YOUTUBE_CHAT_API_URL` if the backend runs
elsewhere.

## Use the Streamlit interface

1. Optionally click **Check backend connection** in the sidebar.
2. Leave both English and Hindi selected to index both tracks when available,
   or choose only the language(s) you want.
3. Leave **Also index an English translation of Hindi captions** selected to
   try YouTube translation and use Gemini when needed. Gemini translation makes
   additional API calls and may incur additional usage costs.
4. Paste a YouTube URL or video ID and click **Index video**.
5. Review the indexing result and any warnings about missing captions or
   unavailable translations.
6. Ask a question in the chat box. Click a source link to open the video at the
   cited time.

Examples of accepted links include standard `youtube.com/watch` URLs,
`youtu.be` short links, and YouTube Shorts or embed URLs.

## Use the API directly

### Check backend health

```bash
curl http://127.0.0.1:8000/health
```

### Fetch a transcript without indexing it

```bash
curl -X POST http://127.0.0.1:8000/transcripts \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["en", "hi"],
    "translate_to_english": true
  }'
```

The response contains a `tracks` array. Each track has its language, whether
it is a translation, its translation provider (when applicable), and timestamped
segments. It also includes `warnings` when a requested track or translation
is unavailable.

### Index a video

```bash
curl -X POST http://127.0.0.1:8000/videos/index \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "languages": ["en", "hi"],
    "translate_to_english": true
  }'
```

The response reports the number of chunks indexed, caption languages,
translation providers, and warnings. For Hindi only, set `"languages": ["hi"]`.
To skip translation, set `"translate_to_english": false`.

### Ask a question about an indexed video

```bash
curl -X POST http://127.0.0.1:8000/videos/ask \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "question": "What are the main points discussed?",
    "top_k": 5
  }'
```

`top_k` can be between 1 and 10. The response includes an answer and `sources`
with the original caption text, language, timestamps, relevance scores, and
YouTube URLs that jump to the source time.

## Run tests

Run all backend and frontend tests from the repository root:

```bash
python -m pytest backend/tests frontend/tests -q
```

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | Google AI Studio key for Gemini services. |
| `PINECONE_API_KEY` | Yes | Pinecone API key. |
| `PINECONE_INDEX_NAME` | Yes | Name of the existing Pinecone dense index. |
| `PINECONE_EMBEDDING_DIMENSION` | Yes for a non-default dimension | Must match the Pinecone index. Default: `768`; this repository setup uses `1024`. |
| `GEMINI_CHAT_MODEL` | No | Gemini chat model override. Default: `gemini-3.5-flash-lite`. |
| `YOUTUBE_CHAT_API_URL` | No | FastAPI base URL used by Streamlit. Default: `http://127.0.0.1:8000`. |

The Pinecone index must match the configured embedding dimension. If these do
not match, the API reports a configuration error rather than indexing vectors
with the wrong size.

## Troubleshooting

### YouTube caption request returns HTTP 502 or `IpBlocked`

This means the backend could not fetch the caption data from YouTube. The
video may have captions, but YouTube can block requests from a particular
network or IP address. Retrying later may work, but it is not guaranteed.
Restarting FastAPI does not necessarily change the network-level restriction.

Gemini translation is only a fallback **after** the backend has fetched the
original Hindi captions; it cannot translate captions that YouTube would not
provide. If caption fetching is consistently blocked, a separate transcript
input or speech-to-text source would need to be added.

### No captions in a requested language

Not every video has both English and Hindi captions. The API and UI report
which selected tracks were missing. Choose a language that is available for
the video, or use the YouTube transcript panel to verify its caption options.

### Hindi captions are indexed but there is no English translation

Check the indexing response warnings. YouTube's translation is not available
for every Hindi track. In that case Gemini is used when the translation option
is enabled and `GOOGLE_API_KEY` is configured. If the Gemini request fails,
the original Hindi captions are still indexed and the warning explains that
translation did not complete.

### Pinecone dimension or credential errors

- Verify the `.env` variable names and that the file is in the repository root.
- Make sure the Pinecone index name and API key are correct.
- Check that `PINECONE_EMBEDDING_DIMENSION` equals the dimension of the index.
- Restart the backend after changing environment variables.

### Backend connection error in Streamlit

- Start FastAPI in its own terminal and confirm `http://127.0.0.1:8000/health`
  returns `{"status":"ok"}`.
- Confirm the FastAPI URL in the Streamlit sidebar is correct.
- Set `YOUTUBE_CHAT_API_URL` if the API is not running at the default address.

### Asking about a video that has not been indexed

Index that video first. Each video has its own Pinecone namespace, so indexing
another video does not index the requested video's captions.

## Project structure

```text
.
├── backend/
│   ├── app/
│   │   ├── indexing.py       # Transcript chunking, Gemini embeddings, Pinecone
│   │   ├── main.py           # FastAPI application and routes
│   │   ├── retrieval.py      # Hybrid RAG pipeline and answer generation
│   │   ├── transcripts.py    # YouTube captions and language handling
│   │   └── translation.py    # Batched Hindi-to-English Gemini fallback
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── api_client.py         # Streamlit-to-FastAPI HTTP client
│   ├── streamlit_app.py      # Streamlit chat interface
│   ├── tests/
│   └── requirements.txt
├── .env                      # Local credentials; create this file
├── .gitignore
└── Youtube_video_rag.ipynb   # Original experimentation notebook
```

## Roadmap

- Improve caption-fetch reliability and evaluate alternative transcript
  acquisition options where appropriate.
- Build the Chrome extension for interacting with the chat alongside a
  YouTube video.
- Continue validating caption selection, translation, and retrieval across
  more videos and languages.

