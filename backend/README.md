# Backend

The FastAPI backend fetches YouTube captions and indexes timestamped transcript
chunks in Pinecone. Gemini generates multilingual embeddings; no vector data is
stored locally.

## Configure Pinecone and Gemini

Create a Pinecone index before using the indexing endpoint:

- Dimension: the value of `PINECONE_EMBEDDING_DIMENSION` (defaults to `768`)
- Metric: cosine

Add credentials and index settings to the project-root `.env` file. The backend
loads it automatically; environment variables already exported in the shell
take precedence.

```dotenv
GOOGLE_API_KEY=your-gemini-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name
PINECONE_EMBEDDING_DIMENSION=1024
```

The dimension must match the Pinecone index; `1024` is used here to match the
current project index.

Keep `.env` private; do not commit API keys.

Start the API from the project root:

```bash
./myven/bin/uvicorn app.main:app --app-dir backend --reload
```

Then open `http://127.0.0.1:8000/docs` and call `POST /videos/index`. The
default language selection is `["en", "hi"]`; every selected caption language
that exists on the video is fetched and indexed (not just the first match).
For example, include `"languages": ["hi"]` to index Hindi only, or set
`"translate_to_english": true` to also request an English translation of Hindi
captions. The original Hindi is retained in all cases. If YouTube does not
offer its own Hindi-to-English translation, Gemini translates the caption
segments in batches; this uses additional Gemini API calls and may incur
additional usage costs. The response reports which translation provider was
used and includes warnings if translation could not be completed. Each video's
vectors are stored in a Pinecone namespace named with that video's ID, and each
chunk retains its language and time range.

## Ask a question about an indexed video

Call `POST /videos/ask` with the video URL and question:

```json
{
  "video_url": "https://www.youtube.com/watch?v=tCMzkJxBM5c",
  "question": "What does the video say about BlackRock's assets?",
  "top_k": 5
}
```

The response includes a transcript-grounded answer and source chunks with
YouTube links that open at their transcript timestamps. Retrieval currently
asks Gemini to rewrite the question, generate up to two complementary search
queries, and route it as a fact lookup, comparison, timeline, or overview.
Each query searches the video's Pinecone namespace; duplicate matches are
merged, then Maximal Marginal Relevance (MMR) chooses relevant, diverse chunks.
The route also adjusts candidate breadth and answer guidance. Hybrid retrieval,
Gemini reranking, and MMR are applied in that order before answer generation.
Gemini reranks up to 24 hybrid candidates against the original question, scores
their relevance, then MMR selects diverse chunks for the final context.
Gemini then compresses each selected transcript source to exact, verbatim
evidence excerpts for the answer prompt. The API still returns the original
source text and timestamp for citations; sources with no useful evidence are
excluded from the answer context.

The current Pinecone index is dense-only, so the keyword side uses BM25 over
transcript text fetched in pages from that video's Pinecone namespace, then
combines dense and lexical rankings with reciprocal-rank fusion. This avoids
recreating the index and searches the full transcript for exact names and
numbers; each ask therefore reads the video's stored chunk metadata from
Pinecone before answer generation.

## Streamlit UI

Install the UI dependencies and run Streamlit from the project root in a
separate terminal:

```bash
./myven/bin/python -m pip install -r frontend/requirements.txt
./myven/bin/streamlit run frontend/streamlit_app.py
```

The UI defaults to `http://127.0.0.1:8000`; set `YOUTUBE_CHAT_API_URL` to use a
different backend address. Start FastAPI first. The sidebar defaults to
indexing both English and Hindi tracks and requesting an English translation
of Hindi captions. YouTube's translation is used when available; otherwise,
Gemini is the fallback. Translation uses additional Gemini API calls. The UI
reports which tracks were indexed, which provider translated captions, and any
missing captions or translation failures. Then ask questions; answers show
clickable source links that open the video at the cited timestamp.

Run backend and UI-client tests from the project root:

```bash
./myven/bin/python -m pytest backend/tests -q
./myven/bin/python -m pytest frontend/tests -q
```
