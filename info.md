# YouTube Chat: Core Concepts and Interview Guide

This guide explains the concepts used in the project and the engineering
problems encountered while building it. The [project README](./README.md)
covers installation and running the application; this document focuses on
understanding and explaining the design in an interview.

## 1. Project overview

YouTube Chat is a retrieval-augmented generation (RAG) application for asking
questions about a video's captions. It has a FastAPI backend, a Streamlit
frontend, Gemini for embeddings and language-model tasks, and Pinecone for
cloud vector storage.

At a high level, the application has two flows:

**Indexing flow**

```text
YouTube URL
  -> extract video ID
  -> retrieve selected caption tracks
  -> optionally translate Hindi captions to English
  -> split tracks into timestamped overlapping chunks
  -> generate embeddings
  -> upsert vectors and metadata into a video-specific Pinecone namespace
```

**Question-answering flow**

```text
Question
  -> rewrite, create alternative queries, and route the question
  -> semantic retrieval from Pinecone + BM25 keyword retrieval
  -> reciprocal-rank fusion
  -> Gemini reranking
  -> maximal marginal relevance (MMR)
  -> contextual compression
  -> grounded answer with timestamp citations
```

## 2. Core concepts and techniques

### 2.1 Document ingestion

In a RAG system, ingestion is the process of turning source material into
searchable records. Here, the source is not a PDF or website: it is one or more
YouTube caption tracks.

Each caption segment contains text and timing information. The backend
normalizes segments into typed models, preserves the video's ID and language,
and creates separate tracks for original captions and translated captions.
This structured representation supports chunking, metadata filtering, source
citations, and diagnostics when tracks are missing.

**Interview explanation:** “I treated transcript acquisition as an ingestion
pipeline, not as part of answer generation. That lets me index once and ask
multiple questions against the stored video namespace.”

### 2.2 Multilingual captions and translation

Users can request English captions, Hindi captions, or both. The backend
inspects available tracks and fetches each selected language rather than
stopping at the first match.

When Hindi-to-English translation is enabled:

1. The original Hindi track is always retained if it was fetched.
2. The backend uses YouTube's translation if the Hindi track advertises an
   English target and the translation fetch succeeds.
3. Otherwise, Gemini translates caption text in bounded batches.
4. Translation results are checked for missing, duplicate, or unexpected
   caption indices and restored to the original caption order.
5. The translated captions keep the original timing and are marked as
   translations, with the provider recorded.

The fallback has additional model usage. Importantly, it only works after the
original Hindi captions have been fetched; it cannot recover captions that
YouTube blocks or does not provide.

### 2.3 Text splitting and chunk design

Embedding a whole video transcript as one document would produce a large,
unfocused representation and make it difficult to return precise evidence.
The project therefore groups caption segments into bounded chunks.

The current chunker:

- Uses a default target of 500 characters per chunk.
- Carries up to 100 characters of overlap from the previous chunk.
- Attempts to split oversized individual caption segments at word boundaries.
- Preserves the start and end times associated with each chunk.
- Creates distinct chunk IDs for original and translated tracks.

Overlap helps retain context when a relevant phrase falls near a chunk boundary.
Timestamps allow a citation to jump to the relevant part of the video.

These are character-based settings rather than model-token counts. A possible
future improvement is tuning chunk sizes against tokenization and retrieval
quality on a representative evaluation set.

### 2.4 Embeddings

An embedding model maps text into a numeric vector such that semantically
related text tends to be near one another in vector space. This lets a user ask
a question with different wording from the transcript and still retrieve
relevant passages.

This project uses Google's `gemini-embedding-001` model. During indexing,
transcript chunks are embedded for document retrieval; at question time, search
queries are embedded for query retrieval. The configured vector dimension must
match the Pinecone index dimension.

Embeddings are not answers and do not preserve all literal details equally
well. This is one reason the retrieval pipeline also includes keyword search.

### 2.5 Vector database and metadata

Pinecone is the hosted vector database. For each video, the video ID is used as
the Pinecone namespace. This keeps one video's chunk search separate from
another video's chunks without requiring a local FAISS database.

Each vector stores metadata such as:

- Video ID.
- Original or translated caption text.
- Start and end times.
- Language code.
- Whether the chunk is translated.

The text and timing metadata are necessary for lexical retrieval, grounding,
and citations; Pinecone is not storing only opaque vectors.

The embedding dimension in the environment must match the provisioned index.
The current project index uses 1024 dimensions, while the code's generic
fallback default is 768. Configure the value to match the actual index.

### 2.6 Retrieval-augmented generation (RAG)

RAG separates knowledge lookup from language-model answer generation:

1. Retrieve passages relevant to the question.
2. Give selected passages to the language model as evidence.
3. Ask the model to answer using that evidence and cite its sources.

This improves answers about a specific video because the model is given
video-specific context at question time. It does not guarantee correctness:
retrieval can miss relevant evidence, transcripts can be wrong, and generation
can still misinterpret context. This is why the project returns sources and
validates compressed evidence.

### 2.7 Pre-retrieval: query planning and domain-aware routing

Users often ask questions in a conversational or underspecified way. The
Gemini query planner returns a structured plan that contains:

- A rewritten search query.
- Up to two alternative queries for complementary evidence.
- A route: fact lookup, comparison, timeline, or overview.

The route selects a search profile, which adjusts candidate breadth, MMR
relevance weighting, and answer guidance. For instance, an overview benefits
from broader coverage, while a factual lookup should favor direct evidence.

The original question is still used later for reranking and answering; the
rewritten queries are search aids rather than replacements for user intent.

### 2.8 During retrieval: dense search, BM25, and hybrid retrieval

**Dense semantic search** compares query embeddings against chunk embeddings.
It is useful when the user's wording differs from the transcript wording.

**BM25 keyword search** scores documents based on query-term matches, including
term frequency and document length normalization. It can be especially useful
for names, acronyms, numbers, and exact phrases.

The project combines semantic and lexical ranked results using weighted
reciprocal-rank fusion (RRF). RRF combines each result's rank rather than
assuming that scores from different retrieval systems share the same scale.
The configured dense and lexical weights are 0.65 and 0.35 respectively.

The Pinecone index is dense-only. The current hybrid implementation lists and
fetches that video's stored chunk metadata from Pinecone to build the BM25
corpus. This avoids changing the existing index type, but adds read latency and
work proportional to transcript size on each question.

### 2.9 During retrieval: reranking

Initial retrieval should favor recall: gather a reasonably broad set that is
likely to contain useful evidence. Reranking then spends more computation on a
smaller candidate set to improve relevance ordering.

Gemini scores up to 24 hybrid candidates against the user's original question.
The implementation validates that every expected chunk ID appears exactly
once and rejects invalid scores. Reranking improves ordering; it does not
create new evidence or fix missing captions.

### 2.10 During retrieval: maximal marginal relevance (MMR)

Selecting only the highest-scoring chunks can return several near-duplicates.
MMR balances:

- **Relevance:** similarity to the question.
- **Diversity:** dissimilarity from chunks already selected.

The implementation iteratively selects a candidate based on its relevance
score and its vector similarity to already selected candidates. This aims to
provide a compact set of useful, less-redundant passages.

### 2.11 Post-retrieval: contextual compression

Retrieved chunks may contain more text than is useful for a particular
question. Contextual compression asks Gemini to identify short, exact excerpts
from the selected sources.

The backend checks that each excerpt is non-empty and is an exact substring of
its corresponding source. Sources with no selected evidence are excluded from
the answer context. The original source text and timestamps are kept for the
response citations.

This is evidence selection, not a free-form summary or translation. Exact
matching helps prevent the compressor from silently inventing or paraphrasing
the quoted evidence.

### 2.12 Augmentation, prompt construction, and grounding

**Augmentation** is the step where retrieved evidence is added to the
question/model input. In this project, prompts include the user's question,
retrieved transcript text, source IDs, language, and timing as appropriate.

Prompts are constructed as explicit system/human message pairs. Instructions
tell the model to answer from transcript evidence, preserve evidence boundaries,
and treat transcript text as untrusted quoted content rather than instructions.
Structured-output schemas are used for query planning, reranking, compression,
and translation.

“Prompt templates” is a useful design concept here, but the code currently
builds message tuples directly; it does not define every prompt as a reusable
LangChain `PromptTemplate`. A future cleanup could factor stable prompt
templates while keeping task-specific inputs explicit.

**Grounding** means the answer should be supported by retrieved transcript
sources. The application supplies the selected evidence and returns citations
with each answer. This makes answers easier to inspect, but should not be
described as a formal proof that every generated claim is correct.

### 2.13 Context-window optimization

Language models have finite context limits, and larger prompts increase
latency and cost. The project controls context size by:

- Limiting the number of selected chunks (`top_k`, from 1 to 10).
- Limiting the context character budget in answer generation.
- Reranking candidates before selecting the final sources.
- Compressing selected sources into exact evidence excerpts.
- Keeping query planning and answer-generation contexts focused on their task.

Character counts are a practical guard, not an exact token count. Token-aware
budgeting could make this more precise across English and Hindi.

### 2.14 LangChain and LCEL execution

LangChain provides the Gemini chat and embedding integrations. The backend
uses model `.invoke(...)` calls for prompt messages and
`.with_structured_output(...)` for outputs validated against Pydantic models.
This is the current orchestration pattern.

**Interview accuracy matters:** the complete RAG pipeline is not currently
written as one LCEL expression composed with the `|` operator. It is an
explicit Python orchestration flow that calls planning, embedding, Pinecone,
reranking, compression, and generation in order. LCEL is relevant as a
LangChain composition model, but say that it is a possible refactor rather
than claiming that the current implementation is an LCEL chain.

### 2.15 FastAPI backend

FastAPI exposes the application through typed HTTP endpoints:

- `GET /health` checks service availability.
- `POST /transcripts` fetches selected caption tracks.
- `POST /videos/index` fetches, optionally translates, embeds, and indexes.
- `POST /videos/ask` answers a question about an indexed video.

Pydantic request and response models validate inputs and define API contracts.
The routes map expected failures to HTTP status codes, such as invalid input,
unavailable captions, provider failures, and missing configuration. The API
also produces interactive OpenAPI docs at `/docs`.

### 2.16 Streamlit frontend

Streamlit provides a lightweight UI over the API; it does not directly call
Pinecone or Gemini. Its API client centralizes HTTP requests, timeouts, and
backend error reporting. The UI lets a user choose caption languages and
translation, index a video, ask questions, and open timestamp citations.

This separation keeps the frontend focused on interaction and the backend
responsible for secrets, ingestion, retrieval, and external service calls.

### 2.17 Testing and provider boundaries

Most automated tests use fakes for YouTube, Gemini, and Pinecone. This makes
tests deterministic and checks behavior such as:

- URL parsing and API request validation.
- Selecting multiple caption languages.
- Translation fallback and preservation of original Hindi.
- Timestamp-aware chunking and translated chunk indexing.
- Embedding dimension checks.
- Retrieval fusion, reranking, compression, and citations.
- Frontend API error handling.

Mocked tests validate application logic, not live provider availability. Live
YouTube access can fail due to network/IP blocks even when captions exist; a
provider outage should not be misrepresented as a passing end-to-end test.

## 3. Problems encountered and resolutions

This section is deliberately separate from the concept notes above. These are
the concrete implementation and integration problems encountered while
building the project.


### 3.3 Pinecone vector dimension mismatch

**Symptom:** The Pinecone index dimension did not agree with the embedding
configuration.

**Cause:** The existing cloud index used 1024 dimensions, while the code's
generic default was 768.

**Resolution:** Configure `PINECONE_EMBEDDING_DIMENSION=1024` for that index
and validate the configured dimension against the Pinecone index before
embedding or querying.

**Interview takeaway:** An embedding dimension is part of the vector-index
contract. Changing the model or output dimensionality may require a compatible
index and potentially re-embedding stored content.

### 3.4 Hindi was skipped when English and Hindi were both requested

**Symptom:** The UI offered both languages, but some videos were indexed in
English only.

**Cause:** The original implementation used a “find first matching language”
operation. With English listed first, it selected English and never fetched
Hindi.

**Resolution:** Enumerate the available caption tracks and fetch each requested
language separately. Return track-specific results and warnings.

### 3.5 Translation availability differed by video

**Symptom:** The translation option did not always produce an English track,
even when the Hindi caption itself was marked translatable.

**Cause:** A track's general `is_translatable` flag does not guarantee that
English is one of the offered translation targets. Some Hindi tracks advertise
English translation; others do not.

**Resolution:** Inspect the track's actual translation target languages.
Attempt YouTube translation when English is offered; otherwise use the Gemini
fallback. Preserve Hindi and report provider or failure details either way.

### 3.6 YouTube blocks caption requests from some environments

**Symptom:** Caption fetches returned an `IpBlocked`-type error even for videos
whose caption metadata showed available English or Hindi tracks.

**Cause:** YouTube may restrict caption requests based on the requester's
network or IP address. Listing caption metadata and downloading the caption
text are distinct operations; the first may work while the second is blocked.

**Resolution:** Return a clear provider error instead of treating a blocked
fetch as proof that captions do not exist. Retrying later may work, but is not
guaranteed. Gemini translation cannot help until the original Hindi text has
been fetched. A separate permitted transcript-upload or speech-to-text path
may be needed for reliable operation.

### 3.7 Hybrid retrieval on a dense-only Pinecone index

**Constraint:** The existing Pinecone index was dense-only, but the product
needed exact-term keyword retrieval as well as semantic retrieval.

**Resolution:** Fetch a video's stored chunk metadata, run BM25 in the
application, and fuse the lexical ranking with Pinecone's dense ranking.
Pagination and batched fetches were added to support videos with more chunks.

**Trade-off:** The index does not need to be recreated, but reading the
namespace for BM25 on every question adds provider calls and latency. For
larger-scale use, consider a separately maintained lexical index or a Pinecone
configuration that supports the desired hybrid retrieval natively.

### 3.8 Incorrect result metadata during reciprocal-rank fusion

**Symptom:** Retrieval results sometimes had incorrect or repeated citation
metadata.

**Cause:** Fusion was not consistently preserving the candidate's matching
metadata when merging ranked lists.

**Resolution:** Keep a canonical match by chunk ID while accumulating its
weighted rank score, then verify fused IDs and metadata with regression tests.

### 3.9 Streamlit import path differed from the test import path

**Symptom:** Frontend tests passed, but launching the UI raised
`ModuleNotFoundError: No module named 'frontend'`.

**Cause:** Streamlit executes the selected script with a different import-path
context than pytest started from the project root.

**Resolution:** Import the API client as a sibling module from the Streamlit
script and verify by opening the running app in a browser, not only by
compiling or unit-testing it.

### 3.10 Live tests versus deterministic tests

**Symptom:** A live provider check could fail even though all local tests
passed.

**Cause:** YouTube, Gemini, and Pinecone are external services. Mocked tests
isolate our code, whereas a live check additionally depends on credentials,
quotas, network access, provider changes, and service availability.

**Resolution:** Use fake-provider unit tests for application behavior and
perform explicitly labeled live smoke tests for integration. Report provider
blocks as environmental limitations rather than hiding them or weakening the
tests.

## 4. Interview preparation

### 4.1 Short project pitch

> “I built a YouTube transcript question-answering application with a FastAPI
> backend and Streamlit UI. It ingests English and Hindi captions, optionally
> translates Hindi to English, chunks the transcript with timestamps, embeds
> the chunks with Gemini, and stores them in Pinecone. At query time it
> combines semantic retrieval and BM25, fuses and reranks candidates, applies
> MMR and evidence compression, and generates an answer with clickable
> timestamp citations. I also handled real integration issues such as vector
> dimension mismatches, multilingual track selection, translation
> availability, and YouTube caption-fetch blocking.”

### 4.2 Likely interview questions

**Why use RAG instead of sending an entire transcript to Gemini?**

RAG keeps the prompt focused on the passages relevant to a question, supports
long videos, reduces unnecessary context, and provides source citations. It
also lets us reuse indexed transcript chunks across many questions.

**Why use both vector search and BM25?**

Vector search handles semantic paraphrases. BM25 handles exact terms such as
proper nouns, acronyms, and numbers. Their scores are not directly comparable,
so the application combines their rankings with reciprocal-rank fusion.

**Why is chunk overlap useful?**

A statement can cross a chunk boundary. Overlap carries nearby text forward so
retrieval can find enough context even when a relevant phrase begins near the
end of one chunk.

**Why keep timestamps and language metadata in Pinecone?**

They let the system show where evidence came from, create clickable citations,
and distinguish original Hindi, English, and translated tracks.

**What does MMR add after reranking?**

Reranking sorts candidates by relevance. MMR then balances relevance with
diversity so the final context is less likely to contain several nearly
identical passages.

**How do you validate that contextual compression did not invent evidence?**

The compressor must return source IDs and exact excerpts. The backend checks
that each excerpt is a non-empty substring of the corresponding original
source. The original text remains available for citations.

**Does a passing test suite prove YouTube caption access works for every
video?**

No. Unit tests use fakes to validate our logic. Live caption access depends on
the video, track availability, network/IP restrictions, and YouTube's service.
Live smoke tests are a separate integration check.

**Is this implementation built as an LCEL chain?**

Not end-to-end. It uses LangChain's Gemini integrations, message-based model
invocation, and structured-output runnables, but the complete RAG flow is
explicit Python orchestration. LCEL could be used in a future refactor.

**What are the main current limitations?**

YouTube can block caption fetching, the dense-only Pinecone index requires
fetching a video's full stored text for BM25, translation and repeated Gemini
retrieval stages add latency and usage costs, and character budgets are only
an approximation of token limits.

### 4.3 Trade-offs to discuss

- **Managed vector DB vs local FAISS:** Pinecone is cloud-accessible and avoids
  managing a local index, at the cost of an external service dependency.
- **Native YouTube translation vs Gemini fallback:** Native translation is
  convenient when offered; Gemini provides broader fallback coverage but costs
  additional model calls.
- **Dense-only index plus BM25 vs native hybrid index:** The current approach
  preserves the existing index but requires fetching the stored text on each
  question.
- **More retrieval stages vs latency:** Query planning, multiple searches,
  reranking, compression, and answer generation can improve relevance and
  evidence quality, but each stage adds latency and can incur model usage.
- **Exact evidence excerpts vs free-form summaries:** Exact excerpts make
  grounding validation stronger, while a separate answer generation step is
  still responsible for explaining the evidence.

### 4.4 Improvements to propose thoughtfully

1. Add a user-provided transcript upload option for cases where YouTube blocks
   caption retrieval.
2. Evaluate a permitted and reliable speech-to-text source if captions are
   unavailable; account for audio rights, provider terms, cost, and consent.
3. Benchmark retrieval on a curated set of videos and questions, tracking
   recall, citation relevance, answer support, latency, and cost.
4. Cache or maintain a lexical index to avoid reading every chunk from
   Pinecone for every question.
5. Use token-aware context budgets and evaluate chunk size/overlap empirically.
6. Add request-level tracing and timings for transcript fetch, translation,
   embedding, retrieval, reranking, compression, and answer generation.
7. Consider LCEL composition where it improves readability or enables
   streaming, while preserving explicit error handling and testable
   dependencies.
8. Build the planned Chrome extension only after the backend's transcript
   acquisition path is dependable enough for the intended users.

## 5. Source code map

| Concern | Implementation |
| --- | --- |
| FastAPI routes and HTTP errors | [`backend/app/main.py`](./backend/app/main.py) |
| Caption retrieval and selection | [`backend/app/transcripts.py`](./backend/app/transcripts.py) |
| Gemini Hindi-to-English fallback | [`backend/app/translation.py`](./backend/app/translation.py) |
| Chunking, embeddings, Pinecone writes | [`backend/app/indexing.py`](./backend/app/indexing.py) |
| Query planning and RAG retrieval | [`backend/app/retrieval.py`](./backend/app/retrieval.py) |
| Streamlit user interface | [`frontend/streamlit_app.py`](./frontend/streamlit_app.py) |
| Setup and execution instructions | [`README.md`](./README.md) |
