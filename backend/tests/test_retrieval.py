from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableSequence

from backend.app.retrieval import (
    AskRequest,
    CandidateScore,
    CompressionResult,
    CompressedSource,
    ContextCompressionError,
    GeminiContextualCompressor,
    GeminiQueryPlanner,
    GeminiTranscriptReranker,
    LexicalDocument,
    NoIndexedTranscriptError,
    PineconeTranscriptRetriever,
    QueryPlan,
    RerankResult,
    RerankingError,
    _fetch_lexical_corpus,
    bm25_search,
    reciprocal_rank_fusion,
    select_mmr_matches,
)


class FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.queries: list[str] = []

    def embed_documents(
        self, texts: list[str], *, task_type: str
    ) -> list[list[float]]:
        assert isinstance(texts, list)
        assert task_type == "RETRIEVAL_DOCUMENT"
        raise AssertionError("Document embedding is not used during retrieval.")

    def embed_query(self, text: str, *, task_type: str) -> list[float]:
        assert isinstance(text, str)
        assert task_type == "RETRIEVAL_QUERY"
        self.queries.append(text)
        return self.vector


class FakePineconeIndex:
    def __init__(
        self,
        matches: list[SimpleNamespace],
        corpus: list[SimpleNamespace] | None = None,
    ) -> None:
        self.matches = matches
        self.corpus = corpus if corpus is not None else matches
        self.query_arguments: list[dict[str, object]] = []
        self.fetched_batches: list[list[str]] = []

    def upsert(self, vectors: list[dict[str, object]], namespace: str) -> None:
        assert not vectors
        assert namespace
        raise AssertionError("Upsert is not used during retrieval.")

    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str,
        include_metadata: bool,
        include_values: bool,
    ) -> SimpleNamespace:
        self.query_arguments.append(
            {
                "vector": vector,
                "top_k": top_k,
                "namespace": namespace,
                "include_metadata": include_metadata,
                "include_values": include_values,
            }
        )
        return SimpleNamespace(matches=self.matches)

    def list_paginated(
        self,
        *,
        namespace: str,
        limit: int,
        pagination_token: str | None = None,
    ) -> SimpleNamespace:
        assert namespace
        assert limit > 0
        offset = int(pagination_token or 0)
        page = self.corpus[offset : offset + limit]
        next_offset = offset + limit
        return SimpleNamespace(
            vectors=[SimpleNamespace(id=vector.id) for vector in page],
            pagination=(
                {"next": str(next_offset)}
                if next_offset < len(self.corpus)
                else None
            ),
        )

    def fetch(self, *, ids: list[str], namespace: str) -> SimpleNamespace:
        assert namespace
        self.fetched_batches.append(ids)
        return SimpleNamespace(
            vectors={
                vector.id: SimpleNamespace(
                    metadata=vector.metadata,
                    values=vector.values,
                )
                for vector in self.corpus
                if vector.id in ids
            }
        )


class FakeQueryPlanner:
    def __init__(self, plan: QueryPlan) -> None:
        self.plan_result = plan
        self.questions: list[str] = []

    def plan(self, question: str) -> QueryPlan:
        self.questions.append(question)
        return self.plan_result


class FakeReranker:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.questions: list[str] = []
        self.candidate_ids: list[list[str]] = []
        self.scores = scores

    def rerank(self, question, candidates):
        self.questions.append(question)
        self.candidate_ids.append([candidate.id for candidate in candidates])
        if self.scores is not None:
            return [
                SimpleNamespace(
                    id=candidate.id,
                    score=self.scores[candidate.id],
                    values=candidate.values,
                    metadata=candidate.metadata,
                )
                for candidate in sorted(
                    candidates,
                    key=lambda candidate: self.scores[candidate.id],
                    reverse=True,
                )
            ]
        return list(candidates)


class FakeCompressor:
    def __init__(self, excerpts: dict[str, list[str]] | None = None) -> None:
        self.excerpts = excerpts
        self.questions: list[str] = []
        self.source_ids: list[list[str]] = []

    def compress(self, question, sources):
        self.questions.append(question)
        self.source_ids.append([source.chunk_id for source in sources])
        return [
            CompressedSource(
                chunk_id=source.chunk_id,
                excerpts=(
                    self.excerpts[source.chunk_id]
                    if self.excerpts is not None
                    else [source.text]
                ),
            )
            for source in sources
        ]


class FakeChatModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[tuple[str, str]] | None = None

    def invoke(self, messages: list[tuple[str, str]]) -> AIMessage:
        self.messages = messages
        return AIMessage(content=self.content)


def make_match(
    match_id: str,
    text: str,
    start: float,
    end: float,
    vector: list[float],
    score: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=match_id,
        score=score,
        values=vector,
        metadata={
            "text": text,
            "start": start,
            "end": end,
            "language_code": "hi",
        },
    )


def test_mmr_prefers_relevant_but_diverse_chunks() -> None:
    query = [1.0, 0.0]
    matches = [
        make_match("first", "First topic", 0, 2, [1.0, 0.0], 0.95),
        make_match("duplicate", "Same topic", 2, 4, [0.99, 0.01], 0.94),
        make_match("diverse", "Related second topic", 4, 6, [0.0, 1.0], 0.80),
    ]

    selected = select_mmr_matches(matches, query, limit=2)

    assert [match.id for match in selected] == ["first", "diverse"]


def test_bm25_ranks_exact_names_and_numbers_for_english_and_hindi() -> None:
    documents = [
        LexicalDocument(
            id="english-hit",
            text="BlackRock manages 15 trillion dollars in assets.",
            values=[1.0, 0.0],
            metadata={},
        ),
        LexicalDocument(
            id="english-other",
            text="A different company manages investment portfolios.",
            values=[0.0, 1.0],
            metadata={},
        ),
        LexicalDocument(
            id="hindi-hit",
            text="ब्लैकरॉक के पास 15 ट्रिलियन डॉलर की संपत्ति है।",
            values=[0.0, 1.0],
            metadata={},
        ),
    ]

    english_results = bm25_search("BlackRock 15 trillion", documents, limit=3)
    hindi_results = bm25_search("ब्लैकरॉक 15 ट्रिलियन", documents, limit=3)

    assert english_results[0][0].id == "english-hit"
    assert hindi_results[0][0].id == "hindi-hit"


def test_hybrid_reciprocal_rank_fusion_includes_lexical_only_matches() -> None:
    dense_match = make_match(
        "semantic",
        "Semantically relevant result.",
        0,
        1,
        [1.0, 0.0],
        0.9,
    )
    lexical_match = make_match(
        "exact",
        "Exact named entity match.",
        2,
        3,
        [0.0, 1.0],
        0.2,
    )

    fused = reciprocal_rank_fusion(
        [
            (0.65, [dense_match]),
            (0.35, [lexical_match]),
        ]
    )

    assert {match.id for match in fused} == {"semantic", "exact"}
    assert fused[0].score == pytest.approx(1.0)


def test_hybrid_reciprocal_rank_fusion_preserves_each_match_metadata() -> None:
    first = make_match("first", "First source.", 1, 2, [1.0, 0.0], 0.9)
    second = make_match("second", "Second source.", 3, 4, [0.0, 1.0], 0.8)

    fused = reciprocal_rank_fusion([(0.65, [first, second])])

    assert [(match.id, match.metadata["text"]) for match in fused] == [
        ("first", "First source."),
        ("second", "Second source."),
    ]


def test_lexical_corpus_fetches_all_paginated_video_chunks() -> None:
    corpus = [
        make_match(
            f"chunk-{index}",
            f"Transcript chunk number {index}.",
            float(index),
            float(index + 1),
            [1.0, 0.0],
            0.5,
        )
        for index in range(205)
    ]
    index = FakePineconeIndex([], corpus=corpus)

    documents = _fetch_lexical_corpus(index, "tCMzkJxBM5c")

    assert len(documents) == 205
    assert [len(batch) for batch in index.fetched_batches] == [100, 100, 5]


def test_retriever_returns_answer_with_timestamped_sources() -> None:
    query_vector = [1.0, 0.0]
    index = FakePineconeIndex(
        [
            make_match(
                "tCMzkJxBM5c:hi:original:0",
                "ब्लैकरॉक के पास 15 ट्रिलियन डॉलर हैं।",
                65.0,
                69.0,
                query_vector,
                0.91,
            )
        ]
    )
    embedder = FakeEmbedder(query_vector)
    chat_model = FakeChatModel("The transcript says $15 trillion [1].")
    retriever = PineconeTranscriptRetriever(
        index=index,
        embedder=embedder,
        chat_model=chat_model,
        query_planner=FakeQueryPlanner(
            QueryPlan(
                route="fact_lookup",
                rewritten_query="BlackRock assets amount",
            )
        ),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        embedding_dimension=2,
    )

    response = retriever.ask(
        video_id="tCMzkJxBM5c",
        question="ब्लैकरॉक के पास कितनी संपत्ति है?",
    )

    assert response.answer == "The transcript says $15 trillion [1]."
    assert response.sources[0].language_code == "hi"
    assert response.sources[0].start == 65
    assert response.sources[0].url.endswith("&t=65s")
    assert embedder.queries == ["BlackRock assets amount"]
    assert index.query_arguments == [
        {
            "vector": query_vector,
            "top_k": 20,
            "namespace": "tCMzkJxBM5c",
            "include_metadata": True,
            "include_values": True,
        }
    ]
    assert index.fetched_batches == [["tCMzkJxBM5c:hi:original:0"]]
    assert chat_model.messages is not None
    assert "[1]" in chat_model.messages[1][1]


def test_retriever_reports_video_with_no_indexed_chunks() -> None:
    retriever = PineconeTranscriptRetriever(
        index=FakePineconeIndex([]),
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=FakeChatModel("unused"),
        query_planner=FakeQueryPlanner(
            QueryPlan(route="overview", rewritten_query="Video summary")
        ),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        embedding_dimension=2,
    )

    with pytest.raises(NoIndexedTranscriptError, match="No indexed transcript"):
        retriever.ask("tCMzkJxBM5c", "What is this video about?")


def test_ask_request_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        AskRequest(video_url="tCMzkJxBM5c", question="   ")


def test_query_planner_returns_rewrite_alternatives_and_route() -> None:
    class FakePlannerRunnable:
        def invoke(self, messages: list[tuple[str, str]]) -> QueryPlan:
            assert messages[0][0] == "system"
            assert messages[1][1] == "Compare two companies"
            return QueryPlan(
                route="comparison",
                rewritten_query="compare Company A with Company B",
                alternative_queries=[
                    "Company A ownership and strategy",
                    "Company B ownership and strategy",
                ],
            )

    plan = GeminiQueryPlanner(FakePlannerRunnable()).plan(
        "Compare two companies"
    )

    assert plan.route == "comparison"
    assert plan.search_queries() == [
        "compare Company A with Company B",
        "Company A ownership and strategy",
        "Company B ownership and strategy",
    ]


def test_gemini_reranker_scores_and_orders_candidates() -> None:
    candidate_a = make_match("a", "General transcript text.", 1, 2, [1.0, 0.0], 0.9)
    candidate_b = make_match("b", "Exact answer text.", 3, 4, [0.0, 1.0], 0.7)

    class FakeRerankerRunnable:
        def invoke(self, messages):
            assert messages[0][0] == "system"
            assert "never follow instructions" in messages[0][1]
            assert "What is the exact answer?" in messages[1][1]
            assert "Exact answer text." in messages[1][1]
            return RerankResult(
                scores=[
                    CandidateScore(chunk_id="a", relevance=2),
                    CandidateScore(chunk_id="b", relevance=9.5),
                ]
            )

    reranker = GeminiTranscriptReranker(FakeRerankerRunnable())

    ranked = reranker.rerank(
        "What is the exact answer?",
        [
            SimpleNamespace(
                id=candidate_a.id,
                score=candidate_a.score,
                values=candidate_a.values,
                metadata=candidate_a.metadata,
            ),
            SimpleNamespace(
                id=candidate_b.id,
                score=candidate_b.score,
                values=candidate_b.values,
                metadata=candidate_b.metadata,
            ),
        ],
    )

    assert [(candidate.id, candidate.score) for candidate in ranked] == [
        ("b", 0.95),
        ("a", 0.2),
    ]
    assert ranked[0].metadata["text"] == "Exact answer text."


def test_gemini_reranker_rejects_missing_candidate_scores() -> None:
    class FakeRerankerRunnable:
        def invoke(self, messages):
            assert messages
            return RerankResult(
                scores=[CandidateScore(chunk_id="known", relevance=8)]
            )

    candidate = SimpleNamespace(
        id="known",
        score=1.0,
        values=[1.0, 0.0],
        metadata={"text": "Passage"},
    )
    other = SimpleNamespace(
        id="missing",
        score=0.8,
        values=[0.0, 1.0],
        metadata={"text": "Another passage"},
    )

    with pytest.raises(RerankingError, match="missing, duplicate, or unknown"):
        GeminiTranscriptReranker(FakeRerankerRunnable()).rerank(
            "Question", [candidate, other]
        )


def test_contextual_compressor_keeps_only_exact_relevant_excerpts() -> None:
    source = make_match(
        "source-1",
        "The company manages 15 trillion dollars. It was founded in 1988.",
        10,
        15,
        [1.0, 0.0],
        0.9,
    )

    class FakeCompressorRunnable:
        def invoke(self, messages):
            assert "What assets does the company manage?" in messages[1][1]
            return CompressionResult(
                sources=[
                    CompressedSource(
                        chunk_id="source-1",
                        excerpts=["manages 15 trillion dollars"],
                    )
                ]
            )

    compressor = GeminiContextualCompressor(FakeCompressorRunnable())
    compressed = compressor.compress(
        "What assets does the company manage?",
        [
            SimpleNamespace(
                chunk_id=source.id,
                text=source.metadata["text"],
                start=source.metadata["start"],
                end=source.metadata["end"],
                language_code="en",
                score=source.score,
                url="https://youtube.com/watch?v=tCMzkJxBM5c&t=10s",
            )
        ],
    )

    assert compressed[0].excerpts == ["manages 15 trillion dollars"]


def test_contextual_compressor_rejects_non_verbatim_content() -> None:
    class FakeCompressorRunnable:
        def invoke(self, messages):
            assert messages
            return CompressionResult(
                sources=[
                    CompressedSource(
                        chunk_id="source-1",
                        excerpts=["The company manages 15 trillion dollars."],
                    )
                ]
            )

    source = SimpleNamespace(
        chunk_id="source-1",
        text="Company manages 15 trillion dollars.",
        start=1,
        end=2,
        language_code="en",
        score=1,
        url="https://youtube.com",
    )

    with pytest.raises(ContextCompressionError, match="exact passage"):
        GeminiContextualCompressor(FakeCompressorRunnable()).compress(
            "Question", [source]
        )


@pytest.mark.parametrize(
    ("route", "candidate_count"),
    [
        ("fact_lookup", 20),
        ("comparison", 30),
        ("timeline", 25),
        ("overview", 25),
    ],
)
def test_query_route_selects_retrieval_profile(route, candidate_count) -> None:
    index = FakePineconeIndex(
        [
            make_match(
                "chunk-1",
                "Transcript excerpt",
                0,
                1,
                [1.0, 0.0],
                0.9,
            )
        ]
    )
    retriever = PineconeTranscriptRetriever(
        index=index,
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=FakeChatModel("Transcript answer [1]."),
        query_planner=FakeQueryPlanner(
            QueryPlan(route=route, rewritten_query="rewritten search")
        ),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        embedding_dimension=2,
    )

    retriever.ask("tCMzkJxBM5c", "What happened?", top_k=5)

    assert index.query_arguments[0]["top_k"] == candidate_count


def test_rag_pipeline_is_composed_as_an_lcel_sequence() -> None:
    retriever = PineconeTranscriptRetriever(
        index=FakePineconeIndex(
            [
                make_match(
                    "chunk-1",
                    "Transcript excerpt",
                    0,
                    1,
                    [1.0, 0.0],
                    0.9,
                )
            ]
        ),
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=FakeChatModel("Transcript answer [1]."),
        query_planner=FakeQueryPlanner(
            QueryPlan(route="fact_lookup", rewritten_query="transcript")
        ),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        embedding_dimension=2,
    )

    assert isinstance(retriever.chain, RunnableSequence)
    assert retriever.ask("tCMzkJxBM5c", "What is said?").answer == (
        "Transcript answer [1]."
    )


def test_multi_query_retrieval_merges_duplicate_chunks() -> None:
    matches = [
        make_match(
            "shared",
            "A shared transcript chunk.",
            10,
            12,
            [1.0, 0.0],
            0.9,
        )
    ]
    index = FakePineconeIndex(matches)
    embedder = FakeEmbedder([1.0, 0.0])
    planner = FakeQueryPlanner(
        QueryPlan(
            route="comparison",
            rewritten_query="compare Alpha and Beta",
            alternative_queries=["Alpha evidence", "Beta evidence"],
        )
    )
    retriever = PineconeTranscriptRetriever(
        index=index,
        embedder=embedder,
        chat_model=FakeChatModel("Alpha and Beta are discussed [1]."),
        query_planner=planner,
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        embedding_dimension=2,
    )

    response = retriever.ask(
        "tCMzkJxBM5c", "Compare Alpha and Beta.", top_k=1
    )

    assert embedder.queries == [
        "compare Alpha and Beta",
        "Alpha evidence",
        "Beta evidence",
    ]
    assert len(index.query_arguments) == 3
    assert index.query_arguments[0]["top_k"] == 6
    assert [source.chunk_id for source in response.sources] == ["shared"]


def test_hybrid_retrieval_finds_exact_match_outside_dense_candidates() -> None:
    dense_candidate = make_match(
        "semantic",
        "A general discussion of asset management.",
        10,
        15,
        [1.0, 0.0],
        0.9,
    )
    lexical_candidate = make_match(
        "exact",
        "BlackRock manages exactly 15 trillion dollars.",
        50,
        55,
        [0.0, 1.0],
        0.2,
    )
    index = FakePineconeIndex(
        [dense_candidate],
        corpus=[dense_candidate, lexical_candidate],
    )
    retriever = PineconeTranscriptRetriever(
        index=index,
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=FakeChatModel("BlackRock manages 15 trillion dollars [1]."),
        query_planner=FakeQueryPlanner(
            QueryPlan(
                route="fact_lookup",
                rewritten_query="BlackRock 15 trillion",
            )
        ),
        reranker=FakeReranker(),
        compressor=FakeCompressor(),
        embedding_dimension=2,
    )

    response = retriever.ask(
        "tCMzkJxBM5c",
        "What exact amount does BlackRock manage?",
        top_k=2,
    )

    assert {source.chunk_id for source in response.sources} == {"semantic", "exact"}
    exact_source = next(source for source in response.sources if source.chunk_id == "exact")
    assert exact_source.url.endswith("&t=50s")


def test_retriever_applies_reranker_before_mmr_and_generation() -> None:
    candidates = [
        make_match("semantic", "General topic context.", 10, 15, [1.0, 0.0], 0.9),
        make_match(
            "specific",
            "The specific answer to the requested question.",
            50,
            55,
            [0.0, 1.0],
            0.8,
        ),
    ]
    reranker = FakeReranker({"semantic": 0.2, "specific": 0.95})
    chat_model = FakeChatModel("The specific answer is supported [1].")
    retriever = PineconeTranscriptRetriever(
        index=FakePineconeIndex(candidates),
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=chat_model,
        query_planner=FakeQueryPlanner(
            QueryPlan(route="fact_lookup", rewritten_query="specific answer")
        ),
        reranker=reranker,
        compressor=FakeCompressor(),
        embedding_dimension=2,
    )

    response = retriever.ask(
        "tCMzkJxBM5c", "What is the specific answer?", top_k=1
    )

    assert reranker.questions == ["What is the specific answer?"]
    assert set(reranker.candidate_ids[0]) == {"semantic", "specific"}
    assert response.sources[0].chunk_id == "specific"
    assert response.sources[0].start == 50
    assert chat_model.messages is not None
    assert "The specific answer to the requested question." in chat_model.messages[1][1]


def test_compressed_evidence_is_used_for_generation_but_citations_keep_original(
) -> None:
    original_text = (
        "The company manages 15 trillion dollars in assets. "
        "The interview was recorded in a studio."
    )
    match = make_match(
        "source-1",
        original_text,
        62,
        69,
        [1.0, 0.0],
        0.9,
    )
    chat_model = FakeChatModel("The company manages 15 trillion dollars [1].")
    compressor = FakeCompressor(
        {"source-1": ["manages 15 trillion dollars in assets"]}
    )
    retriever = PineconeTranscriptRetriever(
        index=FakePineconeIndex([match]),
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=chat_model,
        query_planner=FakeQueryPlanner(
            QueryPlan(route="fact_lookup", rewritten_query="company assets")
        ),
        reranker=FakeReranker(),
        compressor=compressor,
        embedding_dimension=2,
    )

    response = retriever.ask(
        "tCMzkJxBM5c", "How much assets does the company manage?"
    )

    assert compressor.questions == ["How much assets does the company manage?"]
    assert chat_model.messages is not None
    answer_context = chat_model.messages[1][1]
    assert "manages 15 trillion dollars in assets" in answer_context
    assert "interview was recorded in a studio" not in answer_context
    assert response.sources[0].text == original_text
    assert response.sources[0].start == 62


def test_contextual_compression_fails_explicitly_when_no_evidence_remains() -> None:
    match = make_match(
        "source-1",
        "An unrelated transcript passage.",
        1,
        2,
        [1.0, 0.0],
        0.9,
    )
    retriever = PineconeTranscriptRetriever(
        index=FakePineconeIndex([match]),
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=FakeChatModel("unused"),
        query_planner=FakeQueryPlanner(
            QueryPlan(route="fact_lookup", rewritten_query="requested fact")
        ),
        reranker=FakeReranker(),
        compressor=FakeCompressor({"source-1": []}),
        embedding_dimension=2,
    )

    with pytest.raises(NoIndexedTranscriptError, match="no transcript evidence"):
        retriever.ask("tCMzkJxBM5c", "What is the requested fact?")


def test_contextual_compression_excludes_sources_without_relevant_evidence() -> None:
    first = make_match("first", "First unrelated passage.", 1, 2, [1.0, 0.0], 0.9)
    second = make_match("second", "Second relevant passage.", 3, 4, [0.0, 1.0], 0.8)
    chat_model = FakeChatModel("The relevant passage supports this [1].")
    retriever = PineconeTranscriptRetriever(
        index=FakePineconeIndex([first, second]),
        embedder=FakeEmbedder([1.0, 0.0]),
        chat_model=chat_model,
        query_planner=FakeQueryPlanner(
            QueryPlan(route="fact_lookup", rewritten_query="relevant passage")
        ),
        reranker=FakeReranker(),
        compressor=FakeCompressor(
            {"first": [], "second": ["Second relevant passage."]}
        ),
        embedding_dimension=2,
    )

    response = retriever.ask("tCMzkJxBM5c", "What is relevant?", top_k=2)

    assert [source.chunk_id for source in response.sources] == ["second"]
    assert chat_model.messages is not None
    assert "First unrelated passage." not in chat_model.messages[1][1]
