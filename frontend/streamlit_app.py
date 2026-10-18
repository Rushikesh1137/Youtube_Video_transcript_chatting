import os

import streamlit as st

from api_client import ApiClientError, YouTubeChatApiClient

DEFAULT_API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="YouTube Chat",
    page_icon="▶️",
    layout="wide",
)

st.title("Chat with a YouTube video")
st.caption(
    "Index a video's transcript, then ask questions grounded in its English or "
    "Hindi captions."
)

with st.sidebar:
    st.header("Backend")
    api_url = st.text_input(
        "FastAPI URL",
        value=os.getenv("YOUTUBE_CHAT_API_URL", DEFAULT_API_URL),
        help="Run the backend separately, then connect this interface to it.",
    )
    api_client = YouTubeChatApiClient(api_url)
    if st.button("Check backend connection", use_container_width=True):
        try:
            health = api_client.health_check()
        except ApiClientError as error:
            st.error(str(error))
        else:
            st.success(f"Backend status: {health.get('status', 'unknown')}")

    st.divider()
    st.header("Transcript settings")
    selected_languages = st.multiselect(
        "Caption languages to index",
        options=["en", "hi"],
        default=["en", "hi"],
        format_func=lambda code: "English" if code == "en" else "Hindi",
        help="Every selected language available for the video is indexed.",
    )
    translate_to_english = st.checkbox(
        "Also index an English translation of Hindi captions",
        value=True,
        help=(
            "Uses YouTube's translation when the Hindi caption track supports it. "
            "The original Hindi captions are kept either way."
        ),
    )
    top_k = st.slider("Sources per answer", min_value=1, max_value=10, value=5)

video_url = st.text_input(
    "YouTube video URL or ID",
    placeholder="https://www.youtube.com/watch?v=...",
    key="video_url",
)

index_col, status_col = st.columns([1, 3], vertical_alignment="center")
with index_col:
    index_clicked = st.button(
        "Index video",
        type="primary",
        use_container_width=True,
        disabled=not video_url.strip() or not selected_languages,
    )
with status_col:
    st.caption(
        "Indexing fetches captions and stores timestamped transcript chunks in Pinecone."
    )

if index_clicked:
    try:
        with st.spinner("Fetching captions and indexing transcript..."):
            result = api_client.index_video(
                video_url=video_url.strip(),
                languages=selected_languages,
                translate_to_english=translate_to_english,
            )
    except ApiClientError as error:
        st.error(str(error))
    else:
        st.session_state["indexed_video_url"] = video_url.strip()
        st.session_state["messages"] = []
        language_names = {"en": "English", "hi": "Hindi"}
        languages = ", ".join(
            language_names.get(code, code)
            for code in result.get("languages_indexed", [])
        ) or "unknown"
        translations = result.get("translations_indexed", [])
        if "en" in translations:
            providers = ", ".join(
                provider.title()
                for provider in result.get("translation_providers", [])
            )
            translation_status = (
                f"included via {providers}" if providers else "included"
            )
        elif translate_to_english and "hi" in selected_languages:
            translation_status = "unavailable; see details below"
        else:
            translation_status = "not requested"
        st.success(
            f"Indexed {result.get('chunks_indexed', 0)} chunks "
            f"(captions: {languages}; English translation: {translation_status}) "
            f"for video {result.get('video_id', '')}."
        )
        for warning in result.get("warnings", []):
            st.warning(warning)

st.divider()
st.subheader("Ask about the video")

if not st.session_state.get("messages"):
    st.info("Index a video above to begin chatting with its transcript.")

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("sources", []):
            start = int(citation["start"])
            minutes, seconds = divmod(start, 60)
            timestamp = f"{minutes}:{seconds:02d}"
            label = (
                f"{citation['language_code']} · {timestamp} · "
                f"{citation['text'][:180]}"
            )
            st.markdown(f"[{label}]({citation['url']})")

question = st.chat_input(
    "Ask a question about the indexed video...",
    disabled=not st.session_state.get("indexed_video_url"),
)
if question:
    indexed_video_url = st.session_state["indexed_video_url"]
    messages = st.session_state.setdefault("messages", [])
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        with st.spinner("Searching the transcript and preparing an answer..."):
            result = api_client.ask_video(
                video_url=indexed_video_url,
                question=question,
                top_k=top_k,
            )
    except ApiClientError as error:
        st.session_state["messages"].pop()
        st.error(str(error))
    else:
        answer = result["answer"]
        sources = result.get("sources", [])
        messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
            for citation in sources:
                start = int(citation["start"])
                minutes, seconds = divmod(start, 60)
                timestamp = f"{minutes}:{seconds:02d}"
                label = (
                    f"{citation['language_code']} · {timestamp} · "
                    f"{citation['text'][:180]}"
                )
                st.markdown(f"[{label}]({citation['url']})")
