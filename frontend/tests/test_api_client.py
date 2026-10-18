from typing import Any

import pytest
import requests

from frontend.api_client import ApiClientError, YouTubeChatApiClient


class FakeResponse:
    def __init__(self, status_code: int, data: Any) -> None:
        self.status_code = status_code
        self.data = data

    def json(self) -> Any:
        if isinstance(self.data, Exception):
            raise self.data
        return self.data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.calls.append(("GET", url, {"timeout": timeout}))
        return self.response

    def post(
        self, url: str, *, json: dict[str, Any], timeout: float
    ) -> FakeResponse:
        self.calls.append(
            ("POST", url, {"json": json, "timeout": timeout})
        )
        return self.response


def test_client_posts_ask_request_to_backend() -> None:
    session = FakeSession(FakeResponse(200, {"answer": "Grounded answer", "sources": []}))
    client = YouTubeChatApiClient("http://localhost:8000/", session=session)

    result = client.ask_video("video-id", "Question?", top_k=4)

    assert result["answer"] == "Grounded answer"
    assert session.calls == [
        (
            "POST",
            "http://localhost:8000/videos/ask",
            {
                "json": {
                    "video_url": "video-id",
                    "question": "Question?",
                    "top_k": 4,
                },
                "timeout": 180,
            },
        )
    ]


def test_client_reports_backend_error_details() -> None:
    session = FakeSession(
        FakeResponse(404, {"detail": "No indexed transcript chunks were found."})
    )
    client = YouTubeChatApiClient("http://localhost:8000", session=session)

    with pytest.raises(ApiClientError, match="HTTP 404.*No indexed transcript"):
        client.ask_video("video-id", "Question?", top_k=5)


def test_client_reports_connection_errors_without_hiding_them(monkeypatch) -> None:
    session = FakeSession(FakeResponse(200, {"status": "ok"}))

    def failed_get(url: str, *, timeout: float) -> FakeResponse:
        assert url.endswith("/health")
        assert timeout <= 10
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(session, "get", failed_get)
    client = YouTubeChatApiClient("http://localhost:8000", session=session)

    with pytest.raises(ApiClientError, match="Make sure the backend is running"):
        client.health_check()
