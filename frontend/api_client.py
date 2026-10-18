from typing import Any, Protocol

import requests


class ApiClientError(Exception):
    pass


class HttpResponse(Protocol):
    status_code: int

    def json(self) -> Any: ...

    def raise_for_status(self) -> None: ...


class HttpSession(Protocol):
    def get(self, url: str, *, timeout: float) -> HttpResponse: ...

    def post(
        self, url: str, *, json: dict[str, Any], timeout: float
    ) -> HttpResponse: ...


class YouTubeChatApiClient:
    def __init__(
        self,
        base_url: str,
        session: HttpSession | None = None,
        request_timeout: float = 180,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._request_timeout = request_timeout

    def health_check(self) -> dict[str, Any]:
        return self._get("/health")

    def index_video(
        self,
        video_url: str,
        languages: list[str],
        translate_to_english: bool,
    ) -> dict[str, Any]:
        return self._post(
            "/videos/index",
            {
                "video_url": video_url,
                "languages": languages,
                "translate_to_english": translate_to_english,
            },
        )

    def ask_video(
        self, video_url: str, question: str, top_k: int
    ) -> dict[str, Any]:
        return self._post(
            "/videos/ask",
            {
                "video_url": video_url,
                "question": question,
                "top_k": top_k,
            },
        )

    def _get(self, path: str) -> dict[str, Any]:
        try:
            response = self._session.get(
                f"{self._base_url}{path}",
                timeout=min(self._request_timeout, 10),
            )
        except requests.RequestException as error:
            raise ApiClientError(
                f"Could not connect to the FastAPI backend at {self._base_url}. "
                "Make sure the backend is running."
            ) from error
        return self._read_response(response)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.post(
                f"{self._base_url}{path}",
                json=payload,
                timeout=self._request_timeout,
            )
        except requests.RequestException as error:
            raise ApiClientError(
                f"Could not complete the request to {self._base_url}. "
                "Check that the backend is running and try again."
            ) from error
        return self._read_response(response)

    @staticmethod
    def _read_response(response: HttpResponse) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as error:
            raise ApiClientError(
                f"The FastAPI backend returned invalid JSON "
                f"(HTTP {response.status_code})."
            ) from error

        if response.status_code >= 400:
            detail = data.get("detail", data) if isinstance(data, dict) else data
            if isinstance(detail, list):
                message = "; ".join(
                    str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                    for item in detail
                )
            else:
                message = str(detail)
            raise ApiClientError(f"Backend error (HTTP {response.status_code}): {message}")

        if not isinstance(data, dict):
            raise ApiClientError("The FastAPI backend returned an unexpected response.")
        return data
