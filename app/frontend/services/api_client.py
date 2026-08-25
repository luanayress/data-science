"""Typed HTTP boundary used by Streamlit."""

from typing import Any, Dict, Iterable, Optional

import requests


class ApiClientError(Exception):
    pass


class ApiConnectionError(ApiClientError):
    pass


class ApiTimeoutError(ApiClientError):
    pass


class ApiResponseError(ApiClientError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 5.0, batch_timeout: float = 30.0, session=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_timeout = batch_timeout
        self.session = session or requests.Session()

    def health_check(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def get_model_info(self) -> Dict[str, Any]:
        return self._request("GET", "/model-info")

    def predict(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/predict", json=customer, timeout=10.0)

    def predict_batch(self, customers: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        return self._request("POST", "/predict-batch", json={"data": list(customers)}, timeout=self.batch_timeout)

    def monitor_report(self, reference, current, alpha: float = 0.05) -> Dict[str, Any]:
        files = {"reference": reference, "current": current}
        return self._request("POST", "/monitor/report", files=files, data={"alpha": alpha}, timeout=self.batch_timeout)

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        timeout = kwargs.pop("timeout", self.timeout)
        try:
            response = self.session.request(method, self.base_url + path, timeout=timeout, **kwargs)
        except requests.Timeout as exc:
            raise ApiTimeoutError("API request timed out") from exc
        except requests.ConnectionError as exc:
            raise ApiConnectionError("Could not connect to API") from exc
        except requests.RequestException as exc:
            raise ApiConnectionError("API request failed") from exc
        if not 200 <= response.status_code < 300:
            raise ApiResponseError("API returned status {}".format(response.status_code), response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiResponseError("API returned invalid JSON", response.status_code) from exc
