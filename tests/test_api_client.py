import pytest
import requests

from app.frontend.services.api_client import ApiClient, ApiConnectionError, ApiResponseError, ApiTimeoutError


class Response:
    def __init__(self, status_code=200, payload=None, invalid_json=False):
        self.status_code = status_code
        self.payload = payload or {}
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("bad json")
        return self.payload


class Session:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


def client(response=None, error=None):
    return ApiClient("http://api", session=Session(response, error))


def test_health_success():
    assert client(Response(payload={"status": "healthy"})).health_check()["status"] == "healthy"


def test_health_failure_status():
    with pytest.raises(ApiResponseError) as error:
        client(Response(status_code=500)).health_check()
    assert error.value.status_code == 500


def test_timeout():
    with pytest.raises(ApiTimeoutError):
        client(error=requests.Timeout()).health_check()


def test_connection_failure():
    with pytest.raises(ApiConnectionError):
        client(error=requests.ConnectionError()).health_check()


def test_invalid_json():
    with pytest.raises(ApiResponseError):
        client(Response(invalid_json=True)).health_check()


def test_predict_success():
    assert client(Response(payload={"prediction": 0})).predict({"Age": 1})["prediction"] == 0


def test_batch_success():
    api = client(Response(payload={"total_samples": 2}))
    assert api.predict_batch([{}, {}])["total_samples"] == 2
