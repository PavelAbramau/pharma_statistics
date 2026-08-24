from unittest.mock import MagicMock, patch

import pytest
import requests

from pharma_stats.clients.ctgov import CtgovClient, CtgovError


def _resp(status_code, text="{}", headers=None):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.url = "https://clinicaltrials.gov/api/v2/studies"
    r.headers = headers or {}
    return r


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("pharma_stats.clients.ctgov.get_as_of", lambda *a, **k: None)
    monkeypatch.setattr("pharma_stats.clients.ctgov.save_snapshot", lambda *a, **k: None)
    c = CtgovClient(min_interval=0.0, max_retries=3)
    c.session = MagicMock()
    return c


def test_get_returns_body_on_200(client):
    client.session.get.return_value = _resp(200, text='{"ok": true}')
    assert client._get("/studies", {}) == '{"ok": true}'


def test_get_retries_on_429_then_succeeds(client):
    client.session.get.side_effect = [_resp(429, headers={}), _resp(200, text="ok")]
    with patch("time.sleep") as sleep_mock:
        result = client._get("/studies", {})
    assert result == "ok"
    assert sleep_mock.called


def test_get_honors_retry_after_header(client):
    client.session.get.side_effect = [_resp(429, headers={"Retry-After": "3"}), _resp(200, text="ok")]
    with patch("time.sleep") as sleep_mock:
        client._get("/studies", {})
    # First sleep is the Retry-After wait; a second (shorter) sleep now
    # follows from _throttle(), since the 429 also raised min_interval —
    # see test_429_permanently_raises_baseline_throttle.
    sleep_mock.assert_any_call(3.0)


def test_get_raises_after_max_retries_on_5xx(client):
    client.session.get.return_value = _resp(503, text="down")
    with patch("time.sleep"):
        with pytest.raises(CtgovError):
            client._get("/studies", {})


def test_get_raises_immediately_on_non_retryable_4xx(client):
    client.session.get.return_value = _resp(404, text="not found")
    with pytest.raises(CtgovError):
        client._get("/studies", {})
    assert client.session.get.call_count == 1  # no retry loop for a plain 404


def test_get_retries_on_connection_error_then_succeeds(client):
    """Regression test for the 2026-08-20 DNS-outage incident: a transient
    connection failure (raised before any Response exists) must be
    retried like a 429/5xx, not propagate as an immediate hard failure."""
    client.session.get.side_effect = [
        requests.exceptions.ConnectionError("Failed to resolve 'clinicaltrials.gov'"),
        _resp(200, text="recovered"),
    ]
    with patch("time.sleep") as sleep_mock:
        result = client._get("/studies", {})
    assert result == "recovered"
    assert sleep_mock.called


def test_get_raises_ctgov_error_after_max_retries_on_sustained_connection_failure(client):
    client.session.get.side_effect = requests.exceptions.ConnectionError("DNS down")
    with patch("time.sleep"):
        with pytest.raises(CtgovError):
            client._get("/studies", {})
    assert client.session.get.call_count == client.max_retries + 1


def test_429_permanently_raises_baseline_throttle(client):
    assert client.min_interval == 0.0
    client.session.get.side_effect = [_resp(429, headers={}), _resp(200, text="ok")]
    with patch("time.sleep"):
        client._get("/studies", {})
    assert client.min_interval == 2.0


def test_429_throttle_increase_is_one_way_ratchet(client):
    client.min_interval = 5.0  # already higher than the 2.0s floor
    client.session.get.side_effect = [_resp(429, headers={}), _resp(200, text="ok")]
    with patch("time.sleep"):
        client._get("/studies", {})
    assert client.min_interval == 5.0  # unchanged, not lowered to the floor
