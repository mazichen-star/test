"""Supabase Auth / RLS 相关测试（不连接真实 Supabase）。"""

from __future__ import annotations

import requests

from core.storage import SupabaseStorage
from core.supabase_auth import SupabaseAuth


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}" if payload is not None else b""
        self.text = "{}"

    def json(self):
        return self._payload


def test_sign_in_returns_session(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeResponse(
            200,
            {
                "access_token": "token-123",
                "refresh_token": "refresh-123",
                "expires_in": 3600,
                "user": {"id": "user-1", "email": "a@example.com"},
            },
        )

    monkeypatch.setattr(requests, "request", fake_request)
    auth = SupabaseAuth("https://example.supabase.co", "anon-key")
    session = auth.sign_in("a@example.com", "password123")
    assert session["access_token"] == "token-123"
    assert session["refresh_token"] == "refresh-123"
    assert session["user"]["id"] == "user-1"
    assert calls[0][2]["params"]["grant_type"] == "password"
    assert calls[0][2]["headers"]["apikey"] == "anon-key"


def test_storage_uses_user_token_for_rls(monkeypatch):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return FakeResponse(200, [])

    monkeypatch.setattr(requests, "request", fake_request)
    storage = SupabaseStorage("https://example.supabase.co", "anon-key")
    storage.set_access_token("user-jwt")
    storage.list_user_files("tester")
    auth_header = calls[0][2]["headers"]["Authorization"]
    assert auth_header == "Bearer user-jwt"
    assert calls[0][2]["headers"]["apikey"] == "anon-key"


def test_refresh_and_profile(monkeypatch):
    def fake_request(method, url, **kwargs):
        if url.endswith("/auth/v1/token"):
            return FakeResponse(
                200,
                {
                    "access_token": "new-token",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                    "user": {"id": "user-1"},
                },
            )
        if "/rest/v1/profiles" in url:
            return FakeResponse(
                200, [{"id": "user-1", "username": "tester", "email": "a@example.com"}]
            )
        return FakeResponse(200, {})

    monkeypatch.setattr(requests, "request", fake_request)
    auth = SupabaseAuth("https://example.supabase.co", "anon-key")
    session = auth.refresh("old-refresh")
    assert session["access_token"] == "new-token"
    profile = auth.get_profile("new-token", "user-1")
    assert profile["username"] == "tester"
