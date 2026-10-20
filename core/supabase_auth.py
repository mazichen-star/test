"""Supabase Auth 客户端。

它替代原来的“自建 users 表 + 用户名密码”云登录方式：
- 登录后拿到 access_token；
- 所有 PostgREST 请求使用这个 token；
- RLS 通过 auth.uid() 判断用户，而不是相信请求里的 username。
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .config import supabase_key, supabase_url


class SupabaseAuthError(RuntimeError):
    """可展示给用户的认证错误。"""


def _error_message(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    for key in ("msg", "message", "error_description", "error"):
        value = data.get(key)
        if value:
            return str(value)
    return response.text[:300]


class SupabaseAuth:
    def __init__(self, url: str | None = None, key: str | None = None, timeout: int = 20):
        self.url = (url or supabase_url()).rstrip("/")
        self.key = key or supabase_key()
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        access_token: str | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        headers = {
            "apikey": self.key,
            "Content-Type": "application/json",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        try:
            response = requests.request(
                method,
                f"{self.url}{path}",
                headers=headers,
                json=json_body,
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise SupabaseAuthError(f"无法连接 Supabase Auth：{exc}") from exc
        if response.status_code >= 400:
            raise SupabaseAuthError(_error_message(response))
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return None

    @staticmethod
    def _session(data: dict) -> dict:
        if not data or not data.get("access_token"):
            return {}
        expires_in = int(data.get("expires_in") or 3600)
        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "expires_at": time.time() + expires_in,
            "expires_in": expires_in,
            "user": data.get("user") or {},
        }

    def sign_up(self, email: str, password: str, username: str) -> dict:
        data = self._request(
            "POST",
            "/auth/v1/signup",
            json_body={
                "email": email,
                "password": password,
                "data": {"username": username},
            },
        )
        return self._session(data) if isinstance(data, dict) else {}

    def sign_in(self, email: str, password: str) -> dict:
        data = self._request(
            "POST",
            "/auth/v1/token",
            params={"grant_type": "password"},
            json_body={"email": email, "password": password},
        )
        session = self._session(data) if isinstance(data, dict) else {}
        if not session:
            raise SupabaseAuthError("登录失败：未返回 session")
        return session

    def refresh(self, refresh_token: str) -> dict:
        data = self._request(
            "POST",
            "/auth/v1/token",
            params={"grant_type": "refresh_token"},
            json_body={"refresh_token": refresh_token},
        )
        session = self._session(data) if isinstance(data, dict) else {}
        if not session:
            raise SupabaseAuthError("登录已过期，请重新登录")
        return session

    def get_user(self, access_token: str) -> dict:
        data = self._request("GET", "/auth/v1/user", access_token=access_token)
        return data if isinstance(data, dict) else {}

    def get_profile(self, access_token: str, user_id: str) -> dict | None:
        rows = self._request(
            "GET",
            "/rest/v1/profiles",
            params={"id": f"eq.{user_id}", "select": "id,username,email"},
            access_token=access_token,
        )
        return rows[0] if isinstance(rows, list) and rows else None

    def ensure_profile(
        self, access_token: str, user_id: str, username: str, email: str
    ) -> dict:
        profile = self.get_profile(access_token, user_id)
        if profile:
            return profile
        self._request(
            "POST",
            "/rest/v1/profiles",
            json_body={"id": user_id, "username": username, "email": email},
            access_token=access_token,
        )
        profile = self.get_profile(access_token, user_id)
        if not profile:
            raise SupabaseAuthError("无法创建用户资料，请检查 profiles 表与 RLS 配置")
        return profile

    def update_password(self, access_token: str, new_password: str) -> None:
        self._request(
            "PUT",
            "/auth/v1/user",
            json_body={"password": new_password},
            access_token=access_token,
        )

    def recover(self, email: str) -> None:
        self._request("POST", "/auth/v1/recover", json_body={"email": email})

    def delete_account(self, access_token: str) -> None:
        self._request(
            "POST",
            "/rest/v1/rpc/delete_my_account",
            json_body={},
            access_token=access_token,
        )

    def sign_out(self, access_token: str) -> None:
        try:
            self._request("POST", "/auth/v1/logout", access_token=access_token)
        except SupabaseAuthError:
            # 本地 session 清除比服务端注销更重要
            pass
