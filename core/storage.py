"""数据存储层。

支持两种后端：

* ``SupabaseStorage``：通过 PostgREST 访问 Supabase；
* ``SQLiteStorage``：本地零配置运行，适合演示、开发与单元测试。

UI 层只依赖本模块暴露的统一接口，不再直接拼接 PostgREST URL。
"""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import requests

from .config import local_db_path, supabase_key, supabase_url, use_supabase
from .security import hash_password, password_needs_rehash, verify_password


class StorageError(RuntimeError):
    """存储层可预期的错误，UI 可以直接展示给用户。"""


def _b64encode(content: bytes) -> str:
    return "\x" + content.hex()


def _b64decode(value: Any) -> bytes:
    if value is None:
        raise StorageError("文件内容为空")
    if isinstance(value, bytes):
        return value
    text = str(value)
    # PostgREST 的 bytea 字段有时返回 "\x..." 十六进制串
    if text.startswith("\\x"):
        try:
            return bytes.fromhex(text[2:])
        except ValueError as exc:
            raise StorageError("文件内容格式错误") from exc
    try:
        return base64.b64decode(text, validate=False)
    except Exception as exc:
        raise StorageError("文件内容不是有效的 Base64") from exc


def _normalize_material(row: dict[str, Any]) -> dict[str, Any]:
    props = row.get("properties")
    if isinstance(props, str):
        try:
            props = json.loads(props)
        except json.JSONDecodeError:
            props = {}
    if not isinstance(props, dict):
        props = {}
    row["properties"] = props
    return row


class BaseStorage:
    """统一存储接口。"""

    mode = "base"

    def user_exists(self, username: str) -> bool:
        raise NotImplementedError

    def add_user(self, username: str, password: str, email: str) -> None:
        raise NotImplementedError

    def find_user_by_email(self, email: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def verify_user(self, username: str, password: str) -> bool:
        raise NotImplementedError

    def change_password(self, username: str, new_password: str) -> None:
        raise NotImplementedError

    def set_password_hash(self, username: str, password_hash: str) -> None:
        raise NotImplementedError

    def delete_account(self, username: str) -> None:
        raise NotImplementedError

    def list_user_files(self, username: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    def save_user_file(self, username: str, filename: str, content: bytes) -> None:
        raise NotImplementedError

    def load_user_file(self, username: str, filename: str) -> bytes | None:
        raise NotImplementedError

    def delete_user_file(self, username: str, filename: str) -> bool:
        raise NotImplementedError

    def list_materials(self, keyword: str = "") -> list[dict[str, Any]]:
        raise NotImplementedError

    def find_material(self, formula: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def add_material(
        self,
        formula: str,
        structure: str,
        properties: dict[str, Any],
        source: str,
        contributor: str,
        reference: str = "",
    ) -> None:
        raise NotImplementedError

    def add_materials(self, rows: list[dict[str, Any]]) -> int:
        raise NotImplementedError

    def delete_material(self, material_id: Any, username: str) -> bool:
        raise NotImplementedError


class SupabaseStorage(BaseStorage):
    mode = "supabase"

    MATERIAL_COLUMNS = (
        "id,formula,structure,properties,source,reference,contributor,created_at"
    )

    def __init__(self, url: str, key: str, timeout: int = 20) -> None:
        self.url = url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.base_headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        table: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        timeout: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        url = f"{self.url}/rest/v1/{table}"
        request_headers = dict(self.base_headers)
        if headers:
            request_headers.update(headers)
        try:
            response = requests.request(
                method,
                url,
                headers=request_headers,
                params=params,
                json=json_body,
                timeout=timeout or self.timeout,
            )
        except requests.RequestException as exc:
            raise StorageError(f"无法连接云数据库：{exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:300]
            if response.status_code == 409:
                raise StorageError("记录已存在，请勿重复提交")
            raise StorageError(f"数据库错误({response.status_code})：{detail}")

        if response.status_code == 204 or not response.content:
            return []
        try:
            return response.json()
        except ValueError:
            return []

    def user_exists(self, username: str) -> bool:
        rows = self._request(
            "GET",
            "users",
            params={"username": f"eq.{username}", "select": "username", "limit": "1"},
        )
        return bool(rows)

    def add_user(self, username: str, password: str, email: str) -> None:
        self._request(
            "POST",
            "users",
            json_body={
                "username": username,
                "password_hash": hash_password(password),
                "email": email,
            },
        )

    def find_user_by_email(self, email: str) -> dict[str, Any] | None:
        rows = self._request(
            "GET",
            "users",
            params={
                "email": f"eq.{email}",
                "select": "username,password_hash,email",
                "limit": "1",
            },
        )
        return rows[0] if rows else None

    def verify_user(self, username: str, password: str) -> bool:
        rows = self._request(
            "GET",
            "users",
            params={
                "username": f"eq.{username}",
                "select": "username,password_hash",
                "limit": "1",
            },
        )
        if not rows:
            return False
        stored = rows[0].get("password_hash")
        if not verify_password(password, stored):
            return False
        if password_needs_rehash(stored):
            self.set_password_hash(username, hash_password(password))
        return True

    def change_password(self, username: str, new_password: str) -> None:
        self.set_password_hash(username, hash_password(new_password))

    def set_password_hash(self, username: str, password_hash: str) -> None:
        self._request(
            "PATCH",
            "users",
            params={"username": f"eq.{username}"},
            json_body={"password_hash": password_hash},
        )

    def delete_account(self, username: str) -> None:
        self._request("DELETE", "user_files", params={"username": f"eq.{username}"})
        self._request(
            "PATCH",
            "material_library",
            params={"contributor": f"eq.{username}"},
            json_body={"contributor": "已注销用户"},
        )
        self._request("DELETE", "users", params={"username": f"eq.{username}"})

    def list_user_files(self, username: str) -> list[dict[str, Any]]:
        rows = self._request(
            "GET",
            "user_files",
            params={
                "username": f"eq.{username}",
                "select": "id,filename,uploaded_at",
                "order": "uploaded_at.desc,id.desc",
            },
        )
        # 兼容历史重复数据：同名文件只保留最新一条
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            name = str(row.get("filename") or "")
            if name in seen:
                continue
            seen.add(name)
            unique.append(row)
        return unique

    def save_user_file(self, username: str, filename: str, content: bytes) -> None:
        encoded = _b64encode(content)
        rows = self._request(
            "GET",
            "user_files",
            params={
                "username": f"eq.{username}",
                "filename": f"eq.{filename}",
                "select": "id",
                "order": "id.desc",
                "limit": "1",
            },
        )
        if rows:
            self._request(
                "PATCH",
                "user_files",
                params={"id": f"eq.{rows[0]['id']}"},
                json_body={"content": encoded},
            )
        else:
            self._request(
                "POST",
                "user_files",
                json_body={
                    "username": username,
                    "filename": filename,
                    "content": encoded,
                },
            )

    def load_user_file(self, username: str, filename: str) -> bytes | None:
        rows = self._request(
            "GET",
            "user_files",
            params={
                "username": f"eq.{username}",
                "filename": f"eq.{filename}",
                "select": "content",
                "order": "id.desc",
                "limit": "1",
            },
        )
        if not rows:
            return None
        return _b64decode(rows[0].get("content"))

    def delete_user_file(self, username: str, filename: str) -> bool:
        self._request(
            "DELETE",
            "user_files",
            params={"username": f"eq.{username}", "filename": f"eq.{filename}"},
        )
        return True

    def list_materials(self, keyword: str = "") -> list[dict[str, Any]]:
        params: dict[str, str] = {
            "select": self.MATERIAL_COLUMNS,
            "order": "created_at.desc",
        }
        keyword = (keyword or "").strip()
        if keyword:
            params["formula"] = f"ilike.*{keyword}*"
        rows = self._request("GET", "material_library", params=params)
        return [_normalize_material(row) for row in rows]

    def find_material(self, formula: str) -> dict[str, Any] | None:
        target = (formula or "").strip()
        if not target:
            return None
        rows = self._request(
            "GET",
            "material_library",
            params={
                "formula": f"ilike.{target}",
                "select": "id,formula,structure",
                "limit": "1",
            },
        )
        if rows:
            return rows[0]
        # 某些 PostgREST 版本对 ilike 精确匹配的行为不一致，做一次兜底
        for row in self.list_materials():
            if str(row.get("formula") or "").strip().upper() == target.upper():
                return row
        return None

    def add_material(
        self,
        formula: str,
        structure: str,
        properties: dict[str, Any],
        source: str,
        contributor: str,
        reference: str = "",
    ) -> None:
        self._request(
            "POST",
            "material_library",
            json_body={
                "formula": formula.strip(),
                "structure": structure.strip(),
                "properties": properties,
                "source": source.strip(),
                "reference": reference.strip(),
                "contributor": contributor,
            },
        )

    def add_materials(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        result = self._request(
            "POST",
            "material_library",
            json_body=rows,
            headers={"Prefer": "return=representation"},
            timeout=60,
        )
        return len(result) if isinstance(result, list) else len(rows)

    def delete_material(self, material_id: Any, username: str) -> bool:
        self._request(
            "DELETE",
            "material_library",
            params={"id": f"eq.{material_id}", "contributor": f"eq.{username}"},
        )
        return True


class SQLiteStorage(BaseStorage):
    mode = "sqlite"

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else local_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    email TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

                CREATE TABLE IF NOT EXISTS user_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(username, filename)
                );
                CREATE INDEX IF NOT EXISTS idx_user_files_user
                    ON user_files(username, uploaded_at DESC);

                CREATE TABLE IF NOT EXISTS material_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    formula TEXT NOT NULL,
                    structure TEXT NOT NULL DEFAULT '',
                    properties TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL DEFAULT '',
                    reference TEXT NOT NULL DEFAULT '',
                    contributor TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_material_formula
                    ON material_library(formula);
                """
            )

    # ---- 用户 ----
    def user_exists(self, username: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE username = ? LIMIT 1", (username,)
            ).fetchone()
        return row is not None

    def add_user(self, username: str, password: str, email: str) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users(username, password_hash, email) VALUES (?, ?, ?)",
                    (username, hash_password(password), email),
                )
        except sqlite3.IntegrityError as exc:
            raise StorageError("用户名已存在") from exc

    def find_user_by_email(self, email: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT username, password_hash, email FROM users "
                "WHERE email = ? ORDER BY id DESC LIMIT 1",
                (email,),
            ).fetchone()
        return dict(row) if row else None

    def verify_user(self, username: str, password: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ? LIMIT 1",
                (username,),
            ).fetchone()
            if not row:
                return False
            stored = row["password_hash"]
            if not verify_password(password, stored):
                return False
            if password_needs_rehash(stored):
                conn.execute(
                    "UPDATE users SET password_hash = ? WHERE username = ?",
                    (hash_password(password), username),
                )
        return True

    def change_password(self, username: str, new_password: str) -> None:
        self.set_password_hash(username, hash_password(new_password))

    def set_password_hash(self, username: str, password_hash: str) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (password_hash, username),
            )
            if cur.rowcount == 0:
                raise StorageError("用户不存在")

    def delete_account(self, username: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM user_files WHERE username = ?", (username,))
            conn.execute(
                "UPDATE material_library SET contributor = '已注销用户' "
                "WHERE contributor = ?",
                (username,),
            )
            conn.execute("DELETE FROM users WHERE username = ?", (username,))

    # ---- 文件 ----
    def list_user_files(self, username: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, uploaded_at FROM user_files "
                "WHERE username = ? ORDER BY uploaded_at DESC, id DESC",
                (username,),
            ).fetchall()
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            name = str(row["filename"])
            if name in seen:
                continue
            seen.add(name)
            unique.append(dict(row))
        return unique

    def save_user_file(self, username: str, filename: str, content: bytes) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO user_files(username, filename, content) VALUES (?, ?, ?) "
                "ON CONFLICT(username, filename) DO UPDATE SET "
                "content = excluded.content, uploaded_at = CURRENT_TIMESTAMP",
                (username, filename, _b64encode(content)),
            )

    def load_user_file(self, username: str, filename: str) -> bytes | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM user_files WHERE username = ? AND filename = ? "
                "ORDER BY id DESC LIMIT 1",
                (username, filename),
            ).fetchone()
        return _b64decode(row["content"]) if row else None

    def delete_user_file(self, username: str, filename: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM user_files WHERE username = ? AND filename = ?",
                (username, filename),
            )
        return True

    # ---- 材料库 ----
    def _material_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["properties"] = json.loads(item.get("properties") or "{}")
        except json.JSONDecodeError:
            item["properties"] = {}
        return item

    def list_materials(self, keyword: str = "") -> list[dict[str, Any]]:
        keyword = (keyword or "").strip()
        sql = (
            "SELECT id, formula, structure, properties, source, reference, "
            "contributor, created_at FROM material_library"
        )
        params: tuple[Any, ...] = ()
        if keyword:
            sql += " WHERE formula LIKE ? COLLATE NOCASE"
            params = (f"%{keyword}%",)
        sql += " ORDER BY datetime(created_at) DESC, id DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._material_row(row) for row in rows]

    def find_material(self, formula: str) -> dict[str, Any] | None:
        target = (formula or "").strip()
        if not target:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, formula, structure FROM material_library "
                "WHERE formula = ? COLLATE NOCASE LIMIT 1",
                (target,),
            ).fetchone()
        return dict(row) if row else None

    def add_material(
        self,
        formula: str,
        structure: str,
        properties: dict[str, Any],
        source: str,
        contributor: str,
        reference: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO material_library"
                "(formula, structure, properties, source, reference, contributor) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    formula.strip(),
                    structure.strip(),
                    json.dumps(properties, ensure_ascii=False),
                    source.strip(),
                    reference.strip(),
                    contributor,
                ),
            )

    def add_materials(self, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        inserted = 0
        with self._connect() as conn:
            for row in rows:
                try:
                    conn.execute(
                        "INSERT INTO material_library"
                        "(formula, structure, properties, source, reference, contributor) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(row.get("formula", "")).strip(),
                            str(row.get("structure", "")).strip(),
                            json.dumps(row.get("properties", {}), ensure_ascii=False),
                            str(row.get("source", "")).strip(),
                            str(row.get("reference", "")).strip(),
                            str(row.get("contributor", "")),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
        return inserted

    def delete_material(self, material_id: Any, username: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM material_library WHERE id = ? AND contributor = ?",
                (material_id, username),
            )
            return cur.rowcount > 0


_STORAGE: BaseStorage | None = None


def get_storage() -> BaseStorage:
    """返回当前存储后端；没有 Supabase 配置时自动使用本地 SQLite。"""

    global _STORAGE
    if _STORAGE is None:
        if use_supabase():
            _STORAGE = SupabaseStorage(supabase_url(), supabase_key())
        else:
            _STORAGE = SQLiteStorage()
    return _STORAGE


def reset_storage() -> None:
    """测试或切换配置时清空单例。"""

    global _STORAGE
    _STORAGE = None


def storage_label() -> str:
    return "Supabase 云数据库" if use_supabase() else "本地 SQLite"
