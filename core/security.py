"""密码哈希与校验。

新密码使用 PBKDF2-HMAC-SHA256 + 随机盐；同时兼容旧版本的
``sha256(password)`` 60/64 位十六进制哈希，登录成功后自动升级。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 260_000
SALT_BYTES = 16

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{5,20}$")
PASSWORD_RE = re.compile(r"^(?=.*[0-9])(?=.*[A-Za-z])\S{8,}$")
LEGACY_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def hash_password(password: str) -> str:
    """生成带盐密码哈希，格式：``pbkdf2_sha256$迭代次数$盐$哈希``。"""

    if not isinstance(password, str) or password == "":
        raise ValueError("密码不能为空")
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, ITERATIONS
    )
    return f"{ALGORITHM}${ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str | None) -> bool:
    """校验密码，兼容旧版无盐 SHA-256。"""

    if not password or not stored_hash:
        return False

    if stored_hash.startswith(f"{ALGORITHM}$"):
        try:
            _, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
            expected = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(salt_hex),
                int(iterations),
            ).hex()
            return hmac.compare_digest(expected, digest_hex)
        except (ValueError, TypeError):
            return False

    if LEGACY_SHA256_RE.match(stored_hash):
        legacy = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(legacy, stored_hash)

    return False


def password_needs_rehash(stored_hash: str | None) -> bool:
    if not stored_hash or not stored_hash.startswith(f"{ALGORITHM}$"):
        return True
    try:
        _, iterations, _, _ = stored_hash.split("$", 3)
        return int(iterations) < ITERATIONS
    except (ValueError, TypeError):
        return True


def valid_username(username: str) -> bool:
    return bool(USERNAME_RE.match(username or ""))


def valid_password(password: str) -> bool:
    return bool(PASSWORD_RE.match(password or ""))
