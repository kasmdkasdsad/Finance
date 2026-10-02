"""Password hashes for the dashboard login (PBKDF2-HMAC-SHA256, from the standard library).

Only the hash is ever stored (``QP_DASHBOARD_PASSWORD_HASH``): ``pbkdf2_sha256:<iterations>:<salt>:<hash>``
with a random 16-byte salt, URL-safe base64 without padding — letters, digits, ``-``, ``_`` and ``:`` only, so
it survives ``.env`` files and Docker Compose unquoted (a ``$`` would be read as a variable).
``quantpulse-hash-password`` makes one; verification compares in constant time. The password itself is never written, logged or sent anywhere.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

SCHEME = "pbkdf2_sha256"
ITERATIONS = 600_000  # OWASP's current recommendation for PBKDF2-HMAC-SHA256
MIN_ITERATIONS = 200_000
MIN_LENGTH = 12


class PasswordHashError(ValueError):
    pass


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, iterations: int = ITERATIONS, salt: bytes | None = None) -> str:
    if len(password) < MIN_LENGTH:
        raise PasswordHashError(f"use at least {MIN_LENGTH} characters")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{SCHEME}:{iterations}:{_b64(salt)}:{_b64(digest)}"


def parse(encoded: str) -> tuple[int, bytes, bytes]:
    """``(iterations, salt, digest)`` of a stored hash, or :class:`PasswordHashError`."""
    parts = encoded.strip().split(":")
    if len(parts) != 4 or parts[0] != SCHEME:
        raise PasswordHashError(f"not a {SCHEME} hash: make one with quantpulse-hash-password")
    try:
        iterations, salt, digest = int(parts[1]), _unb64(parts[2]), _unb64(parts[3])
    except ValueError as exc:
        raise PasswordHashError("the hash is damaged: make a new one with quantpulse-hash-password") from exc
    if iterations < MIN_ITERATIONS or len(salt) < 8 or len(digest) != 32:
        raise PasswordHashError(
            f"the hash is too weak (at least {MIN_ITERATIONS} iterations): make a new one with "
            "quantpulse-hash-password"
        )
    return iterations, salt, digest


def verify_password(password: str, encoded: str) -> bool:
    try:
        iterations, salt, digest = parse(encoded)
    except PasswordHashError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, digest)
