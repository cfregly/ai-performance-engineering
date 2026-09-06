"""Shared secret redaction for API and MCP response envelopes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "<redacted>"
_SENSITIVE_NAMES = {
    "access_key",
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "client_secret",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}
_SENSITIVE_SUFFIXES = (
    "_access_key",
    "_access_token",
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_passwd",
    "_private_key",
    "_refresh_token",
    "_secret",
    "_token",
)


def is_sensitive_name(name: Any) -> bool:
    """Return whether a mapping key or command flag conventionally contains a secret."""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return normalized in _SENSITIVE_NAMES or normalized.endswith(_SENSITIVE_SUFFIXES)


def _string_values(value: Any) -> set[str]:
    if isinstance(value, str) and value:
        return {value}
    if isinstance(value, Mapping):
        values: set[str] = set()
        for item in value.values():
            values.update(_string_values(item))
        return values
    if isinstance(value, list | tuple):
        values = set()
        for item in value:
            values.update(_string_values(item))
        return values
    return set()


def _collect_sensitive_strings(value: Any) -> set[str]:
    secrets: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if is_sensitive_name(key):
                secrets.update(_string_values(item))
            else:
                secrets.update(_collect_sensitive_strings(item))
        return secrets
    if isinstance(value, list | tuple):
        items = list(value)
        for index, item in enumerate(items):
            if isinstance(item, str) and item.startswith("-"):
                flag, separator, flag_value = item.partition("=")
                if is_sensitive_name(flag.lstrip("-")):
                    if separator and flag_value:
                        secrets.add(flag_value)
                    elif index + 1 < len(items):
                        secrets.update(_string_values(items[index + 1]))
            secrets.update(_collect_sensitive_strings(item))
    return secrets


def _redact_string(value: str, secrets: set[str]) -> str:
    if value in secrets:
        return REDACTED
    redacted = value
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= 8:
            redacted = redacted.replace(secret, REDACTED)
            continue
        # The source field established that this value is a credential. For
        # short values, require boundaries only where the credential begins or
        # ends with a word character, avoiding changes to unrelated words.
        prefix = r"(?<![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", secret[0]) else ""
        suffix = r"(?![A-Za-z0-9_])" if re.match(r"[A-Za-z0-9_]", secret[-1]) else ""
        redacted = re.sub(f"{prefix}{re.escape(secret)}{suffix}", REDACTED, redacted)
    return redacted


def _redact(value: Any, secrets: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {
            key: REDACTED if is_sensitive_name(key) else _redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        redacted_items: list[Any] = []
        redact_next = False
        for item in value:
            if redact_next:
                redacted_items.append(REDACTED)
                redact_next = False
                continue
            if isinstance(item, str) and item.startswith("-"):
                flag, separator, _flag_value = item.partition("=")
                if is_sensitive_name(flag.lstrip("-")):
                    if separator:
                        redacted_items.append(f"{flag}={REDACTED}")
                    else:
                        redacted_items.append(item)
                        redact_next = True
                    continue
            redacted_items.append(_redact(item, secrets))
        return tuple(redacted_items) if isinstance(value, tuple) else redacted_items
    if isinstance(value, str):
        return _redact_string(value, secrets)
    return value


def redact_sensitive_data(value: Any, *, source: Any = None) -> Any:
    """Return a shape-preserving copy with credential values replaced.

    ``source`` lets an envelope redact a credential copied into an otherwise
    innocently named result field, such as a command or an error string.
    """
    secrets = _collect_sensitive_strings(value)
    if source is not None:
        secrets.update(_collect_sensitive_strings(source))
    return _redact(value, secrets)
