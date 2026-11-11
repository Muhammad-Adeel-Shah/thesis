"""Hugging Face access helpers.

Guardrail #5: the token is read from the environment or from Kaggle Secrets and is
never printed, logged, written to disk or committed. Nothing in this module returns
the token inside a message, an exception or a report; only its presence is reported.
"""

from __future__ import annotations

import os
from typing import Any

from huggingface_hub import HfApi

_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def get_hf_token() -> str | None:
    """Return the Hugging Face token, or None if no token is configured.

    Resolution order: process environment first, then Kaggle Secrets. The value is
    never logged. Callers should pass it straight to ``huggingface_hub`` and must not
    interpolate it into strings.
    """
    for var in _TOKEN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    try:
        from kaggle_secrets import UserSecretsClient  # noqa: PLC0415  (Kaggle-only import)

        return UserSecretsClient().get_secret("HF_TOKEN") or None
    except Exception:
        # Not on Kaggle, or no secret named HF_TOKEN is attached to this kernel.
        # This is an expected, non-fatal condition: gated members become BLOCKED-ACCESS.
        return None


def token_present() -> bool:
    """Return whether a token is available. Safe to log."""
    return get_hf_token() is not None


def resolve_revision(repo_id: str, token: str | None = None) -> str:
    """Return the current commit SHA of ``repo_id`` on the Hub.

    Used once, during P0, to fill ``revision:`` in ``configs/decoder_bank.yaml``. At
    run time the pinned value is used instead, so that an upstream re-upload can never
    silently change a decoder fingerprint.
    """
    info = HfApi().model_info(repo_id, token=token if token is not None else get_hf_token())
    return info.sha


def check_access(repo_id: str, token: str | None = None) -> dict[str, Any]:
    """Report reachability and gated-access status for one repository.

    No weights are downloaded. ``auth_check`` is the purpose-built gated-access probe.

    Returns:
        A dict with ``reachable``, ``gated``, ``access_granted``, ``sha``, ``license``
        and, when access is denied, a ``reason`` string. The token never appears in it.
    """
    tok = token if token is not None else get_hf_token()
    api = HfApi()
    out: dict[str, Any] = {
        "repo_id": repo_id,
        "reachable": False,
        "gated": None,
        "access_granted": False,
        "sha": None,
        "license": None,
        "reason": None,
    }
    try:
        info = api.model_info(repo_id, token=tok)
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {_redact(str(exc))}"
        return out

    out["reachable"] = True
    out["gated"] = info.gated
    out["sha"] = info.sha
    out["license"] = (info.card_data or {}).get("license") if info.card_data else None

    try:
        api.auth_check(repo_id, token=tok)
        out["access_granted"] = True
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {_redact(str(exc))}"
    return out


def _redact(message: str, limit: int = 200) -> str:
    """Truncate a Hub error message and strip anything token-shaped from it."""
    tok = get_hf_token()
    if tok:
        message = message.replace(tok, "<HF_TOKEN-REDACTED>")
    return message[:limit]
