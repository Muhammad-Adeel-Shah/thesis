"""Unit tests for the pin verifier, with a mocked Hub API.

No network is touched. The point is to pin down the two branches that decide whether a
run proceeds:

- :data:`RevisionNotFoundError` is **fatal** — the registry points at a commit that does
  not exist, so anything built from it would be silently wrong;
- :data:`GatedRepoError` is **reported but not fatal** — the pin is fine and the problem
  is a licence or token, which is a different thing to fix.

Getting these two the wrong way round is exactly the failure P0 was meant to make
impossible, so it is tested rather than assumed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import requests
from huggingface_hub.utils import GatedRepoError, RevisionNotFoundError

THESIS_DBP = Path(__file__).resolve().parents[1]


def _load_script():
    """Import scripts/p0_verify_pins.py as a module (it is not on the package path)."""
    path = THESIS_DBP / "scripts" / "p0_verify_pins.py"
    spec = importlib.util.spec_from_file_location("p0_verify_pins", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["p0_verify_pins"] = module
    spec.loader.exec_module(module)
    return module


verify = _load_script()


def _http_error(cls, status: int, message: str):
    response = requests.Response()
    response.status_code = status
    return cls(message, response=response)


class FakeHfApi:
    """Stands in for HfApi. Raises whatever the test maps a repo to."""

    def __init__(self, behaviour: dict[str, Exception] | None = None):
        self.behaviour = behaviour or {}
        self.calls: list[tuple[str, str]] = []

    def model_info(self, repo: str, revision: str | None = None, token: str | None = None):
        self.calls.append((repo, revision))
        outcome = self.behaviour.get(repo)
        if isinstance(outcome, Exception):
            raise outcome
        return {"repo": repo, "sha": revision}


@pytest.fixture(scope="module")
def bank():
    from dbp.decoders.registry import load_bank_config

    return load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")


# --------------------------------------------------------------------------
# The decisive branches
# --------------------------------------------------------------------------


def test_missing_revision_is_fatal():
    api = FakeHfApi({"some/repo": _http_error(RevisionNotFoundError, 404, "no such revision")})
    result = verify.verify_hub_pin(api, "k", "vae_kl", "some/repo", "a" * 40)

    assert result.status == verify.STATUS_REVISION_NOT_FOUND
    assert result.is_fatal is True
    assert result.is_blocked is False
    assert "does not exist" in result.detail


def test_gated_repo_is_reported_but_not_fatal():
    api = FakeHfApi({"gated/repo": _http_error(GatedRepoError, 403, "awaiting a review")})
    result = verify.verify_hub_pin(api, "sd35", "vae_kl", "gated/repo", "b" * 40)

    assert result.status == verify.STATUS_BLOCKED
    assert result.is_blocked is True
    assert result.is_fatal is False, "a licence problem must not be reported as a bad pin"


def test_resolving_pin_is_clean():
    api = FakeHfApi()
    result = verify.verify_hub_pin(api, "sd15", "vae_kl", "ok/repo", "c" * 40)

    assert result.status == verify.STATUS_RESOLVES
    assert result.is_fatal is False
    assert api.calls == [("ok/repo", "c" * 40)]


def test_unexpected_exception_is_fatal():
    """A network or rate-limit failure must not be mistaken for a healthy pin."""
    api = FakeHfApi({"flaky/repo": TimeoutError("connection timed out")})
    result = verify.verify_hub_pin(api, "k", "vae_kl", "flaky/repo", "d" * 40)

    assert result.status == verify.STATUS_ERROR
    assert result.is_fatal is True
    assert "TimeoutError" in result.detail


def test_the_pinned_revision_is_what_gets_queried():
    """Guards against querying the branch head and calling a stale pin healthy."""
    api = FakeHfApi()
    verify.verify_hub_pin(api, "k", "vae_kl", "some/repo", "e" * 40)
    assert api.calls[0][1] == "e" * 40


# --------------------------------------------------------------------------
# URL-pinned member
# --------------------------------------------------------------------------


class _FakeResponse:
    headers = {"Content-Length": "67040989"}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_url_pin_ok(bank):
    spec = bank.by_key("realesrgan_x4")
    result = verify.verify_url_pin(spec, opener=lambda *a, **k: _FakeResponse())

    assert result.status == verify.STATUS_URL_OK
    assert result.is_fatal is False
    assert "67040989 bytes" in result.detail


def test_unreachable_url_is_fatal(bank):
    spec = bank.by_key("realesrgan_x4")

    def boom(*args, **kwargs):
        raise OSError("release asset gone")

    result = verify.verify_url_pin(spec, opener=boom)
    assert result.status == verify.STATUS_ERROR
    assert result.is_fatal is True


# --------------------------------------------------------------------------
# Whole-bank behaviour
# --------------------------------------------------------------------------


def test_verify_pins_covers_every_member_and_backbone(bank):
    from dbp.backbones.loader import BACKBONES

    results = verify.verify_pins(
        bank, BACKBONES, FakeHfApi(), url_opener=lambda *a, **k: _FakeResponse()
    )

    assert len(results) == len(bank.members) + len(BACKBONES)
    assert {r.key for r in results} >= {s.key for s in bank.members}
    assert {r.key for r in results} >= set(BACKBONES)
    assert not [r for r in results if r.is_fatal]


def test_verify_pins_surfaces_one_bad_pin_among_many(bank):
    from dbp.backbones.loader import BACKBONES

    bad = bank.by_key("flux")
    api = FakeHfApi({bad.repo: _http_error(RevisionNotFoundError, 404, "gone")})
    results = verify.verify_pins(
        bank, BACKBONES, api, url_opener=lambda *a, **k: _FakeResponse()
    )

    fatal = [r for r in results if r.is_fatal]
    assert [r.key for r in fatal] == ["flux"]


def test_gated_members_do_not_make_the_run_fatal(bank):
    from dbp.backbones.loader import BACKBONES

    api = FakeHfApi(
        {
            bank.by_key("sd35").repo: _http_error(GatedRepoError, 403, "gated"),
            bank.by_key("flux").repo: _http_error(GatedRepoError, 403, "gated"),
        }
    )
    results = verify.verify_pins(
        bank, BACKBONES, api, url_opener=lambda *a, **k: _FakeResponse()
    )

    assert {r.key for r in results if r.is_blocked} == {"sd35", "flux"}
    assert not [r for r in results if r.is_fatal]


def test_format_table_marks_only_fatal_rows(bank):
    rows = [
        verify.PinResult("ok", "vae_kl", "r", "a" * 40, verify.STATUS_RESOLVES),
        verify.PinResult("bad", "vae_kl", "r", "b" * 40, verify.STATUS_REVISION_NOT_FOUND, "x"),
        verify.PinResult("gated", "vae_kl", "r", "c" * 40, verify.STATUS_BLOCKED, "y"),
    ]
    table = verify.format_table(rows)
    lines = {line.split()[0]: line for line in table.splitlines() if line and "-" * 10 not in line}

    assert "***" not in lines["ok"]
    assert "***" in lines["bad"]
    assert "***" not in lines["gated"], "a gated repo is not a broken pin"
