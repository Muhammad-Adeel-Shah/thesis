"""Verify that every pinned model revision still resolves.

Single source of truth for pin verification. `kaggle_run_p0.md` calls this before the
~4.9 GB download, and from P1 onward every data-building script calls it before writing
any data: a dataset built against a pin that points at nothing is worse than no dataset,
because it looks valid.

Covers every pin in the project, discovered from the registry rather than counted
here, so adding a bank member needs no edit to this file:

- every Hugging Face bank member, by commit SHA (12 after the P1-B3 extension);
- ``realesrgan_x4``, which has no Hub commit and is pinned by URL plus SHA-256;
- both backbones from :data:`dbp.backbones.loader.BACKBONES`.

A missing revision is **fatal**: the registry is wrong and must be fixed. A gated repo
the token cannot reach is **reported but not fatal** by default, because it is an
account/licence problem rather than a wrong pin; ``--strict`` makes it fatal too.

Exit codes: 0 = every pin resolves, 1 = at least one pin failed, 2 = could not run.

Usage:
    python scripts/p0_verify_pins.py
    python scripts/p0_verify_pins.py --strict --json reports/p0_pin_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.backbones.loader import BACKBONES  # noqa: E402
from dbp.decoders.registry import BankConfig, DecoderSpec, load_bank_config  # noqa: E402
from dbp.utils.hf import get_hf_token  # noqa: E402

#: Pin resolved successfully.
STATUS_RESOLVES = "RESOLVES"
#: The URL-pinned member's asset is reachable.
STATUS_URL_OK = "URL-OK"
#: The pinned commit does not exist. Fatal: the registry is wrong.
STATUS_REVISION_NOT_FOUND = "REVISION-NOT-FOUND"
#: Gated repo the current token cannot reach. Reported; fatal only with --strict.
STATUS_BLOCKED = "BLOCKED-ACCESS"
#: Anything else (network, rate limit, unreachable URL). Fatal.
STATUS_ERROR = "ERROR"

FATAL_STATUSES = frozenset({STATUS_REVISION_NOT_FOUND, STATUS_ERROR})


@dataclass(frozen=True)
class PinResult:
    """The outcome of checking one pin."""

    key: str
    kind: str
    target: str
    pinned: str
    status: str
    detail: str | None = None

    @property
    def is_fatal(self) -> bool:
        """Whether this outcome must stop the run regardless of --strict."""
        return self.status in FATAL_STATUSES

    @property
    def is_blocked(self) -> bool:
        """Whether this is a gated-access problem rather than a wrong pin."""
        return self.status == STATUS_BLOCKED


def verify_hub_pin(
    api: Any,
    key: str,
    kind: str,
    repo: str,
    revision: str,
    token: str | None = None,
) -> PinResult:
    """Check that ``revision`` exists in ``repo``.

    ``api`` is injected rather than constructed here so the failure branches can be
    unit-tested without touching the network.

    This asserts the commit **exists**; it deliberately does not compare it to the
    branch head, because the entire point of a pin is not to follow upstream.
    """
    from huggingface_hub.utils import GatedRepoError, RevisionNotFoundError

    try:
        api.model_info(repo, revision=revision, token=token)
    except RevisionNotFoundError as exc:
        return PinResult(
            key, kind, repo, revision, STATUS_REVISION_NOT_FOUND,
            f"pinned revision does not exist in {repo}: {str(exc)[:160]}",
        )
    except GatedRepoError as exc:
        return PinResult(
            key, kind, repo, revision, STATUS_BLOCKED,
            f"gated; token cannot access {repo}: {str(exc)[:160]}",
        )
    except Exception as exc:
        return PinResult(
            key, kind, repo, revision, STATUS_ERROR, f"{type(exc).__name__}: {str(exc)[:160]}"
        )
    return PinResult(key, kind, repo, revision, STATUS_RESOLVES)


def verify_url_pin(
    spec: DecoderSpec,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> PinResult:
    """Check that a URL-pinned member's asset is reachable, via a HEAD request.

    The SHA-256 itself is enforced at load time by
    :func:`dbp.decoders.loaders.fetch_sr_weights`; this only confirms the asset is
    still served, so a dead release URL is caught before a long run starts.
    """
    pinned = spec.sha256 or "<unpinned>"
    if not spec.url:
        return PinResult(spec.key, spec.kind, "", pinned, STATUS_ERROR, "no url in registry")
    try:
        request = urllib.request.Request(spec.url, method="HEAD")
        with opener(request, timeout=30) as response:  # noqa: S310 (allow-listed https)
            size = response.headers.get("Content-Length")
    except Exception as exc:
        return PinResult(
            spec.key, spec.kind, spec.url, pinned, STATUS_ERROR,
            f"{type(exc).__name__}: {str(exc)[:160]}",
        )
    return PinResult(
        spec.key, spec.kind, spec.url, pinned, STATUS_URL_OK,
        f"{size} bytes; sha256 verified at load",
    )


def verify_pins(
    bank: BankConfig,
    backbones: dict[str, dict[str, Any]],
    api: Any,
    token: str | None = None,
    url_opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[PinResult]:
    """Verify every bank member and backbone pin. Pure: no printing, no exit."""
    results = [
        verify_hub_pin(api, spec.key, spec.kind, spec.repo, spec.revision, token)
        if spec.is_hub_member
        else verify_url_pin(spec, url_opener)
        for spec in bank.members
    ]
    results.extend(
        verify_hub_pin(api, name, "backbone", entry["repo"], entry["revision"], token)
        for name, entry in backbones.items()
    )
    return results


def format_table(results: list[PinResult]) -> str:
    """Render the results as a fixed-width table."""
    lines = [f"{'key':16s}{'kind':14s}{'pinned':14s}result", "-" * 80]
    for r in results:
        marker = "*** " if r.is_fatal else ""
        detail = f"  ({r.detail})" if r.status == STATUS_URL_OK and r.detail else ""
        lines.append(f"{r.key:16s}{r.kind:14s}{r.pinned[:12]:14s}{marker}{r.status}{detail}")
    lines.append("-" * 80)
    return "\n".join(lines)


class PinVerificationError(RuntimeError):
    """Raised by :func:`require_pins` when a pin does not resolve."""


def require_pins(
    config_path: Path | None = None,
    strict: bool = False,
    printer: Callable[[str], None] = print,
) -> list[PinResult]:
    """Verify every pin and raise unless they all resolve. The P1 entry point.

    CLAUDE.md 7a: from P1 onward, every data-building script calls this **before
    writing any data**. A dataset built against a pin that points at nothing is worse
    than no dataset, because it looks valid. This is the programmatic form of
    :func:`main`, so there is still exactly one implementation of the check.

    Args:
        config_path: The bank registry. Defaults to the shipped one.
        strict: Treat a gated repo the token cannot reach as fatal too.
        printer: Where the table goes. Pass a logger's ``info`` to fold it into a log.

    Returns:
        The results, on success.

    Raises:
        PinVerificationError: If any pin is fatal, or -- with ``strict`` -- blocked.
    """
    from huggingface_hub import HfApi

    bank = load_bank_config(config_path or THESIS_DBP / "configs" / "decoder_bank.yaml")
    token = get_hf_token()
    printer(f"HF_TOKEN present: {token is not None}")
    results = verify_pins(bank, BACKBONES, HfApi(), token)
    printer(format_table(results))

    fatal = [r for r in results if r.is_fatal]
    blocked = [r for r in results if r.is_blocked]
    if fatal or (strict and blocked):
        failures = fatal + (blocked if strict else [])
        detail = "; ".join(f"{r.key}: {r.status} ({r.detail})" for r in failures)
        raise PinVerificationError(
            f"{len(failures)} pin(s) did not resolve, refusing to write data: {detail}"
        )
    printer(
        f"pin check OK: {len(results)} pins, 0 fatal, "
        f"{len(blocked)} blocked by gating (non-fatal)"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify every pinned model revision resolves.")
    parser.add_argument(
        "--config", type=Path, default=THESIS_DBP / "configs" / "decoder_bank.yaml"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat BLOCKED-ACCESS as fatal too (default: reported but non-fatal).",
    )
    parser.add_argument("--json", type=Path, default=None, help="Also write results as JSON.")
    args = parser.parse_args(argv)

    from huggingface_hub import HfApi

    token = get_hf_token()
    print(f"HF_TOKEN present: {token is not None}\n")

    try:
        bank = load_bank_config(args.config)
    except Exception as exc:
        print(f"FATAL: cannot load {args.config}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    results = verify_pins(bank, BACKBONES, HfApi(), token)
    print(format_table(results))

    fatal = [r for r in results if r.is_fatal]
    blocked = [r for r in results if r.is_blocked]
    print(f"checked {len(results)} pins: {len(fatal)} failed, {len(blocked)} blocked by gating")
    for r in blocked:
        print(f"  BLOCKED-ACCESS: {r.key} -> {r.detail}")
    for r in fatal:
        print(f"  FAILURE: {r.key} -> {r.detail}", file=sys.stderr)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json}")

    if fatal:
        print(
            "\nPIN VERIFICATION FAILED - do not proceed; a pin points at nothing.",
            file=sys.stderr,
        )
        return 1
    if blocked and args.strict:
        print("\nPIN VERIFICATION FAILED (--strict): gated members unreachable.", file=sys.stderr)
        return 1
    suffix = " (gated members above need licence acceptance)" if blocked else ""
    print(f"\nALL PINS VERIFIED{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
