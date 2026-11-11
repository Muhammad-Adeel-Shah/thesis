"""Open Images V7 real images from the Flickr originals (P3 Phase B step 3).

The rules are the user's (2026-10-11 and 2026-10-13, ``docs/P3_PLAN.md`` §4a, rules
(a)–(f)):

* **(a) Hosts.** Only ``OriginalURL`` is fetched, and only from an approved host:
  ``farm1``–``farm8.staticflickr.com`` (approved 2026-10-11) and ``c1``–``c8.staticflickr.com``
  (approved 2026-10-13: the same Flickr CDN, and the metadata splits about 50/50 between
  the two, undocumented, so a one-group draw could be biased). A redirect may land on an
  approved host or on ``live.staticflickr.com`` and nowhere else. Any other host is **not
  contacted**: the row is rejected and its host listed.
* **Politeness.** At most 8 HTTP connections are open at once (:data:`MAX_PARALLEL`, a
  semaphore in :class:`Fetcher`, whatever the number of threads). Each request retries
  with exponential backoff, and a shared :class:`Throttle` slows **every** request down
  when Flickr refuses (403/429/503) -- once per wave, honouring ``Retry-After`` -- logs
  each change with the first 20 refusals as evidence, and stops the walk only after
  10 minutes of sustained refusals at the 60 s ceiling, rather than hammer the server.
* **(b) Checksum.** Every file must match the metadata's ``OriginalMD5`` (base64 MD5); a
  mismatch means Flickr served a re-encode, and the row is rejected.
* **(c) Seeded over-draw.** All CC BY 2.0 rows are put in one seed-2026 order
  (:func:`order_key`) and walked until the target is met; the held-out split is seeded
  separately (:func:`split_key`).

Nothing here resizes. Cropping is ``p1_build_reals.build_rows``, the COCO pipeline's own
function, so the new reals are made exactly as T-real was.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import platform
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

#: The only licence kept (rule: per-image CC BY 2.0).
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
OI_SEED = 2026
OI_ORDER_SALT = "open_images"
OI_SPLIT_SALT = "open_images_split"
#: Approved hosts for the original URL: farm1-8 (user, 2026-10-11) and c1-c8 (2026-10-13).
APPROVED_HOST_PATTERNS: tuple[str, ...] = (r"farm[1-8]\.staticflickr\.com",
                                           r"c[1-8]\.staticflickr\.com")
#: Approved only as the target of a redirect.
REDIRECT_HOSTS: tuple[str, ...] = ("live.staticflickr.com",)
#: The CVDF copies of Open Images V7 (user decision 2026-10-15, reversing the 2026-10-11
#: host choice after Flickr refused every route): the public bucket, no credentials, and
#: a regional endpoint of the same bucket if S3 redirects there. The copies are
#: "rescaled to have at most 1024 pixels on their longest side" (CVDF README).
S3_HOST_PATTERNS: tuple[str, ...] = (r"open-images-dataset\.s3\.amazonaws\.com",
                                     r"open-images-dataset\.s3[.-][a-z0-9-]+\.amazonaws\.com")
S3_BASE = "https://open-images-dataset.s3.amazonaws.com"
#: Recorded in every manifest row: which processing history a crop comes from.
SOURCE_COPY = {"s3": "cvdf_s3_rescaled_max1024px", "flickr": "flickr_original"}
#: Most HTTP connections open at once (user rule).
MAX_PARALLEL = 8
MAX_REDIRECTS = 3
#: Status codes that mean "slow down" (403 too: a blanket block must stop the walk,
#: not be retried row by row across 41,620 rows).
REFUSAL_CODES = frozenset({403, 429, 503})
USER_AGENT = "dbp-thesis-research/1.0 (academic; Open Images V7 originals)"


def order_key(image_id: str, seed: int = OI_SEED) -> str:
    """Seeded, platform-independent walk order over the metadata rows."""
    text = f"{OI_ORDER_SALT}:{seed}:{image_id}".encode()
    return hashlib.blake2b(text, digest_size=16).hexdigest()


def split_key(image_id: str, seed: int = OI_SEED) -> str:
    """Seeded key for the train / held-out split, independent of the walk order."""
    text = f"{OI_SPLIT_SALT}:{seed}:{image_id}".encode()
    return hashlib.blake2b(text, digest_size=16).hexdigest()


def load_metadata(path: Path, seed: int = OI_SEED) -> list[dict[str, str]]:
    """The CC BY 2.0 rows of an Open Images metadata CSV, in seed order."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("License") == CC_BY_2]
    return sorted(rows, key=lambda r: order_key(r["ImageID"], seed))


def s3_url(image_id: str, split: str = "validation") -> str:
    """The CVDF S3 copy of one Open Images image (the official downloader's key scheme)."""
    if not re.fullmatch(r"[0-9a-f]{16}", image_id):
        raise ValueError(f"not an Open Images image id: {image_id!r}")
    return f"{S3_BASE}/{split}/{image_id}.jpg"


def host_of(url: str) -> str:
    """The lower-case host of a URL ("" if none)."""
    return (urlparse(url).hostname or "").lower()


def host_approved(host: str, patterns: Sequence[str] = APPROVED_HOST_PATTERNS) -> bool:
    """Whether ``host`` fully matches one approved pattern."""
    return any(re.fullmatch(p, host) for p in patterns)


def host_group(host: str) -> str:
    """``farm*`` (farm1-8), ``c*`` (c1-c8) or ``other``: the metadata's two host groups."""
    if re.fullmatch(APPROVED_HOST_PATTERNS[0], host):
        return "farm*"
    if re.fullmatch(APPROVED_HOST_PATTERNS[1], host):
        return "c*"
    return "other"


def colour_mode_reason(mode: str, info: dict | None = None) -> str | None:
    """Why an original's colour mode is rejected, or None for plain 8-bit RGB.

    User rule (2026-10-14): colour modes are **never converted**. Only 8-bit RGB is kept
    (a baseline/progressive colour JPEG decodes as ``RGB``); CMYK, 16-bit or float,
    palette, alpha and single-channel greyscale originals are rejected, each counted
    under its own reason. (Colour images whose three channels are equal are rejected
    later by the COCO pipeline's own greyscale rule, D-039.)
    """
    if mode == "RGB":
        return "alpha" if info and "transparency" in info else None
    if mode == "CMYK":
        return "CMYK"
    if mode in ("P", "PA"):
        return "palette"
    if "A" in mode or mode.endswith("a"):
        return "alpha"
    if mode.startswith("I") or mode == "F":
        return "16-bit or float"
    if mode in ("L", "1"):
        return "greyscale (mode L/1)"
    return f"other ({mode})"


def decoder_versions() -> dict[str, str]:
    """The Python, Pillow and libjpeg(-turbo) that decode the originals."""
    import PIL
    from PIL import features

    turbo = features.version_feature("libjpeg_turbo") if features.check_feature(
        "libjpeg_turbo") else None
    jpeg = features.version_codec("jpg")
    return {"decoder_python": platform.python_version(), "decoder_pillow": PIL.__version__,
            "decoder_libjpeg": f"libjpeg-turbo {turbo} (jpeglib {jpeg})" if turbo
            else f"libjpeg {jpeg}"}


def md5_b64(data_md5: bytes) -> str:
    """Base64 of a raw MD5 digest, the form ``OriginalMD5`` uses."""
    return base64.b64encode(data_md5).decode("ascii")


def assign_split(image_ids: Iterable[str], holdout: int, seed: int = OI_SEED) -> dict[str, str]:
    """``held_out`` for the ``holdout`` ids with the smallest split key, ``train`` otherwise."""
    ordered = sorted(image_ids, key=lambda i: split_key(i, seed))
    if holdout > len(ordered):
        raise ValueError(f"cannot hold out {holdout} of {len(ordered)} images")
    return {i: ("held_out" if n < holdout else "train") for n, i in enumerate(ordered)}


class Throttle:
    """A delay shared by every request, raised on refusals and relaxed on success.

    Revised after Kaggle run 1 (2026-10-13), where 8 simultaneous refusals from one
    parallel wave doubled the delay 8 times in 0.2 s and the stop rule fired after 3 waves:

    * **One escalation per wave.** A refusal raises the delay only if the last raise is at
      least one current delay old; refusals of the same wave are counted but do not raise
      it again. The raise doubles the delay (from ``base`` up to ``ceiling``), or sets it to
      the server's ``Retry-After`` when that is longer.
    * **Stop only on sustained refusals.** The throttle trips once refusals have continued
      for ``trip_seconds`` (10 minutes) at the ceiling with no success in between; any
      success resets that clock. The walk then stops rather than keep sending requests.
    * ``relax_after`` consecutive successes halve the delay.
    * The first ``sample_limit`` (20) refusals are kept as evidence (status, time, headers
      without cookies, first 200 body bytes).

    Refusals are 403, 429 and 503 (:data:`REFUSAL_CODES`).
    """

    def __init__(self, base: float = 1.0, ceiling: float = 60.0, relax_after: int = 50,
                 trip_seconds: float = 600.0, sample_limit: int = 20,
                 log: Callable[[str], None] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.base, self.ceiling = base, ceiling
        self.relax_after, self.trip_seconds = relax_after, trip_seconds
        self.sample_limit = sample_limit
        self.log = log or (lambda message: None)
        self.sleep = sleep
        self.clock = clock
        self.delay = 0.0
        self.refusals = 0
        self.raises = 0
        self.max_delay = 0.0
        self.tripped = False
        self.events: list[str] = []
        self.samples: list[dict[str, object]] = []
        self._streak = 0
        self._last_raise: float | None = None
        self._ceiling_since: float | None = None
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Sleep the current shared delay before a request."""
        with self._lock:
            delay = self.delay
        if delay > 0:
            self.sleep(delay)

    def refused(self, code: int, retry_after: str | None = None,
                sample: dict[str, object] | None = None) -> None:
        """Record a refusal; raise the shared delay at most once per wave."""
        with self._lock:
            now = self.clock()
            self.refusals += 1
            self._streak = 0
            if sample is not None and len(self.samples) < self.sample_limit:
                self.samples.append(sample)
            hinted = float(retry_after) if retry_after and retry_after.isdigit() else 0.0
            same_wave = self._last_raise is not None and now - self._last_raise < self.delay
            message = ""
            if not same_wave or hinted > self.delay:
                new = min(self.ceiling, max(self.base, 2 * self.delay, hinted))
                self.delay = new
                self.raises += 1
                self._last_raise = now
                self.max_delay = max(self.max_delay, new)
                message = f"http {code}: slowing down, shared delay now {new:.2f} s " \
                          f"(refusal {self.refusals}, raise {self.raises})"
            if self.delay >= self.ceiling:
                if self._ceiling_since is None:
                    self._ceiling_since = now
                sustained = now - self._ceiling_since
                if sustained >= self.trip_seconds and not self.tripped:
                    self.tripped = True
                    message = (f"http {code}: refusals sustained for {sustained:.0f} s at the "
                               f"{self.ceiling:.0f} s ceiling (refusal {self.refusals}) -- "
                               "stopping the walk")
            if message:
                self.events.append(message)
        if message:
            self.log(message)

    def succeeded(self) -> None:
        """Record a success; relax the delay after a run of them."""
        with self._lock:
            self._streak += 1
            self._ceiling_since = None
            if self.delay > 0 and self._streak >= self.relax_after:
                self.delay = 0.0 if self.delay / 2 < self.base else self.delay / 2
                self._streak = 0
                message = f"{self.relax_after} successes: shared delay now {self.delay:.2f} s"
                self.events.append(message)
            else:
                message = ""
        if message:
            self.log(message)

    def summary(self) -> dict[str, object]:
        """What the throttle did, for the run summary."""
        return {"refusals": self.refusals, "raises": self.raises,
                "max_delay_s": self.max_delay, "final_delay_s": self.delay,
                "tripped": self.tripped, "trip_after_s_at_ceiling": self.trip_seconds,
                "events": self.events[:200], "events_total": len(self.events),
                "refusal_samples": self.samples}


#: Response headers never recorded (user rule: no cookies).
_PRIVATE_HEADERS = frozenset({"set-cookie", "cookie"})


def refusal_sample(url: str, code: int, headers, body: bytes) -> dict[str, object]:  # noqa: ANN001
    """Evidence of one refusal: time, host, status, headers without cookies, 200 body bytes."""
    items = headers.items() if hasattr(headers, "items") else []
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": host_of(url),
            "status": code,
            "headers": {k: v for k, v in items if k.lower() not in _PRIVATE_HEADERS},
            "body_head": body[:200].decode("latin-1")}


def _read_head(exc: urllib.error.HTTPError, limit: int = 200) -> bytes:
    """The first bytes of an error response's body ("" when there is none)."""
    try:
        return exc.read(limit) or b""
    except (OSError, AttributeError, ValueError):
        return b""


@dataclass
class FetchResult:
    """What one download attempt produced."""

    status: str  # "ok", "gone", "failed", "host", "stopped"
    detail: str = ""
    size: int = 0
    md5_b64: str = ""
    sha256: str = ""
    final_host: str = ""
    attempts: int = 0
    hosts_seen: list[str] = field(default_factory=list)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirects are returned, not followed, so every hop's host can be checked first."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class Fetcher:
    """Polite downloader that only ever contacts approved hosts.

    At most :data:`MAX_PARALLEL` connections are open at once: a bounded semaphore is held
    from the request until the body has been read, so the cap holds however many threads
    call :meth:`fetch`. ``min_interval`` (seconds, user-approved 2026-10-14) is a fixed
    minimum gap between the **starts** of any two requests, retries and redirects included,
    across all threads -- a steady pace rather than bursts.
    """

    def __init__(self, patterns: Sequence[str] = APPROVED_HOST_PATTERNS,
                 redirect_hosts: Sequence[str] = REDIRECT_HOSTS, timeout: float = 60.0,
                 retries: int = 3, backoff: float = 2.0, throttle: Throttle | None = None,
                 max_parallel: int = MAX_PARALLEL, min_interval: float = 0.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if max_parallel > MAX_PARALLEL:
            raise ValueError(f"at most {MAX_PARALLEL} parallel connections, got {max_parallel}")
        self.patterns = tuple(patterns)
        self.redirect_hosts = tuple(h.lower() for h in redirect_hosts)
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.throttle = throttle or Throttle()
        self.max_parallel = max_parallel
        self._slots = threading.BoundedSemaphore(max_parallel)
        self._opener = urllib.request.build_opener(_NoRedirect)
        if min_interval < 0:
            raise ValueError(f"min_interval must be >= 0, got {min_interval}")
        self.min_interval = min_interval
        self.clock = clock
        self._pace_lock = threading.Lock()
        self._next_start: float | None = None
        self.requests_started = 0

    def _pace(self) -> None:
        """Wait until ``min_interval`` has passed since the previous request started."""
        with self._pace_lock:
            now = self.clock()
            if self._next_start is not None and now < self._next_start:
                self.throttle.sleep(self._next_start - now)
                now = self.clock()
            self._next_start = now + self.min_interval
            self.requests_started += 1

    def _open(self, url: str):  # noqa: ANN202 - a urllib response
        """One request; raises ``urllib.error.HTTPError`` for non-2xx (3xx included)."""
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        return self._opener.open(request, timeout=self.timeout)

    def fetch(self, url: str, destination: Path) -> FetchResult:  # noqa: PLR0911
        """Download ``url`` to ``destination``, checking every host before contacting it."""
        result = FetchResult(status="failed")
        host = host_of(url)
        if not host_approved(host, self.patterns):
            return FetchResult(status="host", detail=host)
        hops = 0
        attempt = 0
        while True:
            if self.throttle.tripped:
                return FetchResult(status="stopped", detail="throttle tripped; not contacted",
                                   attempts=attempt, hosts_seen=result.hosts_seen)
            result.hosts_seen.append(host)
            attempt += 1
            result.attempts = attempt
            self.throttle.wait()
            self._pace()
            try:
                with self._slots, self._open(url) as response:
                    saved = self._save(response, destination, host, result)
                self.throttle.succeeded()
                return saved
            except urllib.error.HTTPError as exc:
                if exc.code in (301, 302, 303, 307, 308):
                    target = urljoin(url, exc.headers.get("Location", ""))
                    target_host = host_of(target)
                    hops += 1
                    if hops > MAX_REDIRECTS:
                        return FetchResult(status="failed", detail="too many redirects",
                                           attempts=attempt, hosts_seen=result.hosts_seen)
                    if not (host_approved(target_host, self.patterns)
                            or target_host in self.redirect_hosts):
                        return FetchResult(status="host", detail=f"redirect:{target_host}",
                                           attempts=attempt, hosts_seen=result.hosts_seen)
                    url, host, attempt = target, target_host, 0
                    continue
                if exc.code in (404, 410):
                    return FetchResult(status="gone", detail=f"http {exc.code}",
                                       attempts=attempt, hosts_seen=result.hosts_seen)
                if exc.code in REFUSAL_CODES:
                    self.throttle.refused(
                        exc.code, exc.headers.get("Retry-After"),
                        refusal_sample(url, exc.code, exc.headers, _read_head(exc)))
                elif exc.code < 500:
                    return FetchResult(status="failed", detail=f"http {exc.code}",
                                       attempts=attempt, hosts_seen=result.hosts_seen)
                reason = f"http {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                reason = f"{type(exc).__name__}: {str(exc)[:80]}"
            if attempt > self.retries:
                return FetchResult(status="failed", detail=f"{reason} after {attempt} attempts",
                                   attempts=attempt, hosts_seen=result.hosts_seen)
            self.throttle.sleep(self.backoff ** attempt)

    @staticmethod
    def _save(response, destination: Path, host: str, result: FetchResult) -> FetchResult:  # noqa: ANN001
        """Stream a response to disk, hashing as it goes; atomic rename at the end."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        md5 = hashlib.md5(usedforsecurity=False)
        sha = hashlib.sha256()
        size = 0
        with partial.open("wb") as handle:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                handle.write(chunk)
                md5.update(chunk)
                sha.update(chunk)
                size += len(chunk)
        os.replace(partial, destination)
        result.status = "ok"
        result.size = size
        result.md5_b64 = md5_b64(md5.digest())
        result.sha256 = sha.hexdigest()
        result.final_host = host
        return result


class LocalFetcher(Fetcher):
    """Dry-run only: serves ``<source_dir>/<URL file name>`` instead of the network.

    Host rules, the connection cap and the throttle work exactly as for the network: a
    fixture row on an unapproved host is rejected the same way; a missing file answers
    404; a sidecar ``<file>.429`` holding ``n`` answers 429 (``Retry-After: 0``) ``n`` times
    first, so the slow-down path runs in the dry-run.
    """

    def __init__(self, source_dir: Path, **kwargs) -> None:  # noqa: ANN003
        kwargs.setdefault("throttle", Throttle(base=0.01, ceiling=0.05))
        super().__init__(**kwargs)
        self.source_dir = Path(source_dir)
        self.backoff = 0.0
        self._sidecar_lock = threading.Lock()

    def _open(self, url: str):  # noqa: ANN202
        path = self.source_dir / Path(urlparse(url).path).name
        sidecar = path.with_name(path.name + ".429")
        with self._sidecar_lock:
            if sidecar.is_file():
                left = int(sidecar.read_text(encoding="utf-8").strip() or 0)
                if left > 0:
                    sidecar.write_text(str(left - 1), encoding="utf-8")
                    raise urllib.error.HTTPError(url, 429, "fixture throttle",
                                                 {"Retry-After": "0"}, None)
        if not path.is_file():
            raise urllib.error.HTTPError(url, 404, "not in the fixture", {}, None)
        return path.open("rb")
