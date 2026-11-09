"""P4 baselines, step 2: score each row of spec section 3 once (one worker per GPU).

Run by ``p2_run_shards.py`` as ``p4_score.py --members A,B --device cuda <args>`` after
``p4_preflight.py`` has written the plan. A row is one (baseline, preprocessing,
condition); rows of the same baseline share one loaded model:

* ``dda_native`` / ``dda_crop308`` -- DDA at ``CenterCrop(336)`` / ``(308)``, clean;
  ``dda_jpeg50`` / ``dda_resize050`` -- DDA native on Panel B's test rows after
  ``dbp.eval.perturb`` on the whole image (the two H5 decision conditions);
* ``cospy_native`` -- CO-SPY's own transform and fused logit;
* ``fire_native`` -- FIRE's ``--resize`` input, one image at a time, the seed rule;
* ``aeroblade`` -- the 768 cap, three AEs through ``reconstruct.py``, minus the minimum
  LPIPS layer-2 distance.

Every score is "larger = more likely fake". Per list, ``image_id, score, flag`` (the row's
preprocessing flag: padded / upscaled / cropped / cut by the cap) goes to
``reports/p4_baseline_scores/<row>/<list>.csv`` (AEROBLADE adds one distance column per
AE), and the summary ``reports/p4_baselines/<row>.json`` is written last, so a summary
means done. **Once only:** a row whose summary exists is refused. ``--limit N`` (the smoke)
scores N rows per list into ``--out-root`` and is never a result.

**Shards** (AEROBLADE; user decision 2026-11-09, execution only): a member ``<row>#<k>``
scores shard ``k``'s manifest (``p4_preflight.build_shards``) into an append-only
checkpoint store (``dbp.baselines.shards.ShardStore``), ``checkpoint_every`` images per
part. On a resume, ``resume_rescore`` saved rows are scored again first and the run stops
if any value differs beyond fp32 rounding; every check is appended to ``resumes.jsonl``.
A finished shard writes ``shard<k>.done.json``; ``p4_merge.py`` builds the row.

**Smoke timing** (fail fast, every row): with ``--limit``, the model first scores one image
untimed (the warm-up), then every smoke image is timed. The summary records ``timing``
(images, seconds, capped pixels) and ``load_seconds``; ``p4_failfast.py`` projects from it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.baselines.common import BaselineError, dryrun  # noqa: E402
from dbp.baselines.shards import (  # noqa: E402
    ShardStore,
    capped_pixels,
    resume_sample,
    within_fp32,
)
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.eval.perturb import CONDITIONS, apply_condition  # noqa: E402
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.watchdog import start_watchdog  # noqa: E402
from p3_s2d_inputs import read_rows  # noqa: E402
from p4_preflight import SCORES, SUMMARIES  # noqa: E402

LOADERS = 4
#: The only host torch.hub may download from in a scoring session: torchvision's VGG16
#: for LPIPS (CLAUDE.md section 6). github.com and dl.fbaipublicfiles.com were allowed for
#: the isolated DDA check only and must never be used here (user, 2026-11-05).
TORCH_HUB_HOSTS = frozenset({"download.pytorch.org"})


def guard_torch_hub() -> None:
    """Refuse ``torch.hub.load`` (it runs downloaded code) and any other download host."""
    from urllib.parse import urlparse

    import torch.hub

    original = torch.hub.download_url_to_file

    def load(*args: Any, **kwargs: Any) -> Any:
        raise ScoreError("torch.hub.load runs downloaded code; refused in a scoring session")

    def download(url: str, *args: Any, **kwargs: Any) -> Any:
        host = urlparse(url).hostname or ""
        if host not in TORCH_HUB_HOSTS:
            raise ScoreError(f"torch.hub download from {host} refused in a scoring session "
                             f"(allowed: {sorted(TORCH_HUB_HOSTS)})")
        return original(url, *args, **kwargs)

    torch.hub.load = load
    torch.hub.download_url_to_file = download


FLAG = {"dda": "padded", "cospy": "upscaled", "fire": "cropped", "aeroblade": "cut_by_cap"}
PREPROCESSING = {
    "dda": "CenterCrop({crop}) (zero-pads under {crop}) -> ToTensor -> CLIP mean/std",
    "cospy": "Resize(384) short side -> CenterCrop(384) -> ToTensor; inner Resize(224, "
             "antialias=False) for the artifact branch",
    "fire": "Resize(256, antialias=True) short side -> long side centre-cropped to a "
            "multiple of 8 -> ToTensor; batch 1; seed rule",
    "aeroblade": "centre crop <= 768 x 768 -> reconstruct.py (multiple of 8) per AE",
}


class ScoreError(RuntimeError):
    """A row that may not or cannot be scored."""


def write_scores(path: Path, rows: list[dict[str, str]], results: list[dict[str, Any]],
                 extra: list[str]) -> str:
    """Atomic ``image_id, score, flag[, extra...]`` CSV; returns its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "score", "flag", *extra])
        for row, result in zip(rows, results, strict=True):
            writer.writerow([row["image_id"], f"{result['score']:.6f}", int(result["flag"]),
                             *(f"{result[k]:.6f}" for k in extra)])
    tmp.replace(path)
    return file_sha256(path)


def refuses_rescore(summary_file: Path, limit: int) -> bool:
    """Once only: a real scoring is refused when its summary exists (a smoke is not)."""
    return summary_file.is_file() and not limit


def weight(plan: dict[str, Any], name: str) -> tuple[Path, str]:
    """A weight file's path and the sha256 the pre-flight verified."""
    entry = plan["weights"][name]
    return Path(entry["path"]), entry["sha256"]


class Scorer:
    """One loaded baseline; ``score(rows, condition)`` returns one result per row."""

    def __init__(self, baseline: str, plan: dict[str, Any], device: str, log: Any) -> None:
        self.baseline, self.plan, self.device, self.log = baseline, plan, device, log
        self.local = Path(plan["local_root"])
        self.info: dict[str, Any] = {}
        started = time.perf_counter()
        getattr(self, f"_load_{baseline}")()
        self.load_seconds = time.perf_counter() - started

    def _arch(self, key: str) -> dict[str, Any]:
        return self.plan.get("dryrun_arch", {}).get(key, {}) if dryrun() else {}

    def _load_dda(self) -> None:
        from dbp.baselines.dda import load_dda

        path, sha = weight(self.plan, "dda")
        self.model = load_dda(path, sha, self.device, self._arch("dda") or None)
        self.info = {"dda_ckpt_sha256": sha}

    def _load_cospy(self) -> None:
        from dbp.baselines.cospy import build_siglip, load_cospy
        from dbp.decoders.loaders import load_member
        from dbp.decoders.registry import load_bank_config

        siglip, sha = weight(self.plan, "siglip")
        arch = self._arch("siglip")
        clip, dim = build_siglip(siglip, sha, arch.get("model", "ViT-SO400M-14-SigLIP-384"),
                                 Path(arch["config"]) if arch.get("config") else None)
        bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
        vae = load_member(bank.by_key("sd15"), self.device, "fp32").model
        names = {k: weight(self.plan, f"cospy_{k}") for k in ("semantic", "artifact",
                                                                "fusion")}
        self.model, vae_record = load_cospy({k: v[0] for k, v in names.items()},
                                            {k: v[1] for k, v in names.items()},
                                            clip.to(self.device), dim, vae, self.device)
        self.info = {"siglip_sha256": sha, "vae": vae_record, "vae_dtype": "fp32",
                     **{f"cospy_{k}_sha256": v[1] for k, v in names.items()}}

    def _load_fire(self) -> None:
        from dbp.baselines.fire import load_fire
        from dbp.decoders.loaders import load_member
        from dbp.decoders.registry import load_bank_config

        bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
        # fp16 as FIRE publishes it (spec 7.13). The local CPU dry-run alone runs fp32: an
        # fp16 SD VAE on a CPU takes hours; Kaggle's GPU smoke exercises the fp16 path.
        precision = "fp32" if dryrun() and self.device == "cpu" else "fp16"
        vae = load_member(bank.by_key("sd15"), self.device, precision).model
        path, sha = weight(self.plan, "fire")
        self.model, source = load_fire(path, sha, vae, self.device)
        self.info = {"fire_sha256": sha, "vae_source": source, "vae_dtype": precision}

    def _load_aeroblade(self) -> None:
        from dbp.baselines.aeroblade import build_lpips
        from dbp.decoders.loaders import load_member
        from dbp.decoders.registry import load_bank_config

        bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
        extra = load_bank_config(THESIS_DBP / "configs" / "p4_aeroblade.yaml")

        def spec(key: str) -> Any:
            for config in (bank, extra):
                if key in {s.key for s in config.members}:
                    return config.by_key(key)
            raise ScoreError(f"AEROBLADE: no AE named {key}")

        self.members = [load_member(spec(k), self.device, "fp32")
                        for k in self.plan["aeroblade_aes"]]
        self.net = build_lpips(self.device)
        self.info = {"aes": {m.spec.key: f"{m.spec.repo}@{m.spec.revision}"
                             for m in self.members}, "lpips": "vgg, v0.1, layer 2"}

    def extra(self) -> list[str]:
        """Additional CSV columns (AEROBLADE: one distance per AE)."""
        return [f"lpips2_{k}" for k in self.plan["aeroblade_aes"]] \
            if self.baseline == "aeroblade" else []

    def score(self, rows: list[dict[str, str]], condition: str, crop: int | None
              ) -> list[dict[str, Any]]:
        """One result (``score``, ``flag``, extras) per row, in order."""
        import torch

        def image(row: dict[str, str]) -> np.ndarray:
            return apply_condition(load_rgb_uint8(self.local / row["relative_path"]),
                                   condition)

        out: list[dict[str, Any]] = []
        if self.baseline in ("dda", "cospy"):
            from dbp.baselines.cospy import cospy_input
            from dbp.baselines.dda import dda_input

            def prepare(row: dict[str, str]) -> tuple[Any, bool]:
                img = image(row)
                return dda_input(img, crop) if self.baseline == "dda" else cospy_input(img)

            size = int(self.plan["batch"][self.baseline])
            with ThreadPoolExecutor(max_workers=LOADERS) as pool:
                for start in range(0, len(rows), size):
                    prepared = list(pool.map(prepare, rows[start: start + size]))
                    batch = torch.stack([p[0] for p in prepared]).to(self.device)
                    with torch.no_grad():
                        logits = self.model(batch).float().cpu().numpy().reshape(-1)
                    out += [{"score": float(v), "flag": p[1]} for v, p in zip(logits, prepared,
                                                                               strict=True)]
        elif self.baseline == "fire":
            from dbp.baselines.fire import score_fire

            for row in rows:
                value, cropped = score_fire(self.model, image(row), row["image_id"],
                                            self.device)
                out.append({"score": value, "flag": cropped})
        else:
            from dbp.baselines.aeroblade import score_aeroblade

            for row in rows:
                value, distances, cut = score_aeroblade(self.members, self.net, image(row),
                                                        self.device)
                out.append({"score": value, "flag": cut,
                            **{f"lpips2_{k}": v for k, v in distances.items()}})
        if not np.all(np.isfinite([r["score"] for r in out])):
            raise ScoreError(f"{self.baseline}: non-finite scores ({condition})")
        return out


def score_row(row_id: str, scorer: Scorer, plan: dict[str, Any], args: argparse.Namespace,
              log: Any) -> dict[str, Any]:
    """Score one row's lists; write its CSVs, then its summary."""
    spec = plan["rows"][row_id]
    out_root = args.out_root or THESIS_DBP
    summary_file = out_root / SUMMARIES / f"{row_id}.json"
    if refuses_rescore(summary_file, args.limit):
        raise ScoreError(f"{row_id}: already scored ({summary_file}); each row is scored once")
    condition = spec.get("condition", "clean")
    if condition not in CONDITIONS:
        raise ScoreError(f"{row_id}: unknown condition {condition}")
    lists = {name: read_rows(Path(plan["list_files"][name]))[1]
             for name in ("panel_b", "luma", "threshold_reals")}
    lists["panel_b_test"] = [r for r in lists["panel_b"] if r["role"] == "test"]
    results: dict[str, Any] = {}
    timing = {"images": 0, "seconds": 0.0, "pixels": 0}
    if args.limit:  # the warm-up image: scored once, untimed, not saved
        scorer.score(lists[spec["lists"][0]][:1], condition, spec.get("crop"))
    for name in spec["lists"]:
        rows = lists[name][: args.limit] if args.limit else lists[name]
        started = time.perf_counter()
        scored = scorer.score(rows, condition, spec.get("crop"))
        seconds = time.perf_counter() - started
        path = out_root / SCORES / row_id / f"{name}.csv"
        results[name] = {"rows": len(rows), "file": str(path),
                         "sha256": write_scores(path, rows, scored, scorer.extra()),
                         FLAG[scorer.baseline]: sum(bool(r["flag"]) for r in scored),
                         "seconds": round(seconds, 1),
                         "seconds_per_image": round(seconds / max(1, len(rows)), 5)}
        log.info("%s %s: %d rows in %.1f s (%.4f s/image); %s %d", row_id, name, len(rows),
                 seconds, seconds / max(1, len(rows)), FLAG[scorer.baseline],
                 results[name][FLAG[scorer.baseline]])
        if args.limit:
            timing["images"] += len(rows)
            timing["seconds"] += seconds
            timing["pixels"] += sum(image_pixels(scorer.local, r) for r in rows)
    summary = describe(scorer, args) | {
        "row": row_id, "condition": condition, "crop": spec.get("crop"),
        "preprocessing": PREPROCESSING[scorer.baseline].format(crop=spec.get("crop")),
        "lists": results}
    if args.limit:
        summary["timing"] = {**timing, "seconds": round(timing["seconds"], 3)}
    write_json(summary_file, summary)
    return summary


def describe(scorer: Scorer, args: argparse.Namespace) -> dict[str, Any]:
    """The fields every summary carries: model, versions, device, smoke or not."""
    import diffusers
    import timm
    import torch
    from PIL import __version__ as pillow

    return {"baseline": scorer.baseline, "score": "larger = more likely fake", **scorer.info,
            "versions": {"torch": torch.__version__, "timm": timm.__version__,
                         "diffusers": diffusers.__version__, "pillow": pillow},
            "device": args.device, "dry_run": dryrun(), "limit": args.limit or None,
            "load_seconds": round(scorer.load_seconds, 1)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def image_pixels(local: Path, row: dict[str, str]) -> int:
    """The image's pixels after AEROBLADE's 768 cap (the unit its time is linear in)."""
    from PIL import Image

    with Image.open(local / row["relative_path"]) as handle:
        return capped_pixels(*handle.size)


def stop_after_parts() -> int:
    """DRY-RUN ONLY: stop a shard after this many parts, to rehearse a deadline (0 = off)."""
    value = os.environ.get("DBP_P4_STOP_AFTER_PARTS", "0")
    return int(value) if dryrun() else 0


def score_shard(unit: str, scorer: Scorer, plan: dict[str, Any], args: argparse.Namespace,
                log: Any) -> dict[str, Any]:
    """Score one shard into its append-only checkpoints, resuming where it stopped."""
    info = plan["units"][unit]
    row, k = info["row"], int(info["shard"])
    spec = plan["shards"][row]
    shard = spec["units"][unit]
    manifest = Path(shard["manifest"])
    if file_sha256(manifest) != shard["sha256"]:
        raise ScoreError(f"{unit}: the shard manifest changed since the pre-flight")
    items = read_rows(manifest)[1]
    if args.limit:  # the smoke: a fresh scratch store, the first N items of each list
        root = args.out_root / SCORES / row / "_shards" / f"shard{k}"
        if root.exists():
            shutil.rmtree(root)
        kept: list[dict[str, str]] = []
        for item in items:
            if sum(1 for i in kept if i["list"] == item["list"]) < args.limit:
                kept.append(item)
        items = kept
    else:
        root = Path(shard["store"])
    done_file = root.parent / f"shard{k}.done.json"
    if done_file.is_file() and not args.limit:
        raise ScoreError(f"{unit}: already complete ({done_file}); each row is scored once")
    columns = ["list", "image_id", "score", "flag", *scorer.extra()]
    store = ShardStore(root, columns)
    meta = {"row": row, "shard": k, "manifest_sha256": shard["sha256"], "columns": columns,
            "aes": plan["aeroblade_aes"]}
    meta_file = root / "meta.json"
    if meta_file.is_file():
        if json.loads(meta_file.read_text("utf-8")) != meta:
            raise ScoreError(f"{unit}: the checkpoints were made for another manifest or "
                             f"model set ({meta_file})")
    else:
        write_json(meta_file, meta)
    saved = store.load()
    for position, row_ in enumerate(saved):
        if (row_["list"], row_["image_id"]) != (items[position]["list"],
                                                items[position]["image_id"]):
            raise ScoreError(f"{unit}: saved row {position} is not the manifest's")
    started = time.perf_counter()
    if saved:  # every resume re-scores saved rows first (user decision 2026-11-09)
        check = resume_check(unit, scorer, items, saved, int(spec["resume_rescore"]), root, log)
        log.info("%s resume: %d saved, %d re-scored, max relative diff %.3g -- %s", unit,
                 len(saved), check["checked"], check["max_rel_diff"],
                 "OK" if check["ok"] else "DIFFERENT")
        if not check["ok"]:
            raise ScoreError(f"{unit}: a re-scored saved row differs beyond fp32 rounding "
                             f"({check['worst']}); stop (user decision 2026-11-09)")
    elif args.limit:
        scorer.score(items[:1], "clean", None)  # the warm-up image, untimed
    every = int(spec["checkpoint_every"])
    remaining = items[len(saved):]
    timing = {"images": 0, "seconds": 0.0, "pixels": 0}
    for parts, start in enumerate(range(0, len(remaining), every), start=1):
        chunk = remaining[start: start + every]
        began = time.perf_counter()
        scored = scorer.score(chunk, "clean", None)
        timing["seconds"] += time.perf_counter() - began
        timing["images"] += len(chunk)
        timing["pixels"] += sum(int(i["pixels"]) for i in chunk)
        store.append([{"list": i["list"], "image_id": i["image_id"], "score": r["score"],
                       "flag": int(bool(r["flag"])), **{c: r[c] for c in scorer.extra()}}
                      for i, r in zip(chunk, scored, strict=True)])
        log.info("%s: %d/%d saved (%.3f s/image)", unit, len(saved) + start + len(chunk),
                 len(items), timing["seconds"] / max(1, timing["images"]))
        if not args.limit and stop_after_parts() and parts >= stop_after_parts():
            raise ScoreError(f"{unit}: DRY-RUN stop after {parts} part(s) (a rehearsed deadline)")
    if args.limit:  # the smoke also rehearses a resume on what it just saved
        check = resume_check(unit, scorer, items, store.load(), 1, root, log)
        if not check["ok"]:
            raise ScoreError(f"{unit}: the smoke's resume check differs ({check['worst']})")
    record = describe(scorer, args) | {
        "unit": unit, "row": row, "shard": k, "items": len(items),
        "pixels": sum(int(i["pixels"]) for i in items), "parts": len(store.entries()),
        "ledger_sha256": file_sha256(store.ledger), "manifest_sha256": shard["sha256"],
        "seconds": round(time.perf_counter() - started, 1),
        "timing": {**timing, "seconds": round(timing["seconds"], 3)}}
    if args.limit:
        write_json(args.out_root / SUMMARIES / f"{unit}.json", record)
    else:
        write_json(done_file, record)
    return record


def resume_check(unit: str, scorer: Scorer, items: list[dict[str, str]],
                 saved: list[dict[str, str]], count: int, root: Path, log: Any
                 ) -> dict[str, Any]:
    """Re-score ``count`` saved rows; append the outcome to ``resumes.jsonl``."""
    picks = resume_sample(len(saved), count)
    again = scorer.score([items[p] for p in picks], "clean", None)
    worst: dict[str, Any] = {}
    max_rel, ok = 0.0, True
    for position, fresh in zip(picks, again, strict=True):
        for column in ["score", *scorer.extra()]:
            old, new = float(saved[position][column]), float(fresh[column])
            rel = abs(old - new) / max(abs(old), abs(new), 1e-30)
            if rel > max_rel:
                max_rel = rel
                worst = {"image_id": saved[position]["image_id"], "column": column,
                         "saved": old, "rescored": new}
            ok = ok and within_fp32(old, new)
    check = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "unit": unit,
             "saved_rows": len(saved), "checked": len(picks), "positions": picks,
             "max_rel_diff": max_rel, "worst": worst, "ok": ok}
    with (root / "resumes.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(check) + "\n")
    return check


def main() -> int:
    """Score every row in ``--members``; 0 when all were scored."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--members", required=True, help="Comma-separated row ids.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan", type=Path, default=THESIS_DBP / "reports" / "p4_plan.json")
    parser.add_argument("--limit", type=int, default=0, help="Smoke: N rows per list.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Where a smoke writes (never the results tree).")
    parser.add_argument("--deadline-hours", type=float, default=0.0)
    args = parser.parse_args()
    log = setup_logging("p4_score")
    guard_torch_hub()
    if args.deadline_hours > 0:
        start_watchdog(args.deadline_hours, log)
    if args.limit and args.out_root is None:
        log.error("--limit is a smoke and needs --out-root outside the results tree")
        return 2
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not plan.get("ok") or plan.get("stage") != "score":
        log.error("the plan is not ok or has nothing to score: %s", plan.get("stage"))
        return 1
    scorers: dict[str, Scorer] = {}
    for unit in [m for m in args.members.split(",") if m]:
        if unit not in plan["due"]:
            log.error("%s is not due in this plan", unit)
            return 1
        info = plan["units"][unit]
        baseline = plan["rows"][info["row"]]["baseline"]
        try:
            if baseline not in scorers:
                for old in list(scorers):  # one baseline in memory at a time
                    del scorers[old]
                _free()
                scorers[baseline] = Scorer(baseline, plan, args.device, log)
            if info["shard"] is None:
                score_row(unit, scorers[baseline], plan, args, log)
            else:
                score_shard(unit, scorers[baseline], plan, args, log)
        except (ScoreError, BaselineError) as error:
            log.error("%s", error)
            return 1
    return 0


def _free() -> None:
    """Release cached GPU memory between baselines."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
