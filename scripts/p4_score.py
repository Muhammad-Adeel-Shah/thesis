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
"""

from __future__ import annotations

import argparse
import csv
import json
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
        getattr(self, f"_load_{baseline}")()

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
    import torch
    from PIL import __version__ as pillow

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
    import diffusers
    import timm

    summary = {
        "row": row_id, "baseline": scorer.baseline, "condition": condition,
        "crop": spec.get("crop"),
        "preprocessing": PREPROCESSING[scorer.baseline].format(crop=spec.get("crop")),
        "score": "larger = more likely fake", **scorer.info,
        "versions": {"torch": torch.__version__, "timm": timm.__version__,
                     "diffusers": diffusers.__version__, "pillow": pillow},
        "device": args.device, "dry_run": dryrun(), "limit": args.limit or None,
        "lists": results,
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_file.with_name(summary_file.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(summary_file)
    return summary


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
    for row_id in [m for m in args.members.split(",") if m]:
        if row_id not in plan["due"]:
            log.error("%s is not due in this plan", row_id)
            return 1
        baseline = plan["rows"][row_id]["baseline"]
        try:
            if baseline not in scorers:
                for old in list(scorers):  # one baseline in memory at a time
                    del scorers[old]
                _free()
                scorers[baseline] = Scorer(baseline, plan, args.device, log)
            score_row(row_id, scorers[baseline], plan, args, log)
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
