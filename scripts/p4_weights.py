"""P4 step 1 (Kaggle CPU, Internet on): fetch and check every baseline weight, before any score.

Spec ``docs/P4_BASELINES_SPEC.md`` sections 1 and 4. Nothing is scored here. For each file:

* **Hugging Face** (pinned revisions; ``hf_hub_download``): ``DDA_ckpt.pth``, CO-SPY's
  ``sd-v1_4.zip`` and SigLIP's ``open_clip_model.safetensors``. Each must have the size and
  sha256 recorded in the spec, or the run stops.
* **FIRE** (Google Drive, ``gdown`` installed by the bootstrap with ``--no-deps``, session
  only): the "Imagenet with adm" checkpoint. Its sha256 is **recorded here** -- there is no
  published hash -- and must be committed before any baseline is scored. **If Google Drive
  refuses, the run stops and says so; there is no mirror.**
* The three ``.pth`` files are extracted from the CO-SPY zip by name (nothing else is
  extracted) and hashed.

Then every checkpoint is opened with ``torch.load(weights_only=True)`` and checked against
the adapters in ``dbp.baselines`` -- the same checks the scoring makes, run here on a CPU so
a mismatch costs no GPU time:

* DDA: the exact key set, then a strict load (spec 7.12);
* CO-SPY: the semantic and fusion ``fc`` keys; the artifact keys against the artifact
  branch, and **whether the file holds VAE keys** (which the spec says stops the run) --
  if it does, whether those tensors equal the pinned ``sd15`` VAE's, so the user can
  decide; SigLIP built from the local file;
* FIRE: every ResNet and filter key; how much of the VAE it holds (``full``/``none``/
  ``partial``), and if full, whether it equals the pinned ``sd15`` VAE cast to fp16.

Writes ``reports/p4_weights.json`` (always) and keeps the files under ``--out`` (the
notebook output the scoring session attaches). Exit 0 only when every file is present,
hash-checked and loads; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.baselines.common import (  # noqa: E402
    BaselineError,
    dryrun,
    file_sha256,
    key_diff,
    load_pth,
)
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import write_record  # noqa: E402

REPORT = Path("reports") / "p4_weights.json"


def fetch_hf(spec: dict[str, Any], out: Path, local_source: Path | None) -> Path:
    """One pinned Hub file into ``out`` (dry-run: copied from ``local_source``)."""
    target = out / spec["file"]
    out.mkdir(parents=True, exist_ok=True)
    if local_source is not None:
        shutil.copyfile(local_source / spec["file"], target)
        return target
    from huggingface_hub import hf_hub_download

    from dbp.utils.hf import get_hf_token

    path = hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                           local_dir=str(out), token=get_hf_token())
    return Path(path)


def fetch_fire(spec: dict[str, Any], out: Path, local_source: Path | None
               ) -> tuple[Path | None, str | None]:
    """FIRE's checkpoint by ``gdown`` (dry-run: copied).

    Returns the path, or None and the reason when Google Drive refuses.
    """
    target = out / spec["file"]
    out.mkdir(parents=True, exist_ok=True)
    if local_source is not None:
        shutil.copyfile(local_source / spec["file"], target)
        return target, None
    import gdown

    try:
        result = gdown.download(id=spec["gdrive_id"], output=str(target), quiet=False)
    except Exception as error:  # gdown raises its own types for a refused download
        return None, f"{type(error).__name__}: {error}"
    if not result or not target.is_file():
        return None, f"gdown returned {result!r} and wrote no file"
    return target, None


def check_file(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Size and sha256 against the recorded values."""
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    record["ok"] = record["bytes"] == spec["bytes"] and record["sha256"] == spec["sha256"]
    return record


def extract_cospy(zip_path: Path, names: list[str], out: Path) -> tuple[dict[str, Any],
                                                                         list[str]]:
    """The three weight files, by base name, from the CO-SPY zip (nothing else)."""
    record: dict[str, Any] = {}
    problems: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        record["members"] = members
        for name in names:
            hits = [m for m in members if m.rsplit("/", 1)[-1] == name]
            if len(hits) != 1:
                problems.append(f"CO-SPY zip: {len(hits)} member(s) named {name}")
                continue
            target = out / name
            with archive.open(hits[0]) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            record[name] = {"member": hits[0], "path": str(target),
                            "bytes": target.stat().st_size, "sha256": file_sha256(target)}
    return record, problems


def tensors_equal(state: dict[str, Any], reference: dict[str, Any], prefix: str,
                  dtype: Any | None = None) -> dict[str, Any]:
    """Whether ``state``'s ``prefix`` tensors equal ``reference`` (cast to ``dtype``)."""
    import torch

    held = {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    if set(held) != set(reference):
        return {"same_keys": False, "equal": False,
                **{f"only_{s}": n for s, n in (("file", len(set(held) - set(reference))),
                                                 ("pinned", len(set(reference) - set(held))))}}
    worst = 0.0
    for key, tensor in held.items():
        ref = reference[key].to(dtype) if dtype is not None else reference[key]
        if tensor.shape != ref.shape:
            return {"same_keys": True, "equal": False, "shape_mismatch": key}
        worst = max(worst, float((tensor.float() - ref.float()).abs().max()))
    equal = all(torch.equal(held[k], reference[k].to(dtype) if dtype is not None
                            else reference[k]) for k in held)
    return {"same_keys": True, "equal": equal, "max_abs_diff": worst}


def inspect_all(paths: dict[str, Path], expect: dict[str, Any], sd15_state: dict[str, Any],
                sd15_vae: Any, log: Any) -> tuple[dict[str, Any], list[str]]:
    """Open every checkpoint with weights_only=True and check it against the adapters."""
    import torch

    from dbp.baselines import cospy, dda, fire

    record: dict[str, Any] = {}
    problems: list[str] = []
    arch = expect.get("dryrun_dda_arch") if dryrun() else None

    # DDA: exact keys, then a strict load.
    checkpoint = load_pth(paths["dda"], expect["hf"]["dda"]["sha256"], "DDA_ckpt.pth")
    state = checkpoint.get("model", {}) if isinstance(checkpoint, dict) else {}
    model = dda.DDADetector(arch)
    diff = key_diff(model.state_dict().keys(), state.keys())
    record["dda"] = {"top_level": sorted(checkpoint) if isinstance(checkpoint, dict) else None,
                     "keys": len(state), **{k: v[:20] for k, v in diff.items()},
                     "n_missing": len(diff["missing"]), "n_unexpected": len(diff["unexpected"])}
    if diff["missing"] or diff["unexpected"]:
        problems.append(f"DDA: {len(diff['missing'])} missing, {len(diff['unexpected'])} "
                        f"unexpected key(s)")
    else:
        model.load_state_dict(state, strict=True)
        record["dda"]["strict_load"] = True
    del model, checkpoint, state
    log.info("DDA: %s", {k: record["dda"][k] for k in ("keys", "n_missing", "n_unexpected")})

    # CO-SPY: fc keys, the artifact branch, VAE keys; SigLIP from the local file.
    sha = {k: file_sha256(paths[f"cospy_{k}"]) for k in ("semantic", "artifact", "fusion")}
    states = {k: load_pth(paths[f"cospy_{k}"], sha[k], f"CO-SPY {k}")
              for k in ("semantic", "artifact", "fusion")}
    art = states["artifact"]
    expected = [k for k in cospy.ArtifactDetector(None).state_dict()]
    non_vae = {k: v for k, v in art.items() if not k.startswith(cospy.VAE_PREFIX)}
    diff = key_diff(expected, non_vae.keys())
    vae_keys = len(art) - len(non_vae)
    record["cospy"] = {"semantic_keys": sorted(states["semantic"]),
                       "fusion_keys": sorted(states["fusion"]), "artifact_keys": len(art),
                       "artifact_vae_keys": vae_keys,
                       "artifact_non_vae": {**{k: v[:20] for k, v in diff.items()}}}
    for name in ("semantic", "fusion"):
        if not {"fc.weight", "fc.bias"} <= set(states[name]):
            problems.append(f"CO-SPY {name}: no fc.weight/fc.bias")
    if diff["missing"] or diff["unexpected"]:
        problems.append(f"CO-SPY artifact: {len(diff['missing'])} missing, "
                        f"{len(diff['unexpected'])} unexpected non-VAE key(s)")
    if vae_keys:
        record["cospy"]["artifact_vae_vs_pinned_sd15"] = tensors_equal(
            art, sd15_state, cospy.VAE_PREFIX)
        problems.append(f"CO-SPY artifact_weights.pth holds {vae_keys} VAE key(s): the spec "
                        f"(section 1) stops here; see artifact_vae_vs_pinned_sd15 and decide")
    siglip = expect["hf"]["siglip"]
    clip, dim = cospy.build_siglip(paths["siglip"], siglip["sha256"],
                                   expect.get("siglip_model", cospy.SIGLIP),
                                   Path(expect["dryrun_siglip_config"]) if dryrun() and
                                   expect.get("dryrun_siglip_config") else None)
    record["cospy"]["siglip_embed_dim"] = dim
    del clip, states, art
    log.info("CO-SPY: %s", {k: record["cospy"][k] for k in ("artifact_keys",
                                                            "artifact_vae_keys")})

    # FIRE: every ResNet and filter key; how much of the VAE it holds.
    state = load_pth(paths["fire"], file_sha256(paths["fire"]), "FIRE checkpoint")
    if not isinstance(state, dict):
        problems.append(f"FIRE: the checkpoint is a {type(state).__name__}, not a state dict")
        state = {}
    model = fire.FIREDetector(sd15_vae)
    held = fire.vae_keys_held(state, list(sd15_state))
    expected = [k for k in model.state_dict() if not k.startswith("vae.")]
    diff = key_diff(expected, [k for k in state if not k.startswith("vae.")])
    record["fire"] = {"keys": len(state), "vae": held,
                      **{k: v[:20] for k, v in diff.items()}}
    if diff["missing"] or diff["unexpected"]:
        problems.append(f"FIRE: {len(diff['missing'])} missing, {len(diff['unexpected'])} "
                        f"unexpected non-VAE key(s)")
    if held == "partial":
        problems.append("FIRE: the checkpoint holds part of the VAE")
    if held == "full":
        record["fire"]["vae_vs_pinned_sd15_fp16"] = tensors_equal(state, sd15_state, "vae.",
                                                                  torch.float16)
    log.info("FIRE: %s", {k: record["fire"][k] for k in ("keys", "vae")})
    return record, problems


def main() -> int:
    """Fetch, hash, extract and inspect. Exit 0 when every check holds."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Where the weights are kept (the notebook output).")
    parser.add_argument("--local-source", type=Path, default=None,
                        help="DRY-RUN ONLY (DBP_DRYRUN=1): copy the files from here.")
    parser.add_argument("--report", type=Path, default=THESIS_DBP / REPORT)
    args = parser.parse_args()
    log = setup_logging("p4_weights")
    if args.local_source is not None and not dryrun():
        log.error("--local-source is for the dry-run only (DBP_DRYRUN=1)")
        return 2
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    record: dict[str, Any] = {"what": "P4 baseline weights: fetched, hashed, inspected "
                                      "(spec sections 1, 4)", "files": {}}
    problems: list[str] = []
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    try:
        for name, spec in expect["hf"].items():
            path = fetch_hf(spec, args.out / name, args.local_source)
            record["files"][name] = check_file(path, spec)
            paths[name] = path
            if not record["files"][name]["ok"]:
                problems.append(f"{name}: {record['files'][name]} is not the recorded "
                                f"{spec['bytes']} B / {spec['sha256']}")
        fire_path, refused = fetch_fire(expect["fire"], args.out / "fire", args.local_source)
        if fire_path is None:
            problems.append(f"FIRE: Google Drive refused the download ({refused}). Stop and "
                            f"report; no mirror (spec section 4).")
            record["files"]["fire"] = {"ok": False, "blocked": "google-drive",
                                       "reason": refused}
        else:
            digest = file_sha256(fire_path)
            recorded = expect["fire"].get("sha256")
            record["files"]["fire"] = {
                "path": str(fire_path), "bytes": fire_path.stat().st_size, "sha256": digest,
                "is_zip": zipfile.is_zipfile(fire_path),
                "recorded_sha256": recorded, "ok": recorded in (None, digest)}
            if recorded not in (None, digest):
                problems.append(f"FIRE: sha256 {digest} is not the recorded {recorded}")
            paths["fire"] = fire_path
        if record["files"].get("cospy_zip", {}).get("ok"):
            extracted, found = extract_cospy(paths["cospy_zip"], expect["cospy_members"],
                                             args.out / "cospy_zip")
            record["cospy_zip"] = extracted
            problems += found
            for name in ("semantic", "artifact", "fusion"):
                entry = extracted.get(f"{name}_weights.pth")
                if entry:
                    paths[f"cospy_{name}"] = Path(entry["path"])
        if not problems:
            from dbp.decoders.loaders import load_member
            from dbp.decoders.registry import load_bank_config

            bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
            # FIRE's VAE is the pinned sd15 in fp16 (spec 7.13); comparisons use fp32.
            fire_vae = load_member(bank.by_key("sd15"), "cpu", "fp16").model
            reference = load_member(bank.by_key("sd15"), "cpu", "fp32").model
            sd15_state = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
            del reference
            record["checks"], found = inspect_all(paths, expect, sd15_state, fire_vae, log)
            problems += found
    except BaselineError as error:
        problems.append(str(error))
    record.update({"seconds": round(time.perf_counter() - started, 1), "ok": not problems,
                   "problems": problems, "dry_run": dryrun(),
                   "commit": (THESIS_DBP / "COMMIT.txt").read_text().split()[0]
                   if (THESIS_DBP / "COMMIT.txt").is_file() else None})
    write_record(args.report, record)
    for problem in problems:
        log.error("%s", problem)
    fire = record["files"].get("fire", {})
    log.info("FIRE sha256 (record and commit before any scoring): %s", fire.get("sha256"))
    log.info("weights: %s", "OK" if not problems else "FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
