"""P4: the isolated DDA equivalence check (user decision 2026-11-01; spec items 11 and 7.12).

Runs **only** in its own CPU-only Kaggle notebook with no Secrets and no private dataset
attached: the inputs are a code-only bundle (this script, ``dbp.baselines`` and the
bootstrap, built from git's bytes) and the public COCO 2017 dataset. Nothing from the
session is reused; its one output is the scores below.

It scores the same 2 COCO val2017 images with:

1. **DDA's own code** -- ``roy-ch/Dual-Data-Alignment`` @ ``08d6b414`` downloaded as a
   GitHub archive, its ``DINOv2ModelWithLoRA`` built as ``Inference/inference.py`` builds
   it (``dinov2_vitl14``, rank 8, alpha 1), which calls ``torch.hub.load`` for DINOv2.
   ``torch.hub`` is pointed at the DINOv2 commit the spec was written against
   (``7764ea0f``, ``main`` on 2026-11-01); nothing else in DDA's code is changed. The
   checkpoint is loaded the way ``inference.py`` does (``checkpoint['model']``, strict),
   and the images go through its ``test_transform``;
2. **our adapter** -- ``dbp.baselines.dda.load_dda`` and ``dda_input(..., 336)``.

**Pass:** the two logits agree within ``--tolerance`` (1e-3) on both images. The
tolerance is never loosened; a failure is fixed at its cause (user, 2026-11-01).
Writes ``--out`` (JSON) and exits 0 on a pass, 1 on a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.baselines.common import dryrun, file_sha256  # noqa: E402

DDA_REPO = "roy-ch/Dual-Data-Alignment"
DINOV2_REPO = "facebookresearch/dinov2"


def fetch_archive(commit: str, work: Path, local: Path | None) -> Path:
    """DDA's repository at ``commit`` as a GitHub archive, extracted; returns Inference/."""
    archive = work / f"dda_{commit[:8]}.zip"
    if local is not None:
        shutil.copyfile(local, archive)
    else:
        url = f"https://github.com/{DDA_REPO}/archive/{commit}.zip"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
    target = work / "dda_repo"
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if "/Inference/" in name and not name.endswith("/") and ".." not in name:
                dest = target / name.split("/", 1)[1]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(handle.read(name))
    return target / "Inference"


def upstream_logits(inference: Path, ckpt: Path, images: list[Any], dinov2_commit: str
                    ) -> list[float]:
    """DDA's own model and transform, on the images (PIL, RGB)."""
    import torch
    from torchvision import transforms

    original = torch.hub.load

    def pinned(repo: str, model: str, *args: Any, **kwargs: Any) -> Any:
        if repo == DINOV2_REPO:
            repo = f"{DINOV2_REPO}:{dinov2_commit}"
            kwargs.setdefault("trust_repo", True)
            kwargs.setdefault("skip_validation", True)
        return original(repo, model, *args, **kwargs)

    torch.hub.load = pinned
    sys.path.insert(0, str(inference))
    try:
        from models.dinov2_models_lora import DINOv2ModelWithLoRA

        model = DINOv2ModelWithLoRA(name="dinov2_vitl14", lora_rank=8, lora_alpha=1,
                                    lora_targets=None)
    finally:
        torch.hub.load = original
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # Inference/inference.py's test_transform, verbatim.
    test_transform = transforms.Compose([
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    with torch.no_grad():
        return [float(model(test_transform(image).unsqueeze(0)).flatten()[0])
                for image in images]


def our_logits(ckpt: Path, sha256: str, images: list[Any], arch: dict[str, Any] | None
               ) -> list[float]:
    """``dbp.baselines.dda`` on the same images (``arch``: the dry-run's tiny model only)."""
    import numpy as np
    import torch

    from dbp.baselines.dda import NATIVE_CROP, dda_input, load_dda

    model = load_dda(ckpt, sha256, "cpu", arch)
    out = []
    with torch.no_grad():
        for image in images:
            x, _ = dda_input(np.asarray(image, dtype=np.uint8), NATIVE_CROP)
            out.append(float(model(x.unsqueeze(0)).flatten()[0]))
    return out


def main() -> int:
    """Score both implementations; 0 when every logit agrees within the tolerance."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--coco-val", type=Path, required=True,
                        help="The val2017 directory of the public COCO 2017 dataset.")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-archive", type=Path, default=None,
                        help="DRY-RUN ONLY (DBP_DRYRUN=1): DDA's archive from this file.")
    parser.add_argument("--local-ckpt", type=Path, default=None,
                        help="DRY-RUN ONLY (DBP_DRYRUN=1): the checkpoint from this file.")
    args = parser.parse_args()
    if (args.local_archive or args.local_ckpt) and not dryrun():
        print("--local-archive/--local-ckpt are for the dry-run only (DBP_DRYRUN=1)")
        return 2
    import torch
    from PIL import Image

    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    args.work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spec = expect["dda_ckpt"]
    if args.local_ckpt is not None:
        ckpt = args.local_ckpt
    else:
        from huggingface_hub import hf_hub_download

        ckpt = Path(hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"],
                                    local_dir=str(args.work / "ckpt"), token=False))
    sha256 = file_sha256(ckpt)
    record: dict[str, Any] = {
        "what": "P4 isolated DDA equivalence check (DDA's own code vs dbp.baselines.dda)",
        "dda_commit": expect["dda_commit"], "dinov2_commit": expect["dinov2_commit"],
        "ckpt_sha256": sha256, "ckpt_ok": sha256 == spec["sha256"],
        "tolerance": expect["tolerance"], "python": platform.python_version(),
        "torch": torch.__version__, "dry_run": dryrun(), "images": []}
    images = []
    for name in expect["images"]:
        path = args.coco_val / name
        with Image.open(path) as handle:
            images.append(handle.convert("RGB"))
        record["images"].append({"file": name, "sha256": file_sha256(path),
                                 "size": list(images[-1].size)})
    if not record["ckpt_ok"]:
        record.update({"pass": False, "problem": "DDA_ckpt.pth is not the recorded file"})
    else:
        inference = fetch_archive(expect["dda_commit"], args.work, args.local_archive)
        record["archive_files"] = sorted(str(p.relative_to(inference)) for p in
                                         inference.rglob("*.py"))
        theirs = upstream_logits(inference, ckpt, images, expect["dinov2_commit"])
        ours = our_logits(ckpt, spec["sha256"], images,
                          expect.get("dryrun_dda_arch") if dryrun() else None)
        for entry, a, b in zip(record["images"], theirs, ours, strict=True):
            entry.update({"upstream_logit": a, "ours_logit": b, "abs_diff": abs(a - b)})
        worst = max(e["abs_diff"] for e in record["images"])
        record.update({"max_abs_diff": worst, "pass": worst <= expect["tolerance"]})
    record["seconds"] = round(time.perf_counter() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for entry in record["images"]:
        print(f"{entry['file']}: upstream {entry.get('upstream_logit')} ours "
              f"{entry.get('ours_logit')} |diff| {entry.get('abs_diff')}")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"DDA EQUIVALENCE: {'PASS' if record['pass'] else 'FAIL'} (max |diff| "
          f"{record.get('max_abs_diff')}, tolerance {expect['tolerance']}); "
          f"{args.out.name} sha256 {digest}")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
