"""P5 Hub audit (metadata only): the source of every pin, size, hash and licence tag in
docs/P5_PLAN.md and configs/video_bank.yaml.

For each model repo: the commit, gating, licence tags/card fields, and the VAE files' sizes
and LFS sha256. For each dataset repo: the commit, gating, licence, file count and bytes.
No weights, no videos. Writes ``reports/p5_plan/p5_hub_audit.json`` (atomic).

Usage:
    python scripts/p5_hub_audit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]

MODELS = {   # repo: VAE subfolder
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "vae", "Wan-AI/Wan2.1-T2V-14B-Diffusers": "vae",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "vae", "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "vae",
    "Qwen/Qwen-Image": "vae", "hunyuanvideo-community/HunyuanVideo": "vae",
    "hunyuanvideo-community/HunyuanVideo-I2V": "vae", "tencent/HunyuanVideo": "",
    "zai-org/CogVideoX-2b": "vae", "zai-org/CogVideoX-5b": "vae",
    "zai-org/CogVideoX1.5-5B": "vae", "Lightricks/LTX-Video": "vae",
    "genmo/mochi-1-preview": "vae", "rhymes-ai/Allegro": "vae",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "vae",
    "nvidia/Cosmos-1.0-Diffusion-7B-Text2World": "vae", "Lightricks/LTX-2.5-Diffusers": "vae",
}
DATASETS = ["zengxianyu/open-sora-pexels-subset", "AIGVDBench/AIGVDBench", "Clarifiedfish/AEGIS",
            "Vchitect/VBench_sampled_video", "FastVideo/Mixkit-Src"]


def card(info: Any) -> dict[str, Any]:
    data = info.card_data.to_dict() if getattr(info, "card_data", None) else {}
    return {k: data.get(k) for k in ("license", "license_name", "license_link") if data.get(k)}


def main() -> int:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    print(f"HF_TOKEN present: {bool(os.environ.get('HF_TOKEN'))}")
    out: dict[str, Any] = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "models": {}, "datasets": {}}
    for repo, sub in MODELS.items():
        try:
            info = api.model_info(repo, files_metadata=True)
        except (GatedRepoError, HfHubHTTPError) as exc:
            out["models"][repo] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            continue
        files = {s.rfilename: {"size": s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None}
                 for s in info.siblings or [] if sub and s.rfilename.startswith(sub + "/")
                 and s.rfilename.endswith((".safetensors", ".json"))}
        out["models"][repo] = {"sha": info.sha, "gated": info.gated, "card": card(info),
                               "license_tags": [t for t in info.tags or []
                                                if t.startswith("license:")], "vae_files": files}
    for repo in DATASETS:
        try:
            info = api.dataset_info(repo, files_metadata=True)
        except (GatedRepoError, HfHubHTTPError) as exc:
            out["datasets"][repo] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
            continue
        siblings = info.siblings or []
        out["datasets"][repo] = {
            "sha": info.sha, "gated": info.gated, "card": card(info),
            "license_tags": [t for t in info.tags or [] if t.startswith("license:")],
            "n_files_listed": len(siblings), "bytes_listed": sum(s.size or 0 for s in siblings),
            "listing_note": "the API listing is capped at 100,000 files" if len(siblings) >= 100000
            else ""}
    target = THESIS_DBP / "reports" / "p5_plan" / "p5_hub_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    print(f"wrote {target}: {len(out['models'])} models, {len(out['datasets'])} datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
