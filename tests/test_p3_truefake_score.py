"""TrueFake scoring stage: disjointness, the scoring list, and the descriptive analysis.

Every score here is synthetic; no TrueFake image or score is read.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECT = json.loads((ROOT / "env" / "p3_truefake_score_expect.json").read_text(encoding="utf-8"))
RUNBOOK = ROOT / "env" / "kaggle_run_p3_truefake_score.py"


def _load(name: str, script: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREP = _load("_tf_prepare", "p3_truefake_prepare.py")
ANA = _load("_tf_analyze", "p3_truefake_analyze.py")


class _Log:
    def info(self, *args):
        pass


def test_expect_is_the_committed_sample_and_the_four_models():
    for key in ("sample", "manifest"):
        digest = hashlib.sha256((ROOT / EXPECT[key]).read_bytes()).hexdigest()
        assert digest == EXPECT[f"{key}_sha256"]
    final = json.loads((ROOT / EXPECT["final_expect"]).read_text(encoding="utf-8"))
    ids = {m["id"] for m in final["models"]}
    assert EXPECT["models"] == list(ANA.MODELS) and set(EXPECT["models"]) <= ids
    assert EXPECT["oi_manifest"] == final["section5"]["oi_manifest"]
    assert EXPECT["members"] == 16000 and EXPECT["coco_splits"] == ["train2017"]


def test_hamming_min_matches_the_pairwise_distance():
    from dbp.data.phash import hamming_distance

    rng = np.random.default_rng(1)
    values = rng.integers(0, 2**63, 300, dtype=np.uint64)
    refs = rng.integers(0, 2**63, 700, dtype=np.uint64)
    refs[5] = values[17] ^ np.uint64(0b1011)          # 3 bits from value 17
    got = PREP.hamming_min(values, refs, chunk=64)
    want = [min(hamming_distance(int(v), int(r)) for r in refs) for v in values]
    assert got.tolist() == want and got[17] == 3
    assert PREP.hamming_min(values[:3], np.array([], dtype=np.uint64)).tolist() == [64] * 3


def test_relative_drops_the_archive_prefix():
    assert PREP.relative("TrueFake/Twitter/Real/FFHQ/00001.jpg") == "Twitter/Real/FFHQ/00001.jpg"


def _photo(path: Path, seed: int, size=(400, 360)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)   # smooth: a stable pHash
    Image.fromarray(small).resize(size, Image.BILINEAR).save(path)


def test_disjointness_drops_an_id_in_every_version_and_counts_per_pool(tmp_path):
    from dbp.data.phash import phash_file

    local = tmp_path / "local"
    reals = []
    for pool, ident, seed in (("FFHQ", "a", 1), ("FFHQ", "b", 2), ("FORLAB", "c", 3),
                              ("FORLAB", "d", 4)):
        for share in ("PreSocial", "Twitter"):
            member = f"TrueFake/{share}/Real/{pool}/{ident}.png"
            _photo(local / PREP.relative(member), seed + (10 if share == "Twitter" else 0))
            sha = hashlib.sha256((local / PREP.relative(member)).read_bytes()).hexdigest()
            reals.append({"member": member, "id": f"Real/{pool}/{ident}",
                          "stratum": f"Real/{pool}", "sha256": sha})
    # b's Twitter version is a training real by sha256; c's pre-share version by pHash.
    shas = {reals[3]["sha256"]}
    refs = np.array([phash_file(local / PREP.relative(reals[4]["member"]))], dtype=np.uint64)
    matched, record = PREP.disjointness(reals, local, shas, refs, workers=2)
    assert sorted(matched) == ["Real/FFHQ/b", "Real/FORLAB/c"]
    assert matched["Real/FFHQ/b"]["members"][0]["sha256_match"] is True
    assert matched["Real/FORLAB/c"]["members"][0]["phash_distance"] == 0
    assert record["pools"] == {"Real/FFHQ": {"ids": 2, "dropped_ids": 1},
                               "Real/FORLAB": {"ids": 2, "dropped_ids": 1}}


def test_references_hash_coco_originals_after_their_sha256(tmp_path):
    from dbp.data.phash import phash_file, phash_hex

    coco = tmp_path / "coco"
    _photo(coco / "train2017" / "000000000009.jpg", 7)
    _photo(coco / "train2017" / "000000000025.jpg", 8)
    good = hashlib.sha256((coco / "train2017/000000000009.jpg").read_bytes()).hexdigest()
    columns = ["sha256", "source_sha256", "source_relative_path", "phash", "excluded_reason"]
    t_rows = [{"sha256": "c1", "source_sha256": good, "source_relative_path":
               "000000000009.jpg", "phash": "00000000000000ff", "excluded_reason": ""},
              {"sha256": "c2", "source_sha256": "f" * 64, "source_relative_path":
               "000000000025.jpg", "phash": "00000000000000fe", "excluded_reason": ""},
              {"sha256": "c3", "source_sha256": "e" * 64, "source_relative_path":
               "000000000030.jpg", "phash": "00000000000000fd", "excluded_reason": "D-039"}]
    oi_rows = [{"sha256": "o1", "source_sha256": "o1s", "phash": "0000000000000001",
                "source_phash": "0000000000000002", "split": "train"},
               {"sha256": "o2", "source_sha256": "o2s", "phash": "0000000000000003",
                "source_phash": "0000000000000004", "split": "held_out"}]
    for name, cols, rows in (("t.csv", columns, t_rows),
                             ("oi.csv", list(oi_rows[0]), oi_rows)):
        with (tmp_path / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
    expect = {"t_real_manifest": "t.csv", "oi_manifest": "oi.csv",
              "coco_splits": ["train2017"]}
    shas, refs, record = PREP.references(tmp_path, expect, coco, 2, _Log())
    assert shas == {"c1", "c2", good, "f" * 64, "o1", "o1s"}       # excluded and held out: no
    coco_record = record["coco_originals"]
    assert coco_record["hashed"] == 1 and coco_record["n_sha256_mismatch"] == 1
    assert coco_record["n_missing"] == 0 and record["t_real"] == 2
    original = phash_file(coco / "train2017/000000000009.jpg")
    assert original in refs.tolist() and len(refs) == 2 + 2 + 1
    assert phash_hex(original) != "00000000000000ff"


# ------------------------------------------------------------------- analysis ---

N_POOL, N_GEN = 6, 4


def _synthetic(tmp: Path, drop: str = "", small: str = "", same_shares: bool = False,
               broken: str = ""):
    """A list over every pool and generator, 4 shares; synthetic scores per model."""
    rows = []
    for group in ANA.POOLS + ANA.GENERATORS:
        label = "real" if group in ANA.POOLS else "fake"
        for k in range(N_POOL if label == "real" else N_GEN):
            ident = f"{'Real' if label == 'real' else 'Fake'}/{group}/{k:03d}"
            for share in ANA.SHARES:
                twin = share == "Twitter" and group == "StableDiffusion1.5"
                sha = f"{ident}|{'PreSocial' if twin else share}"
                rows.append({"image_id": f"TrueFake/{share}/{ident}.png",
                             "relative_path": f"{share}/{ident}.png", "id": ident,
                             "share": share, "stratum": f"{ident.rsplit('/', 1)[0]}",
                             "label": label, "group": group, "sha256": sha,
                             "width": "400", "height": "400",
                             "dropped": "1" if ident == drop else "0",
                             "too_small": "1" if (ident == small and share == "Facebook")
                             else "0",
                             "unreadable": "1" if (ident == broken and share == "Twitter")
                             else "0"})
    list_path = tmp / "list.csv"
    with list_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREP.LIST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    order = ANA.scored_order(rows)
    rng = np.random.default_rng(5)
    list_sha = hashlib.sha256(list_path.read_bytes()).hexdigest()
    threshold_rows = [{"image_id": f"t{k}"} for k in range(40)]
    with (tmp / "thr.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write("image_id\n" + "".join(f"{r['image_id']}\n" for r in threshold_rows))
    for model in ANA.MODELS:
        base = {r["id"]: rng.normal(1.0 if r["label"] == "fake" else 0.0) for r in order}
        scores = [base[r["id"]] + (0.0 if same_shares else rng.normal(0, 0.5))
                  for r in order]
        path = tmp / "scores" / model / "truefake.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("image_id,score\n" + "".join(
                f"{r['image_id']},{s:.6f}\n" for r, s in zip(order, scores, strict=True)))
        summary = {"model": model, "limit": None, "list_sha256": list_sha,
                   "lora_sha256": "l", "head_sha256": "h", "lists": {"truefake": {
                       "rows": len(order),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
        (tmp / "summaries").mkdir(exist_ok=True)
        (tmp / "summaries" / f"{model}.json").write_text(json.dumps(summary), encoding="utf-8")
        thr = tmp / "thr_scores" / model / "clean" / "threshold_reals.csv"
        thr.parent.mkdir(parents=True, exist_ok=True)
        thr.write_text("image_id,score\n" + "".join(
            f"t{k},{v:.6f}\n" for k, v in enumerate(np.linspace(-1, 1, 40))), encoding="utf-8")
    return list_path, rows


def _run(tmp: Path, resamples: int = 30):
    return ANA.run(tmp / "list.csv", tmp / "scores", tmp / "summaries", tmp / "thr.csv",
                   tmp / "thr_scores", resamples=resamples)


def test_excluded_ids_leave_every_version_and_are_counted(tmp_path):
    _synthetic(tmp_path, drop="Real/FORLAB/002", small="Fake/StyleGAN2/001",
               broken="Real/FFHQ/004")
    result = _run(tmp_path)
    counts = result["counts"]
    assert counts["kept_ids"]["FORLAB"] == N_POOL - 1
    assert counts["kept_ids"]["StyleGAN2"] == N_GEN - 1
    assert counts["kept_ids"]["FFHQ"] == N_POOL - 1
    assert counts["excluded"] == {"dropped": {"FORLAB": 1}, "too_small": {"StyleGAN2": 1},
                                  "unreadable": {"FFHQ": 1}}
    assert counts["twitter_identical"]["StableDiffusion1.5"] == N_GEN
    assert counts["twitter_identical"]["FLUX.1"] == 0


def test_identical_versions_give_a_zero_delta_with_a_zero_interval(tmp_path):
    _synthetic(tmp_path, same_shares=True)
    result = _run(tmp_path)
    for model in ANA.MODELS:
        for share in ANA.PLATFORMS:
            for pool in ANA.POOLS:
                for item in ANA.GENERATORS + ("diffusion", "stylegan"):
                    value = result["results"][model]["delta"][share][pool][item]
                    assert value == {"point": 0.0, "lower": 0.0, "upper": 0.0}


def test_points_macros_and_fpr_are_what_they_say(tmp_path):
    _list, rows = _synthetic(tmp_path)
    result = _run(tmp_path)
    order = ANA.scored_order(rows)
    model = "s2d_p35_seed0"
    with (tmp_path / "scores" / model / "truefake.csv").open(encoding="utf-8") as handle:
        score = {r["image_id"]: float(r["score"]) for r in csv.DictReader(handle)}
    pick = lambda g, s: np.array([score[r["image_id"]] for r in order  # noqa: E731
                                  if r["group"] == g and r["share"] == s])
    from dbp.eval.metrics import roc_auc

    reals, fakes = pick("FFHQ", "Telegram"), pick("StyleGAN3", "Telegram")
    want = roc_auc(np.r_[np.zeros(len(reals)), np.ones(len(fakes))], np.r_[reals, fakes])
    res = result["results"][model]["auc"]
    assert res["Telegram"]["FFHQ"]["StyleGAN3"]["point"] == pytest.approx(want, abs=1e-6)
    macro = np.mean([res["Facebook"]["FORLAB"][g]["point"] for g in ANA.DIFFUSION])
    assert res["Facebook"]["FORLAB"]["diffusion"]["point"] == pytest.approx(macro, abs=1e-6)
    delta = result["results"][model]["delta"]["Twitter"]["FFHQ"]["stylegan"]
    assert delta["point"] == pytest.approx(res["Twitter"]["FFHQ"]["stylegan"]["point"] -
                                           res["PreSocial"]["FFHQ"]["stylegan"]["point"],
                                           abs=1e-5)
    assert delta["lower"] <= delta["point"] <= delta["upper"]
    fpr = result["forlab_fpr_at_k0_union_threshold"]["by_model"][model]
    assert fpr["Facebook"] == pytest.approx(np.mean(pick("FORLAB", "Facebook") >=
                                                    fpr["threshold"]))
    assert "FFHQ" not in json.dumps(result["forlab_fpr_at_k0_union_threshold"])


def test_the_bootstrap_is_seeded_and_paired(tmp_path):
    _synthetic(tmp_path)
    first, second = _run(tmp_path), _run(tmp_path)
    assert first["results"] == second["results"]
    assert first["bootstrap"]["seed"] == 2026


def test_a_summary_that_does_not_describe_the_scores_is_refused(tmp_path):
    _synthetic(tmp_path)
    path = tmp_path / "summaries" / "full_bank_seed0.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**summary, "limit": 2}), encoding="utf-8")
    with pytest.raises(ANA.AnalysisError, match="full_bank_seed0"):
        _run(tmp_path)
    path.write_text(json.dumps({**summary, "list_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ANA.AnalysisError, match="full_bank_seed0"):
        _run(tmp_path)


def test_a_truncated_file_is_recorded_not_repaired(tmp_path):
    """Kaggle 2026-10-31: a TrueFake JPEG, byte-identical to the archive, is truncated."""
    good = tmp_path / "good.jpg"
    _photo(good, 3)
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(good.read_bytes()[:-400])
    assert PREP.decode_check(good) == (400, 360, "")
    width, height, error = PREP.decode_check(cut)
    assert (width, height) == (400, 360) and "truncated" in error
    from PIL import ImageFile

    assert ImageFile.LOAD_TRUNCATED_IMAGES is False       # never switched on to "repair" it


def test_the_scorer_and_the_analysis_score_the_same_rows():
    sys.path.insert(0, str(ROOT / "scripts"))
    score = _load("_tf_score", "p3_truefake_score.py")
    rows = [{"dropped": d, "too_small": s, "unreadable": u, "image_id": f"{d}{s}{u}"}
            for d in "01" for s in "01" for u in "01"]
    assert score.scored_rows(rows) == ANA.scored_order(rows) == [rows[0]]


def test_runbook_finds_the_sample_in_a_notebook_output_or_a_dataset(tmp_path):
    import re

    text = RUNBOOK.read_text(encoding="utf-8")
    block = text[text.index('"sample": one('):text.index('"coco": one(')]
    patterns = re.findall(r'"(\*/[^"]*PreSocial/Real/FFHQ)"', block)
    assert patterns, "no sample pattern in Cell 1"
    layouts = {"notebooks/muhammadadeelshah/truefake-fetch/truefake_sample":
               "notebook output (user, 2026-10-29)",
               "datasets/muhammadadeelshah/dbp-truefake-sample": "dataset",
               "datasets/muhammadadeelshah/dbp-truefake-sample/truefake_sample":
               "dataset holding the folder"}
    for n, (root, what) in enumerate(layouts.items()):
        inputs = tmp_path / f"input{n}"          # one /kaggle/input per layout
        (inputs / root / "PreSocial/Real/FFHQ").mkdir(parents=True)
        (inputs / "notebooks/muhammadadeelshah/truefake-fetch/Implementation/thesis_dbp/"
                  "reports/truefake").mkdir(parents=True, exist_ok=True)
        found = {p.parents[2] for g in patterns for p in inputs.glob(g)}
        assert found == {inputs / root}, what


def test_model_sources_use_the_final_runbooks_proven_markers():
    """Kaggle 2026-10-31: the tune output also holds s2b_seed0, so finding the train output
    by s2b_seed0 matched two outputs. Each marker must be one only its source holds; the
    final runbook's markers were proven on Kaggle with the same inputs."""
    import re

    final = (ROOT / "env" / "kaggle_run_p3_final.py").read_text(encoding="utf-8")
    text = RUNBOOK.read_text(encoding="utf-8")
    pattern = r'"(\w+)": runs_root\("([\w.]+)"\)'
    proven = dict(re.findall(pattern, final))
    used = dict(re.findall(pattern, text))
    assert used == {k: proven[k] for k in ("train", "tune", "s2d")}
    assert used["train"] != "s2b_seed0"


def test_runbook_is_thin_and_finds_every_input_by_content():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "import dbp" not in text and "\n!" not in text and "\n%" not in text
    for needed in ("PreSocial/Real/FFHQ", "coco2017/train2017", 'runs_root("s2b_seed1")',
                   'runs_root("s2b_deg_p35_seed0")', 'runs_root("s2d_p35_seed1")',
                   "full_bank_seed0", "p3_truefake_prepare.py", "p3_truefake_score.py",
                   '"--stage", "truefake"'):
        assert needed in text, needed
