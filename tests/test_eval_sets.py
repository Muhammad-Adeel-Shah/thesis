"""End-to-end tests for the evaluation-set builder (P1-B7).

The builder is the piece that decides what every reported number is computed over, so
the properties tested here are the ones whose violation would quietly change a result:

- an image is described by what it *is*, not by where it sits;
- a file that two labels' globs both match is an error, not a coin flip;
- the JPEG-matched view re-encodes only the never-compressed fakes, at native
  resolution, drawing from the set's own real distribution;
- the originals are still byte-identical afterwards;
- rebuilding the view twice gives the same bytes;
- leakage between a training corpus and an evaluation set is found.

The fixture reproduces the shape of E3's ``Raw`` view as measured from the Hub listing
on 2026-09-20: every real is a JPEG, most fakes are PNG that have never been
compressed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dbp.data.jpeg import encode_jpeg_bytes, jpeg_metadata, parameter_separability
from dbp.data.manifest import file_sha256, read_manifest, write_manifest
from dbp.data.phash import phash_from_array, phash_hex

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_builder():
    """Import ``p1_build_eval.py`` by path; ``scripts/`` is not a package."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("p1_build_eval", SCRIPTS / "p1_build_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _photo(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    base = (
        112.0
        + 48.0 * np.sin(xs / (9.0 + seed % 5))
        + 37.0 * np.cos(ys / (14.0 + seed % 7))
        + rng.normal(0, 4.0, size=(height, width))
    )
    stack = np.stack([base, base * 0.87 + 18.0, base * 0.71 + 36.0], axis=-1)
    return np.clip(stack, 0, 255).astype(np.uint8)


@pytest.fixture
def e3_like(tmp_path: Path) -> Path:
    """An E3-shaped set: reals all JPEG, fakes mostly never-compressed PNG."""
    root = tmp_path / "set"
    qualities = [72, 78, 84, 90, 93, 96]
    for index in range(12):
        directory = root / "images" / "Raw" / f"{'Texture' if index % 2 else 'Structure'} Images"
        real_dir = directory / "real"
        real_dir.mkdir(parents=True, exist_ok=True)
        payload = encode_jpeg_bytes(
            _photo(112 + index, 137 + index, seed=index), qualities[index % len(qualities)], 2
        )
        (real_dir / f"{index}.jpg").write_bytes(payload)

    for index in range(12):
        directory = root / "images" / "Raw" / f"{'Texture' if index % 2 else 'Structure'} Images"
        fake_dir = directory / "GPT-Image2-fake"
        fake_dir.mkdir(parents=True, exist_ok=True)
        image = _photo(120 + index, 151 + index, seed=100 + index)
        if index < 9:  # most fakes have never been compressed, as in E3
            Image.fromarray(image, mode="RGB").save(fake_dir / f"{index}.png")
        else:
            (fake_dir / f"{index}.jpg").write_bytes(encode_jpeg_bytes(image, 88, 2))
    return root


def _run(root: Path, tmp_path: Path, set_id: str = "e3_like", extra: list[str] | None = None):
    argv = [
        "p1_build_eval.py",
        "--set-id",
        set_id,
        "--root",
        str(root),
        "--real-glob",
        "images/Raw/*/real/*",
        "--fake-glob",
        "images/Raw/*/*-fake/*",
        "--field",
        "subset=2",
        "--field",
        "source_label=3",
        "--out-root",
        str(tmp_path / "views"),
        *(extra or []),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        return builder.main()
    finally:
        sys.argv = old


def test_labels_and_fields_come_from_the_path(e3_like, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    assert _run(e3_like, tmp_path) == 0

    rows = read_manifest(tmp_path / "reports" / "p1_eval_e3_like_manifest.csv")
    assert len(rows) == 24
    labels = sorted({row["label"] for row in rows})
    assert labels == ["fake", "real"]
    assert {row["source_label"] for row in rows} == {"real", "GPT-Image2-fake"}
    assert {row["subset"] for row in rows} == {"Structure Images", "Texture Images"}


def test_sizes_are_the_true_pixel_sizes_and_nothing_is_resized(e3_like, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    _run(e3_like, tmp_path)

    rows = read_manifest(tmp_path / "reports" / "p1_eval_e3_like_manifest.csv")
    for row in rows:
        with Image.open(row["source_path"]) as handle:
            assert handle.size == (int(row["width"]), int(row["height"]))
    assert len({(row["width"], row["height"]) for row in rows}) > 1, "fixture is not varied"


def test_the_compression_confound_is_detected(e3_like, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    _run(e3_like, tmp_path)

    summary = json.loads(
        (tmp_path / "reports" / "p1_eval_e3_like_summary.json").read_text(encoding="utf-8")
    )
    assert summary["compression_confounded"] is True
    assert "CONFOUNDED" in summary["compression_verdict"]
    table = summary["compression_table"]
    assert table["real"]["non_jpeg"] == 0
    assert table["fake"]["non_jpeg"] == 9
    assert "| label | n | JPEG |" in summary["compression_table_markdown"]


def test_matched_view_reencodes_only_never_compressed_fakes(e3_like, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    before = {
        path: file_sha256(path) for path in sorted(e3_like.rglob("*")) if path.is_file()
    }

    assert _run(e3_like, tmp_path, extra=["--jpeg-match"]) == 0

    view = read_manifest(tmp_path / "reports" / "p1_eval_e3_like_jpeg_matched_manifest.csv")
    assert len(view) == 24
    reencoded = [row for row in view if row["reencoded"] == "True"]
    assert len(reencoded) == 9, "only the never-compressed fakes should be re-encoded"
    assert all(row["label"] == "fake" for row in reencoded)
    assert all(row["reencoded"] == "False" for row in view if row["label"] == "real")

    # Every original is untouched.
    after = {path: file_sha256(path) for path in sorted(e3_like.rglob("*")) if path.is_file()}
    assert after == before, "the builder modified a source file"

    summary = json.loads(
        (tmp_path / "reports" / "p1_eval_e3_like_summary.json").read_text(encoding="utf-8")
    )
    assert summary["jpeg_matched_view"]["compression_confounded"] is False


def test_matched_view_draws_from_the_real_distribution_without_resizing(
    e3_like, tmp_path, monkeypatch
):
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    _run(e3_like, tmp_path, extra=["--jpeg-match"])

    rows = read_manifest(tmp_path / "reports" / "p1_eval_e3_like_manifest.csv")
    real_qualities = {int(r["jpeg_quality"]) for r in rows if r["label"] == "real"}
    real_subsampling = {r["subsampling"] for r in rows if r["label"] == "real"}

    view = read_manifest(tmp_path / "reports" / "p1_eval_e3_like_jpeg_matched_manifest.csv")
    for row in view:
        if row["reencoded"] != "True":
            continue
        assert int(row["matched_quality"]) in real_qualities
        assert row["matched_subsampling"] in real_subsampling
        assert jpeg_metadata(row["view_path"])["is_jpeg"] is True
        with Image.open(row["source_path"]) as source, Image.open(row["view_path"]) as encoded:
            assert source.size == encoded.size, "the matched view must not resize (D-001)"


def test_matched_view_is_byte_identical_when_rebuilt(e3_like, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    _run(e3_like, tmp_path, extra=["--jpeg-match"])
    first = {
        row["image_id"]: file_sha256(row["view_path"])
        for row in read_manifest(
            tmp_path / "reports" / "p1_eval_e3_like_jpeg_matched_manifest.csv"
        )
        if row["reencoded"] == "True"
    }

    _run(e3_like, tmp_path, extra=["--jpeg-match"])
    second = {
        row["image_id"]: file_sha256(row["view_path"])
        for row in read_manifest(
            tmp_path / "reports" / "p1_eval_e3_like_jpeg_matched_manifest.csv"
        )
        if row["reencoded"] == "True"
    }
    assert first == second, "the seeded matched view is not reproducible"


def test_ambiguous_globs_are_refused(e3_like, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    argv = [
        "p1_build_eval.py",
        "--set-id",
        "ambiguous",
        "--root",
        str(e3_like),
        "--real-glob",
        "images/Raw/*/*/*",
        "--fake-glob",
        "images/Raw/*/*-fake/*",
        "--out-root",
        str(tmp_path / "views"),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        assert builder.main() == 2, "overlapping globs must not be silently resolved"
    finally:
        sys.argv = old


def test_leakage_against_a_training_manifest_is_found(e3_like, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    # A training manifest containing the perceptual hash of one of the evaluation reals.
    leaked = sorted((e3_like / "images" / "Raw").rglob("real/*.jpg"))[0]
    with Image.open(leaked) as handle:
        leaked_array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    training = [
        {"image_id": "train_leaked", "phash": phash_hex(phash_from_array(leaked_array))},
        {"image_id": "train_clean", "phash": phash_hex(phash_from_array(_photo(200, 200, 999)))},
    ]
    training_manifest = tmp_path / "reports" / "p1_reals_manifest_t_real.csv"
    write_manifest(training, training_manifest, ("image_id", "phash"))

    _run(
        e3_like,
        tmp_path,
        extra=["--leak-check-against", str(training_manifest)],
    )

    findings = json.loads((tmp_path / "reports" / "p1_leakage.json").read_text(encoding="utf-8"))
    entry = findings["e3_like"]["against"][training_manifest.name]
    assert entry["status"] == "LEAKAGE"
    assert [m["training_id"] for m in entry["matches"]] == ["train_leaked"]


def test_a_disjoint_training_manifest_reports_clean(e3_like, tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    training = [
        {"image_id": f"train_{i}", "phash": phash_hex(phash_from_array(_photo(180, 180, 500 + i)))}
        for i in range(5)
    ]
    training_manifest = tmp_path / "reports" / "p1_reals_manifest_t_ctrl.csv"
    write_manifest(training, training_manifest, ("image_id", "phash"))

    _run(e3_like, tmp_path, extra=["--leak-check-against", str(training_manifest)])

    findings = json.loads((tmp_path / "reports" / "p1_leakage.json").read_text(encoding="utf-8"))
    assert findings["e3_like"]["against"][training_manifest.name]["status"] == "CLEAN"


def test_a_set_with_no_real_jpegs_cannot_be_matched(tmp_path, monkeypatch):
    """Inventing a distribution would be worse than reporting the confound."""
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    root = tmp_path / "allpng"
    for label in ("real", "GPT-Image2-fake"):
        directory = root / "images" / "Raw" / "Texture Images" / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            Image.fromarray(_photo(96, 96, seed=index + (0 if label == "real" else 50))).save(
                directory / f"{index}.png"
            )

    with pytest.raises(ValueError, match="no real JPEG"):
        rows = [
            builder.describe(
                path,
                root,
                "allpng",
                "real" if "/real/" in path.as_posix() else "fake",
                {},
            )
            for path in sorted(root.rglob("*.png"))
        ]
        builder.build_matched_view(rows, "allpng", tmp_path / "views", _NullLog())


class _NullLog:
    """A logger stand-in for the direct-call test above."""

    def info(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


# ---------------------------------------------------------------------------
# Seeded, stratified subsampling (D-036)
# ---------------------------------------------------------------------------


def _rows(n_per_generator: int = 40, generators=("flux", "got", "nova")):
    return [
        {"image_id": f"{gen}_{i:05d}", "label": "fake", "generator": gen}
        for gen in generators
        for i in range(n_per_generator)
    ]


class _Quiet:
    def info(self, *a, **k):
        return None

    def warning(self, *a, **k):
        return None

    def error(self, *a, **k):
        return None


def test_subsample_keeps_the_requested_number_from_every_stratum():
    kept, prov = builder.stratified_subsample(_rows(), 10, "generator", "e8_evalgen", _Quiet())
    assert len(kept) == 30
    per = {}
    for row in kept:
        per[row["generator"]] = per.get(row["generator"], 0) + 1
    assert per == {"flux": 10, "got": 10, "nova": 10}
    assert prov["applied"] is True
    assert prov["n_before"] == 120 and prov["n_after"] == 30


def test_subsample_is_deterministic_and_independent_of_input_order():
    rows = _rows()
    first, _ = builder.stratified_subsample(rows, 10, "generator", "e8_evalgen", _Quiet())
    shuffled = list(reversed(rows))
    second, _ = builder.stratified_subsample(shuffled, 10, "generator", "e8_evalgen", _Quiet())
    assert [r["image_id"] for r in first] == [r["image_id"] for r in second]


def test_subsample_is_not_just_the_first_n():
    """In a set laid out by generator, 'the first N files' returns one generator."""
    kept, _ = builder.stratified_subsample(_rows(), 10, "generator", "e8_evalgen", _Quiet())
    flux = sorted(r["image_id"] for r in kept if r["generator"] == "flux")
    assert flux != [f"flux_{i:05d}" for i in range(10)]


def test_subsample_records_every_kept_id():
    kept, prov = builder.stratified_subsample(_rows(), 5, "generator", "e8_evalgen", _Quiet())
    assert prov["kept_image_ids"] == [r["image_id"] for r in kept]
    assert prov["stratify_by"] == "generator"
    assert "seed_rule" in prov and "sampler" in prov


def test_a_short_stratum_is_kept_whole_and_named():
    rows = _rows(40, ("flux", "got")) + [
        {"image_id": f"tiny_{i:05d}", "label": "fake", "generator": "tiny"} for i in range(3)
    ]
    kept, prov = builder.stratified_subsample(rows, 10, "generator", "e8_evalgen", _Quiet())
    assert prov["strata"]["tiny"] == {"available": 3, "kept": 3}
    assert prov["strata_smaller_than_requested"] == ["tiny"]
    assert len(kept) == 23


def test_a_different_salt_selects_a_different_subset():
    a, _ = builder.stratified_subsample(_rows(), 10, "generator", "e8_evalgen", _Quiet())
    b, _ = builder.stratified_subsample(_rows(), 10, "generator", "e7_dda_coco", _Quiet())
    assert [r["image_id"] for r in a] != [r["image_id"] for r in b]


def test_subsample_without_stratification_uses_one_stratum():
    kept, prov = builder.stratified_subsample(_rows(), 7, None, "e8_evalgen", _Quiet())
    assert len(kept) == 7
    assert list(prov["strata"]) == ["_all"]


def test_a_non_positive_subsample_is_refused():
    with pytest.raises(ValueError, match="must be >= 1"):
        builder.stratified_subsample(_rows(), 0, "generator", "e8_evalgen", _Quiet())


def test_a_greyscale_real_cannot_break_the_matched_view(tmp_path, monkeypatch):
    """I-039: two greyscale reals among 1,251 aborted the whole E4 build.

    A greyscale JPEG has no chroma channels, so Pillow reports its subsampling as -1 and
    there is no setting to copy onto a colour image. The draw raised and took the run
    with it, after the manifest had been written but before the summary, the matched
    view and the leakage check. The reference must skip such a row -- the image itself
    stays in the set, since only its use as a chroma reference is affected.
    """
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    root = tmp_path / "mixed"
    real_dir = root / "images" / "Raw" / "Texture Images" / "real"
    fake_dir = root / "images" / "Raw" / "Texture Images" / "GPT-Image2-fake"
    for directory in (real_dir, fake_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Colour reals, plus two greyscale ones, which is the shape E4 had.
    for index in range(6):
        Image.fromarray(_photo(96, 96, seed=index)).save(
            real_dir / f"colour_{index}.jpg", quality=90
        )
    for index in range(2):
        Image.fromarray(_photo(96, 96, seed=40 + index)).convert("L").save(
            real_dir / f"grey_{index}.jpg", quality=90
        )
    # Never-compressed fakes, so the view has something to re-encode.
    for index in range(6):
        Image.fromarray(_photo(96, 96, seed=50 + index)).save(fake_dir / f"{index}.png")

    rows = [
        builder.describe(path, root, "mixed", label, {})
        for label, directory in (("real", real_dir), ("fake", fake_dir))
        for path in sorted(directory.iterdir())
    ]
    greyscale = [
        r for r in rows if r["label"] == "real" and str(r["subsampling"]).startswith("unknown")
    ]
    assert len(greyscale) == 2, "the fixture must contain the failing case"

    view_rows, view_summary = builder.build_matched_view(
        rows, "mixed", tmp_path / "view", _NullLog()
    )

    assert view_summary["reference_excluded_unmappable_subsampling"] == 2
    assert view_summary["reencoded"] == 6, "every never-compressed fake is re-encoded"
    assert len(view_rows) == len(rows), "no image is dropped from the set"
    assert all(r["matched_subsampling"] in ("4:4:4", "4:2:2", "4:2:0", "") or r["label"] == "real"
               for r in view_rows if r["reencoded"])


def test_the_matched_view_reports_its_own_residual(tmp_path, monkeypatch):
    """D-042 on the view, not only on the raw set.

    "Not confounded" says the never-compressed images are gone. It does not say the
    encoding parameters stopped predicting the label -- E3's matched view is clean
    pooled and still lifts +6.5 points on one subset, because fakes that were already
    JPEG keep their own quality distribution. The view has to measure itself.
    """
    monkeypatch.setattr(builder, "THESIS_DBP", tmp_path)
    root = tmp_path / "resid"
    real_dir = root / "images" / "Raw" / "Texture Images" / "real"
    fake_dir = root / "images" / "Raw" / "Texture Images" / "GPT-Image2-fake"
    for directory in (real_dir, fake_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Reals at one quality; fakes that are ALREADY JPEG at another, so they are left
    # in place by the view and keep the setting that gives them away.
    for index in range(8):
        Image.fromarray(_photo(96, 96, seed=index)).save(
            real_dir / f"{index}.jpg", quality=75, subsampling=2
        )
    for index in range(8):
        Image.fromarray(_photo(96, 96, seed=60 + index)).save(
            fake_dir / f"{index}.jpg", quality=100, subsampling=0
        )

    rows = [
        builder.describe(path, root, "resid", label, {})
        for label, directory in (("real", real_dir), ("fake", fake_dir))
        for path in sorted(directory.iterdir())
    ]
    view_rows, _ = builder.build_matched_view(rows, "resid", tmp_path / "view", _NullLog())

    assert all(not row["reencoded"] for row in view_rows), "nothing to re-encode here"

    measured = [
        {**row, "subsampling": row["matched_subsampling"], "jpeg_quality": row["matched_quality"]}
        for row in view_rows
    ]
    result = parameter_separability(measured, "subsampling")
    assert result["separable"] is True, "the residual must be visible, not assumed away"



# --------------------------------------------------------------------------- #
# D-043: the pipeline-matched view. Chameleon's reals are all 500 px wide
# because they were thumbnailed; re-encoding leaves that untouched and
# `width == 500 -> real` still scores 100%.
# --------------------------------------------------------------------------- #


def _chameleon_like(tmp_path):
    """Reals uniformly 500 px wide at q75 4:2:0; fakes native and much larger."""
    root = tmp_path / "cham"
    real_dir, fake_dir = root / "test" / "0_real", root / "test" / "1_fake"
    for directory in (real_dir, fake_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(6):
        Image.fromarray(_photo(360, 500, seed=index)).save(
            real_dir / f"{index}.jpg", quality=75, subsampling=2
        )
    for index in range(6):
        Image.fromarray(_photo(1024, 1536, seed=90 + index)).save(fake_dir / f"{index}.png")
    rows = [
        builder.describe(path, root, "cham", label, {})
        for label, directory in (("real", real_dir), ("fake", fake_dir))
        for path in sorted(directory.iterdir())
    ]
    return root, rows


def test_the_width_rule_is_perfect_before_the_pipeline_view(tmp_path):
    _root, rows = _chameleon_like(tmp_path)
    before = builder.width_rule_accuracy(rows, 500)

    assert before["accuracy"] == 1.0, "this is the confound the view exists for"
    assert before["reals_at_width"] == 6
    assert before["fakes_at_width"] == 0


def test_the_pipeline_view_destroys_the_width_rule(tmp_path):
    _root, rows = _chameleon_like(tmp_path)
    view_rows, summary = builder.build_pipeline_matched_view(
        rows, "cham", tmp_path / "view", 500, 75, "4:2:0", "lanczos", _NullLog()
    )

    assert summary["resized"] == 6, "every fake is downscaled"
    assert summary["reals_untouched"] == 6
    assert summary["width_rule_before"]["accuracy"] == 1.0
    assert summary["width_rule_after"]["accuracy"] == summary["width_rule_after"]["baseline"], (
        "after the view, knowing the width must be worth nothing"
    )
    assert all(int(r["width"]) == 500 for r in view_rows), "both classes share one width"


def test_the_pipeline_view_preserves_aspect_ratio(tmp_path):
    _root, rows = _chameleon_like(tmp_path)
    view_rows, _ = builder.build_pipeline_matched_view(
        rows, "cham", tmp_path / "view", 500, 75, "4:2:0", "lanczos", _NullLog()
    )
    fake = next(r for r in view_rows if r["label"] == "fake")

    # 1536x1024 -> 500x333, not squashed to the reals' shape.
    assert (int(fake["width"]), int(fake["height"])) == (500, 333)


def test_the_pipeline_view_leaves_the_reals_alone(tmp_path):
    _root, rows = _chameleon_like(tmp_path)
    before = {r["image_id"]: r["sha256"] for r in rows if r["label"] == "real"}
    view_rows, _ = builder.build_pipeline_matched_view(
        rows, "cham", tmp_path / "view", 500, 75, "4:2:0", "lanczos", _NullLog()
    )

    for row in view_rows:
        if row["label"] == "real":
            assert row["sha256"] == before[row["image_id"]]
            assert row["pipeline_applied"] is False


def test_both_resample_filters_are_available_for_the_sensitivity_check(tmp_path):
    _root, rows = _chameleon_like(tmp_path)
    digests = {}
    for resample in ("lanczos", "bicubic"):
        view_rows, _ = builder.build_pipeline_matched_view(
            rows, "cham", tmp_path / resample, 500, 75, "4:2:0", resample, _NullLog()
        )
        digests[resample] = [r["sha256"] for r in view_rows if r["label"] == "fake"]

    assert digests["lanczos"] != digests["bicubic"], (
        "the filters must differ, or the sensitivity check would be vacuous"
    )


def test_an_unknown_resample_filter_is_refused(tmp_path):
    _root, rows = _chameleon_like(tmp_path)
    with pytest.raises(ValueError, match="resample must be one of"):
        builder.build_pipeline_matched_view(
            rows, "cham", tmp_path / "view", 500, 75, "4:2:0", "nearest", _NullLog()
        )
