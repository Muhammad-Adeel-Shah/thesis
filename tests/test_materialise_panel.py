"""Tests for freezing the evaluation panel (B1b, D-050).

The properties that matter: every copy is verified against the hash the manifest
already recorded, a mismatch stops the run rather than re-recording, no rewritten path
can leave the panel root, and the digest identifies the panel rather than the order it
was built in.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
THESIS_DBP = Path(__file__).resolve().parents[1]


def _load():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "p2_materialise_panel", SCRIPTS / "p2_materialise_panel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mat = _load()


# ------------------------------------------------------------- safe_name ---


def test_a_plain_id_becomes_a_file_name() -> None:
    assert mat.safe_name("2ab2fdc72c37441b", ".png") == "2ab2fdc72c37441b.png"


@pytest.mark.parametrize(
    "bad", ["../escape", "a/b", "..", "C:evil", "", "dir/../x"],
)
def test_an_id_that_could_escape_the_root_is_refused_not_cleaned(bad: str) -> None:
    """A sanitised path is a path nobody checked."""
    with pytest.raises(mat.MaterialiseError, match="not usable as a file name"):
        mat.safe_name(bad, ".png")


# ---------------------------------------------------------- resolve_source ---


def test_view_path_wins_when_it_exists(tmp_path: Path) -> None:
    derived = tmp_path / "derived.jpg"
    derived.write_bytes(b"d")
    row = {"view_path": str(derived), "relative_path": "nope.jpg", "source_path": ""}
    assert mat.resolve_source(row, "v", {}) == derived


def test_relative_path_resolves_under_the_view_root(tmp_path: Path) -> None:
    root = tmp_path / "e7"
    (root / "FLUX.1").mkdir(parents=True)
    image = root / "FLUX.1" / "x.jpg"
    image.write_bytes(b"x")
    row = {"view_path": "", "relative_path": "FLUX.1/x.jpg", "source_path": ""}
    assert mat.resolve_source(row, "e7_dda_coco", {"e7_dda_coco": root}) == image


def test_source_path_is_the_last_resort_for_an_underived_row(tmp_path: Path) -> None:
    image = tmp_path / "orig.png"
    image.write_bytes(b"o")
    row = {"view_path": "", "relative_path": "", "source_path": str(image)}
    assert mat.resolve_source(row, "v", {}) == image


def test_a_derived_row_never_falls_back_to_its_pre_derivation_source(
    tmp_path: Path,
) -> None:
    """The bug the hash check caught on E5.

    An E5 pipeline row records a 500-px derived JPEG, while its relative_path and
    source_path both point at the untouched 4K Chameleon original. Returning that
    original hands back a different picture whose hash cannot match -- and if the
    recorded hash had been trusted instead of checked, the panel would have held the
    wrong images with no sign of it.
    """
    chameleon = tmp_path / "chameleon"
    (chameleon / "test" / "1_fake").mkdir(parents=True)
    original = chameleon / "test" / "1_fake" / "ceca3448.jpg"
    _image(original, 1)

    row = {
        "view_path": "/kaggle/working/gone/eval/e5_pipeline/11cbe530.jpg",
        "relative_path": "test/1_fake/ceca3448.jpg",
        "source_path": str(original),
    }
    # Given the Chameleon root, the only honest answer is "not here".
    assert mat.resolve_source(row, "e5", {"e5": chameleon}) is None

    # Given the derived view's root, the basename resolves.
    derived_dir = tmp_path / "pipeline"
    derived_dir.mkdir()
    derived = derived_dir / "11cbe530.jpg"
    _image(derived, 2)
    assert mat.resolve_source(row, "e5", {"e5": derived_dir}) == derived


def test_nothing_resolvable_returns_none(tmp_path: Path) -> None:
    row = {"view_path": "/gone/a", "relative_path": "b", "source_path": "/gone/c"}
    assert mat.resolve_source(row, "v", {"v": tmp_path}) is None


# -------------------------------------------------------------- the digest ---


def test_the_digest_does_not_depend_on_row_order() -> None:
    rows = [{"image_id": f"i{i}", "sha256": f"{i:064d}"} for i in range(20)]
    assert mat.panel_digest(rows) == mat.panel_digest(list(reversed(rows)))


def test_the_digest_changes_when_any_hash_changes() -> None:
    rows = [{"image_id": f"i{i}", "sha256": f"{i:064d}"} for i in range(5)]
    before = mat.panel_digest(rows)
    rows[2] = {"image_id": "i2", "sha256": "f" * 64}
    assert mat.panel_digest(rows) != before


def test_the_digest_changes_when_a_row_is_dropped() -> None:
    rows = [{"image_id": f"i{i}", "sha256": f"{i:064d}"} for i in range(5)]
    assert mat.panel_digest(rows) != mat.panel_digest(rows[:-1])


def test_the_digest_matches_its_documented_definition() -> None:
    rows = [{"image_id": "b", "sha256": "2" * 64}, {"image_id": "a", "sha256": "1" * 64}]
    expected = hashlib.sha256(
        f"a\t{'1' * 64}\nb\t{'2' * 64}".encode()
    ).hexdigest()
    assert mat.panel_digest(rows) == expected


# ------------------------------------------------------------ parse_mapping ---


def test_mapping_parses_repeated_pairs() -> None:
    out = mat.parse_mapping(["a=/x", "b=/y/z"], "source")
    assert out == {"a": Path("/x"), "b": Path("/y/z")}


def test_a_mapping_without_an_equals_is_refused() -> None:
    with pytest.raises(mat.MaterialiseError, match="expects view=path"):
        mat.parse_mapping(["oops"], "source")


# -------------------------------------------------------------- end to end ---


def _image(path: Path, seed: int, mode: str = "JPEG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path, format=mode)


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Path]:
    """One panel CSV per approved view, with real files behind every row."""
    from dbp.data.manifest import file_sha256
    from dbp.eval.panel import PANEL_CAPS

    reports = tmp_path / "reports"
    e7_root = tmp_path / "e7"
    other = tmp_path / "other"
    columns = ["image_id", "relative_path", "source_path", "view_path",
               "label", "sha256", "source_vae", "panel_set_id"]

    for view in sorted(PANEL_CAPS):
        rows = []
        for i in range(4):
            label = "real" if i % 2 else "fake"
            image_id = f"{view[:3]}{i:04d}"
            if view == "e7_dda_coco":
                vae = "FLUX.1" if i < 2 else "sdxl-vae"
                if label == "real":
                    vae = "real"
                relative = f"{vae}/{image_id}.jpg"
                path = e7_root / relative
                _image(path, i)
                rows.append({
                    "image_id": image_id, "relative_path": relative,
                    "source_path": "", "view_path": "", "label": label,
                    "sha256": file_sha256(path), "source_vae": vae,
                    "panel_set_id": view,
                })
            else:
                path = other / view / f"{image_id}.png"
                _image(path, 100 + i, mode="PNG")
                rows.append({
                    "image_id": image_id, "relative_path": "",
                    "source_path": "", "view_path": str(path), "label": label,
                    "sha256": file_sha256(path), "source_vae": "",
                    "panel_set_id": view,
                })
        _write(reports / f"p2_panel_{view}.csv", columns, rows)

    return {"reports": reports, "e7": e7_root, "panel": tmp_path / "panel"}


def _run(fixture: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable, str(SCRIPTS / "p2_materialise_panel.py"),
        "--manifests", str(fixture["reports"]),
        "--panel-root", str(fixture["panel"]),
        "--source", f"e7_dda_coco={fixture['e7']}",
        "--expect-rows", "24",            # 6 views x 4 rows in the fixture
        *extra,
    ]
    return subprocess.run(argv, capture_output=True, text=True, cwd=THESIS_DBP)


def test_end_to_end_materialises_every_view(fixture: dict[str, Path]) -> None:
    result = _run(fixture)
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8")))
    assert len(rows) == 24
    for row in rows:
        assert (fixture["panel"] / row["relative_path"]).is_file(), row


def test_every_rewritten_path_stays_inside_the_panel_root(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    root = fixture["panel"].resolve()
    for row in csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8")
    ):
        assert not Path(row["relative_path"]).is_absolute()
        assert (fixture["panel"] / row["relative_path"]).resolve().is_relative_to(root)


def test_the_copy_is_byte_identical_to_the_recorded_hash(fixture: dict[str, Path]) -> None:
    from dbp.data.manifest import file_sha256

    assert _run(fixture).returncode == 0
    for row in csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8")
    ):
        assert file_sha256(fixture["panel"] / row["relative_path"]) == row["sha256"]


def test_the_source_suffix_is_preserved(fixture: dict[str, Path]) -> None:
    """Re-encoding would change the pixels the panel was selected on."""
    assert _run(fixture).returncode == 0
    by_view = {}
    for row in csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8")
    ):
        by_view.setdefault(row["view"], set()).add(Path(row["relative_path"]).suffix)
    assert by_view["e7_dda_coco"] == {".jpg"}
    assert by_view["e8_evalgen"] == {".png"}


def test_a_changed_source_stops_the_run_and_does_not_rewrite_the_hash(
    fixture: dict[str, Path],
) -> None:
    """A re-derived matched view must be byte-identical (I-010)."""
    _image(fixture["e7"] / "FLUX.1" / "e7_0000.jpg", 999)
    result = _run(fixture)
    assert result.returncode == 1
    assert "sha256 mismatch" in result.stderr + result.stdout
    assert not (fixture["reports"] / "p2_panel_digest.json").is_file()


def test_a_missing_source_stops_the_run(fixture: dict[str, Path]) -> None:
    (fixture["e7"] / "FLUX.1" / "e7_0000.jpg").unlink()
    result = _run(fixture)
    assert result.returncode == 1
    assert "not found" in result.stderr + result.stdout


def test_a_row_without_a_recorded_hash_is_refused(fixture: dict[str, Path]) -> None:
    path = fixture["reports"] / "p2_panel_e8_evalgen.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    columns = list(rows[0])
    rows[0]["sha256"] = ""
    _write(path, columns, rows)
    result = _run(fixture)
    assert result.returncode == 1
    assert "records no sha256" in result.stderr + result.stdout


def test_a_wrong_row_count_refuses_to_write_the_digest(fixture: dict[str, Path]) -> None:
    argv = [
        sys.executable, str(SCRIPTS / "p2_materialise_panel.py"),
        "--manifests", str(fixture["reports"]),
        "--panel-root", str(fixture["panel"]),
        "--source", f"e7_dda_coco={fixture['e7']}",
        "--expect-rows", "12150",
    ]
    result = subprocess.run(argv, capture_output=True, text=True, cwd=THESIS_DBP)
    assert result.returncode == 1
    assert "Not writing the digest" in result.stderr + result.stdout
    assert not (fixture["reports"] / "p2_panel_digest.json").is_file()


def test_the_digest_is_recorded_and_reproducible(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    first = json.loads((fixture["reports"] / "p2_panel_digest.json").read_text())
    assert len(first["panel_digest"]) == 64
    assert first["rows"] == 24
    assert _run(fixture).returncode == 0          # idempotent: files already present
    second = json.loads((fixture["reports"] / "p2_panel_digest.json").read_text())
    assert second["panel_digest"] == first["panel_digest"]


def test_the_luma_variant_is_greyscale_and_not_counted_in_the_panel(
    fixture: dict[str, Path],
) -> None:
    from dbp.data.colour import classify_colour
    from dbp.utils.io import load_rgb_uint8

    assert _run(fixture, "--luma-subsets", "e7_dda_coco:FLUX.1,real").returncode == 0

    panel = list(csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8")))
    luma = list(csv.DictReader(
        (fixture["reports"] / "p2_panel_images_luma.csv").open(encoding="utf-8")))

    assert len(panel) == 24, "the luma rows must not inflate the panel"
    assert luma, "the pre-registration requires this variant"
    for row in luma:
        assert row["view"] == "e7_dda_coco_luma"
        assert Path(row["relative_path"]).suffix == ".png"
        image = load_rgb_uint8(fixture["panel"] / row["relative_path"])
        verdict = classify_colour(image)
        assert verdict["is_greyscale"] is True
        assert verdict["channel_spread"] == 0

    summary = json.loads((fixture["reports"] / "p2_panel_digest.json").read_text())
    assert summary["luma_variant"]["rows"] == len(luma)
    assert summary["luma_variant"]["digest"] != summary["panel_digest"]


def test_the_luma_variant_covers_both_classes(fixture: dict[str, Path]) -> None:
    """Chroma can only be made uninformative if both classes are converted."""
    assert _run(fixture, "--luma-subsets", "e7_dda_coco:FLUX.1,real").returncode == 0
    labels = {r["label"] for r in csv.DictReader(
        (fixture["reports"] / "p2_panel_images_luma.csv").open(encoding="utf-8"))}
    assert labels == {"real", "fake"}, labels


def test_an_empty_luma_selection_stops_the_run(fixture: dict[str, Path]) -> None:
    result = _run(fixture, "--luma-subsets", "e7_dda_coco:NoSuchSubset")
    assert result.returncode == 1
    assert "would be empty" in result.stderr + result.stdout


def test_a_missing_panel_csv_stops_the_run(fixture: dict[str, Path]) -> None:
    (fixture["reports"] / "p2_panel_e3_matched.csv").unlink()
    result = _run(fixture)
    assert result.returncode == 2
    assert "run p2_build_panel.py" in result.stderr + result.stdout


def test_a_derived_view_can_be_attached_rather_than_rebuilt(tmp_path: Path) -> None:
    """E5's pipeline JPEGs exist in an earlier session's output, at a dead path.

    Their `relative_path` still points at the untouched Chameleon source, which is a
    different image whose hash would not match, so the basename of `view_path` under
    the attached root is the only correct resolution.
    """
    attached = tmp_path / "attached" / "e5_pipeline"
    attached.mkdir(parents=True)
    derived = attached / "abc123.jpg"
    _image(derived, 7)
    row = {
        "view_path": "/kaggle/working/gone/outputs/p1/eval/e5_pipeline/abc123.jpg",
        "relative_path": "test/1_fake/original.jpg",
        "source_path": "/tmp/gone/original.jpg",
    }
    assert mat.resolve_source(row, "e5", {"e5": tmp_path / "attached"}) == derived


def test_an_underived_row_prefers_its_relative_path_over_a_same_named_decoy(
    tmp_path: Path,
) -> None:
    """With no view_path the hash is of the source, so relative_path is authoritative."""
    root = tmp_path / "r"
    (root / "sub").mkdir(parents=True)
    correct = root / "sub" / "x.jpg"
    _image(correct, 1)
    decoy = root / "elsewhere" / "x.jpg"
    _image(decoy, 2)
    row = {"view_path": "", "relative_path": "sub/x.jpg", "source_path": str(decoy)}
    assert mat.resolve_source(row, "v", {"v": root}) == correct


def test_a_derived_row_ignores_relative_path_entirely(tmp_path: Path) -> None:
    """relative_path names the pre-derivation image, so it must not be consulted."""
    root = tmp_path / "r"
    (root / "sub").mkdir(parents=True)
    pre_derivation = root / "sub" / "x.jpg"
    _image(pre_derivation, 1)
    derived = root / "derived" / "other.jpg"
    _image(derived, 2)
    row = {"view_path": "/gone/other.jpg", "relative_path": "sub/x.jpg", "source_path": ""}
    assert mat.resolve_source(row, "v", {"v": root}) == derived


# --------------------------------------------------- I-047: --min-side on the panel ---


def _add_sizes(fixture: dict[str, Path], sizes: dict[str, tuple[int, int]]) -> None:
    """Rewrite each view CSV with width/height columns, defaulting to 512x512."""
    for path in sorted(fixture["reports"].glob("p2_panel_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            w, h = sizes.get(row["image_id"], (512, 512))
            row["width"], row["height"] = str(w), str(h)
        _write(path, [*rows[0].keys()], rows)


def test_min_side_drops_rows_below_the_evaluation_crop(fixture: dict[str, Path]) -> None:
    _add_sizes(fixture, {"e7_0000": (512, 200), "e3_0001": (200, 512)})
    result = _run(fixture, "--min-side", "320", "--expect-rows", "22")
    assert result.returncode == 0, result.stdout + result.stderr
    listed = list(csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8")))
    ids = {r["image_id"] for r in listed}
    assert len(listed) == 22
    assert "e7_0000" not in ids and "e3_0001" not in ids


def test_the_dropped_rows_are_recorded_per_view_and_label(fixture: dict[str, Path]) -> None:
    _add_sizes(fixture, {"e7_0000": (512, 200)})
    assert _run(fixture, "--min-side", "320", "--expect-rows", "23").returncode == 0
    summary = json.loads(
        (fixture["reports"] / "p2_panel_digest.json").read_text(encoding="utf-8"))
    rule = summary["min_side"]
    assert rule["threshold"] == 320
    assert rule["total_dropped"] == 1
    assert rule["by_view"]["e7_dda_coco"]["dropped"] == 1
    assert "e7_0000" in rule["by_view"]["e7_dda_coco"]["image_ids"]
    assert sum(rule["by_view"]["e7_dda_coco"]["by_label"].values()) == 1


def test_the_luma_companion_stays_a_subset_of_the_filtered_panel(
    fixture: dict[str, Path],
) -> None:
    """A dropped E7 row must take its luma copy with it, or the companion would score
    an image the panel no longer contains."""
    _add_sizes(fixture, {"e7_0000": (512, 200)})
    result = _run(fixture, "--min-side", "320", "--expect-rows", "23",
                  "--luma-subsets", "e7_dda_coco:FLUX.1,real")
    assert result.returncode == 0, result.stdout + result.stderr
    panel_ids = {r["image_id"] for r in csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8"))}
    luma_ids = {r["image_id"] for r in csv.DictReader(
        (fixture["reports"] / "p2_panel_images_luma.csv").open(encoding="utf-8"))}
    assert luma_ids <= panel_ids
    assert "e7_0000" not in luma_ids


def test_rows_without_a_recorded_size_are_kept(fixture: dict[str, Path]) -> None:
    """The fixture records no width/height, as the T-ctrl panel does."""
    assert _run(fixture, "--min-side", "320").returncode == 0
    summary = json.loads(
        (fixture["reports"] / "p2_panel_digest.json").read_text(encoding="utf-8"))
    assert summary["min_side"]["total_dropped"] == 0
    assert summary["rows"] == 24


def test_a_second_run_needs_no_source_once_the_panel_is_present(
    fixture: dict[str, Path],
) -> None:
    """Freezing the panel is pointless if re-deriving its list needs the corpora back.
    A file already under the panel root that hashes correctly IS the panel image."""
    assert _run(fixture).returncode == 0
    first = (fixture["reports"] / "p2_panel_digest.json").read_text(encoding="utf-8")

    argv = [
        sys.executable, str(SCRIPTS / "p2_materialise_panel.py"),
        "--manifests", str(fixture["reports"]),
        "--panel-root", str(fixture["panel"]),
        "--expect-rows", "24",          # no --source at all
    ]
    again = subprocess.run(argv, capture_output=True, text=True, cwd=THESIS_DBP)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "24 row(s): 0 copied, 24 already present" in (again.stdout + again.stderr) or \
        "already present" in (again.stdout + again.stderr)
    assert (fixture["reports"] / "p2_panel_digest.json").read_text(encoding="utf-8") == first
