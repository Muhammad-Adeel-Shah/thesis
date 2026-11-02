"""P2-B1: decide which images each stream is run on, and write it down.

Three lists, and between them they commit the entire GPU budget:

* ``p2_s1_images.csv`` -- every image S1 caches features for: the reals, every pair row
  and every ``gen_matched`` row that has a split. Cheap, so it is exhaustive.
* ``p2_panel_images.csv`` -- every row of the shared evaluation panel (D-048), which is
  what makes a double-fault rate meaningful: the streams must have seen the same images.
* ``p2_probe_images.csv`` -- the panel plus the S3 training subset and validation. This
  is the expensive one: every row costs twelve decoder reconstructions plus twelve LPIPS
  forwards, so its length is the S3 bill.

**Paths are resolved and verified here, not at extraction time.** The manifests disagree
about what a path is -- the reals and pairs record one relative to the corpus root, while
``p1_gen_matched_manifest.csv`` recorded an absolute path from the session that built it,
with a ``gen_matched/gen_matched`` nesting. Every row written to a list is checked to
exist and stored relative to ``--root``, so a missing file costs one cheap CPU cell here
rather than surfacing nine GPU-hours into Session B.

**The S3 subset is balanced per protocol and seeded** (B5.2): 1,000 training reals shared
across protocols, plus 1,000 training fakes for each protocol spread evenly over that
protocol's active members and ``gen_matched``. A protocol trains its head on fakes only
from lineages it is allowed to see; the probe itself always runs the full twelve-member
bank, because the bank is fixed at test time.

``t_ctrl_diagnostic`` has no evaluation manifest to draw a panel from -- it is the
lossless DIV2K control, built from the reals manifest and their reconstructions -- so
this script also writes ``p2_panel_t_ctrl_diagnostic.csv`` through the same seeded
sampler and the same caps as every other panel, which is what keeps the panel total at
the 12,150 the budget was computed against.

Usage:
    python scripts/p2_build_image_lists.py --root /kaggle/working/p1 \\
        --pairs-manifest reports/p1_pairs_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256, read_manifest, write_manifest  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.panel import (  # noqa: E402
    DIAGNOSTIC_ONLY_SETS,
    LUMA_VIEWS,
    PANEL_CAPS,
    luma_image_id,
    sample_per_stratum,
)
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

LIST_COLUMNS = ("image_id", "relative_path", "label", "view", "root")

#: Which root a row's ``relative_path`` is relative to. The corpus and the frozen panel
#: are two different directories -- the panel is materialised from evaluation sets that
#: are not in the corpus at all -- so a single ``--root`` cannot resolve both, and a
#: list that mixed them silently would fail halfway through a GPU run.
ROOT_CORPUS = "corpus"
ROOT_PANEL = "panel"

#: Protocols the S3 heads are trained for. Secondary runs reuse the same probe
#: features, so only their training fakes differ.
S3_PROTOCOLS = ("p_a_sd1", "4ch_only", "full_bank")

#: B5.2, approved: the S3 training subset.
S3_TRAIN_REALS = 1000
S3_TRAIN_FAKES_PER_PROTOCOL = 1000

#: Validation rows added to the probe list, so the heads have a split to fit on.
S3_VAL_CAP = 2000

SEED = 2026


class ImageListError(RuntimeError):
    """Raised when a list cannot be written as specified."""


def resolve_relative(root: Path, recorded: str, subdir: str) -> str | None:
    """Path of a manifest row relative to ``root``, or ``None`` if it is not there.

    Mirrors ``p2_verify_inputs.resolve_row_path``, and exists for the same reason: a
    path that was absolute in the session that wrote it cannot simply be joined, since
    ``Path(root) / "/abs"`` discards the root silently.

    Args:
        root: Corpus root.
        recorded: The path as the manifest records it.
        subdir: Sub-directory of ``root`` the file belongs in.

    Returns:
        A POSIX-style relative path, or ``None``.
    """
    text = str(recorded).replace("\\", "/")

    if not PurePosixPath(text).is_absolute() and (root / text).is_file():
        return text

    marker = f"/{subdir}/"
    if marker in text:
        tail = text[text.rindex(marker) + 1 :]
        if (root / tail).is_file():
            return tail

    base = root / subdir
    if base.is_dir():
        name = PurePosixPath(text).name
        for hit in base.rglob(name):
            if hit.is_file():
                return hit.relative_to(root).as_posix()
    return None


def balanced_take(
    by_group: dict[str, list[str]], total: int, seed: int, salt: str
) -> list[str]:
    """Take ``total`` ids spread as evenly as possible over the groups, deterministically.

    Groups are visited in sorted order and drawn from round-robin, so a group with few
    members contributes all of them and the rest absorb the remainder. Selection inside
    a group is by a stable hash of the id, never by position, so a manifest read in a
    different order yields the same subset.

    Args:
        by_group: Group label -> candidate ids.
        total: How many ids to take in all.
        seed: Base seed.
        salt: Distinguishes one call from another.

    Returns:
        The chosen ids, sorted.
    """
    import hashlib

    def key(group: str, value: str) -> int:
        digest = hashlib.blake2b(
            f"{seed}:{salt}:{group}:{value}".encode(), digest_size=8
        ).digest()
        return int.from_bytes(digest, "big")

    ordered = {
        group: sorted(ids, key=lambda v: key(group, v))
        for group, ids in sorted(by_group.items())
        if ids
    }
    if not ordered:
        return []

    chosen: list[str] = []
    cursors = dict.fromkeys(ordered, 0)
    while len(chosen) < total:
        progressed = False
        for group, ids in ordered.items():
            if len(chosen) >= total:
                break
            index = cursors[group]
            if index < len(ids):
                chosen.append(ids[index])
                cursors[group] = index + 1
                progressed = True
        if not progressed:
            break  # every group exhausted; the population is smaller than `total`
    return sorted(chosen)


def write_list(rows: list[dict[str, Any]], path: Path, log: Any) -> dict[str, Any]:
    """Write one list, refusing duplicate ids.

    Raises:
        ImageListError: If an id appears twice. A duplicate would be reconstructed
            twice by S3 and would weight that image twice in every metric.
    """
    seen = Counter(row["image_id"] for row in rows)
    duplicates = [image_id for image_id, n in seen.items() if n > 1]
    if duplicates:
        raise ImageListError(
            f"{path.name}: {len(duplicates)} duplicate image_id(s), e.g. "
            f"{duplicates[:5]}. A duplicate is reconstructed twice and counted twice."
        )
    write_manifest(rows, path, LIST_COLUMNS)
    summary = {
        "list": path.name,
        "rows": len(rows),
        "by_view": dict(sorted(Counter(str(r["view"]) for r in rows).items())),
        "by_label": dict(sorted(Counter(str(r["label"]) for r in rows).items())),
        "by_root": dict(sorted(Counter(str(r.get("root", "")) for r in rows).items())),
        "sha256": file_sha256(path),
    }
    log.info("%-26s %6d rows  %s", path.name, len(rows), summary["by_view"])
    return summary


def main() -> int:
    """Write the three image lists. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Staged corpus root.")
    parser.add_argument("--manifests", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--splits", type=Path, default=THESIS_DBP / "reports" / "p2_splits.csv")
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    parser.add_argument(
        "--panel-list", type=Path, default=None,
        help="The materialised panel list from p2_materialise_panel.py, whose paths are "
             "relative to the panel root. Without it the panel rows keep whatever path "
             "their view manifest recorded, which is only valid in the session that "
             "built that view.",
    )
    parser.add_argument(
        "--luma-list", type=Path, default=None,
        help="The luma companion list from p2_materialise_panel.py --luma-subsets. Its "
             "rows join the probe list under `<id>__luma` and are written separately "
             "for the S1 cache (pre-registration section 8).",
    )
    parser.add_argument(
        "--train-reals", type=int, default=S3_TRAIN_REALS,
        help="S3 training reals, shared across protocols (B5.2).",
    )
    parser.add_argument(
        "--train-fakes-per-protocol", type=int, default=S3_TRAIN_FAKES_PER_PROTOCOL,
        help="S3 training fakes per protocol. Lowering this and --val-cap is the "
             "first lever if the measured S3 cost overruns its allowance.",
    )
    parser.add_argument("--val-cap", type=int, default=S3_VAL_CAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "reports")
    args = parser.parse_args()
    log = setup_logging("p2_build_image_lists")

    root: Path = args.root
    if not root.is_dir():
        log.error("corpus root not found: %s", root)
        return 2
    if not args.splits.is_file():
        log.error("%s not found -- run p2_build_splits.py first", args.splits)
        return 2

    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    runs = load_run_configs(THESIS_DBP / "configs" / "p2_runs.yaml", bank)

    side: dict[str, str] = {}
    kind: dict[str, str] = {}
    for row in read_manifest(args.splits):
        side[row["row_id"]] = row["split"]
        kind[row["row_id"]] = row["kind"]
    log.info("splits: %s", dict(sorted(Counter(side.values()).items())))

    # ---- every corpus row, resolved and verified ----------------------------
    corpus: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []

    def add(
        image_id: str, recorded: str, subdir: str, label: str,
        member: str, source: str = "", sha256: str = "", set_id: str = "",
    ) -> None:
        relative = resolve_relative(root, recorded, subdir)
        if relative is None:
            unresolved.append(f"{subdir}:{image_id}")
            return
        corpus[image_id] = {
            "image_id": image_id,
            "relative_path": relative,
            "label": label,
            "view": side.get(image_id, ""),
            "member": member,
            "root": ROOT_CORPUS,
            # The real a row came from: its own id for a real, the reconstructed real
            # for a pair row. Selecting T-ctrl's reconstructions by an id prefix
            # happened to work on the real corpus and is wrong in general.
            "source": source or image_id,
            # Carried from the source manifest, not computed. The T-ctrl diagnostic
            # panel is built from these rows and p2_materialise_panel.py verifies every
            # copy against this value; a blank here makes that panel unmaterialisable.
            "sha256": sha256,
            # Which P1 set the row belongs to, or the real it reconstructs belongs to.
            # The training lists filter on this: T-ctrl is diagnostic-only (B1), and
            # without the field there is nothing to filter on, because a pair row's id
            # is `<real>__<member>` and parsing ids is how I-043 happened.
            "set_id": set_id,
        }

    # D-039 is a flag, not a deletion, and the flag has to be honoured by everything
    # that reads the manifest -- including for the rows *derived* from a flagged real.
    flagged: set[str] = set()
    for set_id in ("t_real", "t_ctrl"):
        for row in read_manifest(args.manifests / f"p1_reals_manifest_{set_id}.csv"):
            if row.get("excluded_reason"):
                flagged.add(row["image_id"])
                continue
            add(
                row["image_id"], row["relative_path"], "reals", "real", "",
                sha256=row.get("sha256", ""), set_id=set_id,
            )

    # A flagged real's twelve reconstructions are greyscale too. Keeping them while
    # dropping the real puts the D-039 shortcut straight back: a greyscale image in the
    # fake class and none in the real class is the cheapest feature in the corpus.
    skipped_pairs = 0
    for row in read_manifest(args.pairs_manifest):
        if row["image_id"] in flagged:
            skipped_pairs += 1
            continue
        add(
            row["pair_id"], row["relative_path"], "pairs", "fake",
            row["member"], row["image_id"], sha256=row.get("sha256", ""),
            set_id=row.get("set_id", ""),
        )
    log.info(
        "D-039: %d flagged real(s) and their %d reconstruction(s) excluded",
        len(flagged), skipped_pairs,
    )

    for row in read_manifest(args.manifests / "p1_gen_matched_manifest.csv"):
        add(
            row["gen_id"], row["jpg_path"], "gen_matched", "fake", "gen_matched",
            sha256=row.get("jpg_sha256", ""), set_id="gen_matched",
        )

    log.info("corpus rows resolved: %d", len(corpus))
    if unresolved:
        log.error(
            "%d manifest row(s) could not be resolved under %s, e.g. %s",
            len(unresolved), root, unresolved[:5],
        )
        return 1

    # ---- the t_ctrl diagnostic panel, which has no eval manifest -------------
    cap = PANEL_CAPS["t_ctrl_diagnostic"]["cap"]
    t_ctrl_ids = {
        row["image_id"]
        for row in read_manifest(args.manifests / "p1_reals_manifest_t_ctrl.csv")
        if not row.get("excluded_reason")
    }
    # A T-ctrl real, or a reconstruction OF a T-ctrl real -- matched on the recorded
    # source, not on an id prefix.
    diagnostic_candidates = [
        entry
        for image_id, entry in sorted(corpus.items())
        if image_id in t_ctrl_ids
        or (entry["label"] == "fake" and entry["source"] in t_ctrl_ids)
    ]
    if not diagnostic_candidates:
        log.error("no T-ctrl rows resolved; the lossless diagnostic panel would be empty")
        return 1
    diagnostic, diagnostic_summary = sample_per_stratum(
        diagnostic_candidates, ("label",), cap, "t_ctrl_diagnostic"
    )
    unhashed = [row["image_id"] for row in diagnostic if not row.get("sha256")]
    if unhashed:
        log.error(
            "%d of %d T-ctrl diagnostic row(s) carry no sha256, e.g. %s. "
            "p2_materialise_panel.py verifies every panel copy against this value, so "
            "the panel could not be frozen. The hash comes from the reals and pair "
            "manifests -- check they have a `sha256` column.",
            len(unhashed), len(diagnostic), unhashed[:3],
        )
        return 1
    for entry in diagnostic:
        entry["panel_set_id"] = "t_ctrl_diagnostic"
        entry["panel_stratum"] = entry["label"]
    write_manifest(
        diagnostic,
        args.out_dir / "p2_panel_t_ctrl_diagnostic.csv",
        (*LIST_COLUMNS, "member", "source", "set_id", "panel_set_id", "panel_stratum",
         "sha256"),
    )
    log.info(
        "t_ctrl diagnostic panel: %d rows %s",
        len(diagnostic), {k: v["taken"] for k, v in diagnostic_summary["by_stratum"].items()},
    )

    # ---- the panel list, from every p2_panel_*.csv ---------------------------
    panel_rows: list[dict[str, Any]] = []
    # Named from PANEL_CAPS rather than globbed. `p2_panel_images.csv` is one of this
    # script's own outputs and would be picked up by a `p2_panel_*.csv` glob on any
    # second run, at which point its missing `panel_set_id` column is a KeyError.
    panel_files = [
        path
        for path in (args.out_dir / f"p2_panel_{set_id}.csv" for set_id in sorted(PANEL_CAPS))
        if path.is_file()
    ]
    if not panel_files:
        log.error(
            "none of %s found under %s -- run p2_build_panel.py first",
            [f"p2_panel_{s}.csv" for s in sorted(PANEL_CAPS)], args.out_dir,
        )
        return 2
    missing_panels = [
        set_id for set_id in sorted(PANEL_CAPS)
        if not (args.out_dir / f"p2_panel_{set_id}.csv").is_file()
    ]
    if missing_panels:
        log.info(
            "%d panel(s) absent and therefore unscored: %s", len(missing_panels), missing_panels
        )
    if args.panel_list and args.panel_list.is_file():
        for row in read_manifest(args.panel_list):
            panel_rows.append(
                {
                    "image_id": row["image_id"],
                    "relative_path": row["relative_path"],
                    "label": row.get("label", ""),
                    "view": row["view"],
                    "root": ROOT_PANEL,
                }
            )
        log.info(
            "panel from the materialised list %s: %d row(s), paths relative to the "
            "panel root", args.panel_list.name, len(panel_rows),
        )
    else:
        log.info(
            "no --panel-list given: panel paths are taken from the view manifests and "
            "are only valid in the session that built those views. Run "
            "p2_materialise_panel.py and pass --panel-list before Session B."
        )
        for path in panel_files:
            for row in read_manifest(path):
                recorded = str(row.get("view_path") or row.get("relative_path") or "")
                panel_rows.append(
                    {
                        "image_id": row["image_id"],
                        "relative_path": recorded,
                        "label": row.get("label", ""),
                        "view": row["panel_set_id"],
                        "root": ROOT_PANEL,
                    }
                )

    # ---- diagnostic-only sets leave the training lists ------------------------
    # B1: T-ctrl is measured, never trained on. The split label on those rows is a
    # split label and nothing more -- the rows had been landing in every training list
    # because `corpus` is the population for both the lists and the diagnostic panel.
    def trainable(entry: dict[str, Any]) -> bool:
        """Whether a corpus row may enter a training list."""
        return str(entry.get("set_id", "")) not in DIAGNOSTIC_ONLY_SETS

    withheld = [entry for entry in corpus.values() if not trainable(entry)]
    log.info(
        "diagnostic-only (%s): %d row(s) withheld from every training list "
        "(%d real, %d reconstruction); they stay in the diagnostic panel",
        ",".join(sorted(DIAGNOSTIC_ONLY_SETS)), len(withheld),
        sum(1 for e in withheld if e["label"] == "real"),
        sum(1 for e in withheld if e["label"] == "fake"),
    )

    # ---- S1: everything with a split, bar the diagnostic-only sets -----------
    s1_rows = [
        {k: entry[k] for k in LIST_COLUMNS}
        for image_id, entry in sorted(corpus.items())
        if image_id in side and trainable(entry)
    ]

    # ---- S3: panel + a balanced, seeded training subset + validation --------
    train_reals = {
        image_id: entry
        for image_id, entry in corpus.items()
        if side.get(image_id) == "train" and entry["label"] == "real" and trainable(entry)
    }
    chosen_reals = balanced_take(
        {"real": sorted(train_reals)}, args.train_reals, args.seed, "s3_reals"
    )

    chosen_fakes: set[str] = set()
    per_protocol: dict[str, int] = {}
    for protocol in S3_PROTOCOLS:
        allowed = set(runs.by_id(protocol).active_members(bank)) | {"gen_matched"}
        by_member: dict[str, list[str]] = defaultdict(list)
        for image_id, entry in corpus.items():
            if side.get(image_id) != "train" or entry["label"] != "fake":
                continue
            if not trainable(entry):
                continue
            if entry["member"] in allowed:
                by_member[entry["member"]].append(image_id)
        picked = balanced_take(
            by_member, args.train_fakes_per_protocol, args.seed, f"s3_fakes:{protocol}"
        )
        per_protocol[protocol] = len(picked)
        chosen_fakes.update(picked)
        log.info(
            "%-12s %d member(s) allowed, %d fake(s) picked",
            protocol, len(by_member), len(picked),
        )

    by_label_val: dict[str, list[str]] = defaultdict(list)
    for image_id, entry in corpus.items():
        # Validation fits the temperatures, the thresholds and the stacker, so a
        # diagnostic-only row here is still a row the streams were tuned on.
        if side.get(image_id) == "val" and trainable(entry):
            by_label_val[entry["label"]].append(image_id)
    chosen_val = balanced_take(by_label_val, args.val_cap, args.seed, "s3_val")

    # The panel already carries the T-ctrl diagnostic rows, and those rows are also
    # training reals, so they would be added twice -- once as panel, once as subset.
    # A duplicate is reconstructed twice by S3 and weighted twice in every metric, so
    # the panel wins and the subset is taken from what is left.
    panel_ids = {row["image_id"] for row in panel_rows}
    probe_ids = sorted({*chosen_reals, *chosen_fakes, *chosen_val} - panel_ids)
    overlap = len({*chosen_reals, *chosen_fakes, *chosen_val}) - len(probe_ids)
    if overlap:
        log.info(
            "%d subset row(s) are already in the panel and are not listed twice", overlap
        )
    probe_rows = panel_rows + [
        {k: corpus[image_id][k] for k in LIST_COLUMNS} for image_id in probe_ids
    ]

    # ---- the luma-only companion (pre-registration §8) -----------------------
    # Its rows carry `<id>__luma`, not their parent's id. They have to be in the SAME
    # probe list as everything else: p2_probe_features.py writes one .npz per bank
    # member and invalidates it when the image-list digest changes, so a second run
    # over a luma-only list would recompute and overwrite the first.
    luma_rows: list[dict[str, Any]] = []
    luma_list = args.luma_list
    if luma_list and luma_list.is_file():
        views = sorted(LUMA_VIEWS)
        for row in read_manifest(luma_list):
            view = str(row.get("view", ""))
            if view not in LUMA_VIEWS:
                log.error(
                    "%s row %s has view %r, which is not a declared luma view %s",
                    luma_list.name, row.get("image_id"), view, views,
                )
                return 1
            luma_rows.append(
                {
                    "image_id": luma_image_id(row["image_id"]),
                    "relative_path": row["relative_path"],
                    "label": row.get("label", ""),
                    "view": view,
                    "root": ROOT_PANEL,
                }
            )
        probe_rows = probe_rows + luma_rows
        log.info(
            "luma companion: %d row(s) from %s, added to the probe list and written to "
            "p2_luma_images.csv for the S1 cache", len(luma_rows), luma_list.name,
        )
    else:
        log.info(
            "no luma list at %s: the FLUX.1 luma-only slice cannot be scored, and "
            "pre-registration §8 makes the standard FLUX.1 number unclaimable. Run "
            "p2_materialise_panel.py with --luma-subsets first.",
            luma_list if luma_list else "(not given)",
        )

    # ---- write ---------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        write_list(s1_rows, args.out_dir / "p2_s1_images.csv", log),
        write_list(panel_rows, args.out_dir / "p2_panel_images.csv", log),
        write_list(probe_rows, args.out_dir / "p2_probe_images.csv", log),
    ]
    if luma_rows:
        summaries.append(write_list(luma_rows, args.out_dir / "p2_luma_images.csv", log))

    report = {
        "root": str(root),
        "seed": args.seed,
        "decision": "D-048",
        "d039_excluded": {"reals": len(flagged), "pair_rows": skipped_pairs},
        "diagnostic_only": {
            "sets": sorted(DIAGNOSTIC_ONLY_SETS),
            "rule": "P2 prompt B1: measured, never trained on",
            "withheld_rows": len(withheld),
            "withheld_reals": sum(1 for e in withheld if e["label"] == "real"),
            "withheld_reconstructions": sum(1 for e in withheld if e["label"] == "fake"),
        },
        "luma": {
            "list": luma_list.name if luma_list and luma_list.is_file() else None,
            "rows": len(luma_rows),
            "views": sorted({r["view"] for r in luma_rows}),
            "id_suffix": "__luma",
            "why": (
                "pre-registration section 8: the luma-only FLUX.1 number is the one "
                "that enters M1 and M2"
            ),
        },
        "s3_subset": {
            "train_reals": len(chosen_reals),
            "train_reals_requested": args.train_reals,
            "train_fakes_per_protocol": per_protocol,
            "train_fakes_requested": args.train_fakes_per_protocol,
            "train_fakes_distinct": len(chosen_fakes),
            "val": len(chosen_val),
            "val_cap": args.val_cap,
            "already_in_panel": overlap,
        },
        "t_ctrl_diagnostic_panel": diagnostic_summary,
        "panel_files": [p.name for p in panel_files],
        "panel_list": args.panel_list.name if args.panel_list else None,
        "panels_absent": missing_panels,
        "lists": summaries,
        "note": (
            "p2_probe_images.csv is the S3 bill: every row costs twelve reconstructions "
            "plus twelve LPIPS forwards. The 9.63 GPU-h estimate assumed 18,150 rows."
        ),
    }
    path = args.out_dir / "p2_image_lists_summary.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

    probe_n = summaries[2]["rows"]
    log.info("S3 probe list: %d rows (budget assumed 18,150)", probe_n)
    if probe_n > 20000:
        log.info(
            "that is %.0f%% over the budgeted 18,150 -- report before starting the GPU "
            "extraction", 100 * (probe_n / 18150 - 1),
        )
    log.info("wrote reports/%s", path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
