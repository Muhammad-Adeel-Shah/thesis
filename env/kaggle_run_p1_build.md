# Kaggle run — P1 build (B4 + B5 + B6)

**Purpose.** Build the real-image corpora, the aligned reconstruction pairs and the
SD1.5 generation set. This is the long run; it trains nothing.

**Do not start this before the preflight gate in `kaggle_run_p1_preflight.md` is
green.** Everything below depends on `reports/p1_throughput.csv`, which the preflight
produces, and on the regression check having passed.

**Session.** Accelerator **GPU T4 x2**, internet **on**, persistence on. Budget from the
preflight's measured numbers, not from the P0 extrapolation. Every stage is resumable,
so hitting the 12-hour limit costs time, not work.

---

## Step 1 — attach the inputs

- Your code dataset (`dbp-p1`), as in the preflight.
- The preflight's `p1_throughput.csv`. Either add
  `thesis_dbp/reports/p1_throughput.csv` to the code zip after committing it, or attach
  the preflight output dataset and copy the file in (Cell 3 below).
- **COCO 2017**: Add Data → search `awsaf49/coco-2017-dataset`. It is large; it is read in
  place and **never copied**.

> **Do not hard-code the mount path (I-034).** Kaggle mounts inputs under
> `/kaggle/input/<slug>` in some sessions and `/kaggle/input/datasets/<owner>/<slug>` in
> others. Every cell below finds its inputs by searching. And **always run the code from
> the staged copy under `/kaggle/working`**: the staging cell's
> `assert not str(DBP).startswith("/kaggle/input")` exists because a run straight from
> the read-only mount fails at the first write, with `OSError: Read-only file system`.

> **T-real is built from `train2017` only — never from the dataset root.** E7 DDA-COCO's
> real images are the MSCOCO **validation** set. Pointing `--root` at the dataset root
> would let T-real select `val2017` images, putting E7's evaluation reals into our
> training corpus. That is issue I-024, and it is silent: nothing would fail, the numbers
> would just be wrong. Step 4 finds the train images directory explicitly, and the
> leakage check in `kaggle_run_p1_eval.md` verifies the outcome.

DIV2K is fetched inside the session (Step 3) rather than attached, because the official
archive is the licensed source.

---

## Step 2 — Cells 1–4 from the preflight

Stage the repo, set the caches, run the bootstrap, put `HF_TOKEN` into `os.environ` and
verify the pins. Identical to `kaggle_run_p1_preflight.md`; do not skip the pin check —
CLAUDE.md §7a requires it before anything is written, and each build script re-runs it
itself as a backstop.

---

## Step 3 — DIV2K (T-ctrl)

Allow-listed host `data.vision.ee.ethz.ch` (CLAUDE.md §6a). **The download happens here,
inside the Kaggle session, never to the local machine** — condition (a).

DIV2K is distributed for *academic research only*; that constraint is recorded in
`docs/data_cards/t_ctrl_div2k.md` and in CLAUDE.md §6c.

```python
import pathlib, subprocess
div2k = pathlib.Path("/tmp/div2k")
div2k.mkdir(parents=True, exist_ok=True)
url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
archive = div2k / "DIV2K_train_HR.zip"
if not archive.exists():
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(archive), url], check=True)
print("bytes:", archive.stat().st_size, "(expected 3530603713)")
subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(div2k)], check=True)
print(sum(1 for _ in (div2k / "DIV2K_train_HR").glob("*.png")), "PNGs")
```

`/tmp`, not `/kaggle/working`: the extracted set is ~3.5 GB and is not worth persisting.

---

## Step 4 — B4, the real corpora

First locate the **train** images directory, and check it holds what it should:

```python
import pathlib
candidates = [p for p in pathlib.Path("/kaggle/input").rglob("train2017") if p.is_dir()]
print("train2017 candidates:", [str(p) for p in candidates])
assert len(candidates) == 1, "expected exactly one train2017 directory"
COCO_TRAIN = candidates[0]
CAPTIONS = next(pathlib.Path("/kaggle/input").rglob("captions_train2017.json"))
INSTANCES = next(pathlib.Path("/kaggle/input").rglob("instances_train2017.json"), None)
print(COCO_TRAIN, "->", sum(1 for _ in COCO_TRAIN.glob("*.jpg")), "jpg (expect 118287)")
print("captions :", CAPTIONS)
print("instances:", INSTANCES, "(needed for the greyscale re-scan's supercategory table)")
# Sanity: there must be NO val2017 images under the chosen root (I-024).
assert not list(COCO_TRAIN.rglob("*val2017*")), "val2017 reachable from the train root"
```

The search walks every COCO image once, so give it a minute.

```python
!cd {DBP} && python scripts/p1_build_reals.py --set t_real \
    --root {COCO_TRAIN} --limit 2500 --crop 384
!cd {DBP} && python scripts/p1_build_reals.py --set t_ctrl \
    --root /tmp/div2k/DIV2K_train_HR --limit 1600 --crop 384 --crops-per-image 2
```

`--crops-per-image 2` is how T-ctrl reaches 1,600 rows from DIV2K's 800 photographs
(I-021). The two crops are **non-overlapping**, taken from the middle of the image along
its longer axis. Every row carries `source_image_id`, and **any split must group by that
column** — two crops of one photograph on opposite sides of a train/test boundary is
leakage however disjoint the pixels are. The build logs a warning saying exactly this.
T-ctrl stays **diagnostic-only** for the same reason.

Read the summary lines:
- `kept` vs the limit. If fewer were kept, the corpus really is smaller and every later
  count must say so — do not compensate by loosening the size filter.
- `skipped as smaller than 384px`. COCO has many small images; this is expected and is
  the correct behaviour (D-001 forbids upscaling).
- the near-duplicate groups. They are dropped and listed.
- the compression table: COCO is JPEG, DIV2K is PNG. That difference is why they are two
  separate sets and are never pooled without saying so.
- `distinct_source_images` and `split_grouping_column` in the T-ctrl summary: 1,600 rows
  from ~800 sources. Carry that column into every split you build later.

---

## Step 5 — B5, the reconstruction pairs

Member-major and resumable. The member list comes from `distinct_members()`, so
`sd_ft_mse` is skipped by the registry rather than by a name written into the script.

Two GPUs, two shards. Run both cells; they write to different member directories and
different manifests, so they do not collide.

```python
import subprocess
procs = []
for shard, device in ((0, "cuda:0"), (1, "cuda:1")):
    log = open(f"/kaggle/working/pairs_shard{shard}.log", "w")
    procs.append(subprocess.Popen(
        ["python", "scripts/p1_build_pairs.py",
         "--shard", str(shard), "--num-shards", "2", "--device", device],
        cwd=str(DBP), stdout=log, stderr=subprocess.STDOUT))
for p in procs:
    print("exit", p.wait())
```

```python
!tail -30 /kaggle/working/pairs_shard0.log
!tail -30 /kaggle/working/pairs_shard1.log
```

If the session ends mid-run, re-attach the output dataset, restage and re-run the same
cell: each member's manifest is read first and finished images are skipped.

Merge when both shards are complete:

```python
!cd {DBP} && python scripts/p1_build_pairs.py --merge
```

Check in `reports/p1_pairs_summary_shard*.json`:
- every member `OK` (`PARTIAL` means some images failed; the ids are listed);
- `precision` per member matches `fp16_build_allowed` from the preflight;
- total rows in `p1_pairs_manifest.csv` = kept reals × distinct members.

---

## Step 6 — B6, the SD1.5 generation set

```python
!cd {DBP} && python scripts/p1_build_generations.py \
    --captions {CAPTIONS} --limit 2500 --steps 25 --crop 384
```

If that path does not exist, find it first — the COCO mirror's internal layout is not
guaranteed:

```python
import pathlib
print([str(p) for p in pathlib.Path("/kaggle/input/coco-2017-dataset").rglob("captions_train2017.json")])
```

Read out of `reports/p1_gen_manifest.json`:
- `safety_checker`: must be `enabled`. If it says `absent`, the checkpoint shipped none
  and the data card must state that rather than implying a filter was applied.
- `safety_checker_flagged_and_discarded`: explains any shortfall against `--limit`.
- `near_duplicate_groups`: recorded, **not** dropped. Duplicate generations are a
  property of the generator and belong in the report.

---

## Step 6b — the JPEG-matched generation variant (I-023)

T-real is entirely JPEG; the generations above are PNG that have never been compressed.
Trained on that pairing, the easiest separating feature is not a generator artefact at
all — it is the presence or absence of an 8×8 quantisation lattice. That is the same
"Fake or JPEG?" confound [R7] that §6b fixes on the evaluation side, sitting in the
**training** data, where it is worse: a model can learn it and then appear to generalise.

This step writes a **second variant** of every generation, encoded with a
`(quality, chroma subsampling)` pair drawn from the reals it is matched against. The PNG
variant is kept: P2 trains on `gen_matched` by default and reports raw-vs-matched as an
ablation, so the effect of this step is measured rather than assumed.

```python
!cd {DBP} && python scripts/p1_match_generations.py \
    --gen-manifest reports/p1_gen_manifest.csv \
    --reals-manifest reports/p1_reals_manifest_t_real.csv
```

T-real, **not** T-ctrl: DIV2K is lossless PNG and has no JPEG distribution to match
against. The script refuses a reals manifest with no usable JPEG rows rather than
inventing one.

Read out of `reports/p1_gen_matched_manifest.json`:

- `pairing_exact` vs `pairing_sampled` — how many generations found the T-real row built
  from the same COCO image, and how many drew one by seed instead. Both are legitimate;
  the split is recorded so it can be reported.
- `quality_min` / `quality_max` / `subsampling_counts` — these must sit inside T-real's
  own observed range. The pair is drawn **jointly** from a single named real, so a
  combination that occurs in no real image cannot be produced.
- `resized: false` and `png_variant_kept: true` — both are invariants, not options.
- `failures` — must be empty.

The per-image `matched_quality`, `subsampling`, `seed`, `source_real_id`, `png_sha256`
and `jpg_sha256` are in `reports/p1_gen_matched_manifest.csv`, so the variant is
reproducible from the manifest alone.

DIV2K is the control that shows whether a model leans on JPEG structure at all: lossless
reals and their reconstructions, with no JPEG anywhere. Report it alongside.

## Step 6c — the greyscale re-scan, then rebuild the matched set (D-039)

Run this **after** Step 6b, and expect it to change the matched set. T-real contains 11
greyscale JPEGs; a greyscale JPEG has no chroma channels, so there is no
`(quality, subsampling)` pair to copy onto a colour generation, and 13 generations failed
on exactly that. Worse, a greyscale image in one class and not the other is a shortcut a
detector can learn instead of learning anything about decoders.

```python
!cd {DBP} && python scripts/p1_rescan_greyscale.py --set-id t_real --instances {INSTANCES}
```

It re-scans every stored crop, sets `excluded_reason=greyscale` on the affected rows, and
writes `reports/p1_greyscale_rescan_t_real.json`. **Nothing is deleted and nothing is
rebuilt**: the PNGs stay, the 49,200 B5 pair rows stay, and downstream manifests filter on
that column by joining on `image_id`. Drop `--instances` if the file is missing; the
per-supercategory table is then recorded as `NOT-RUN` rather than guessed.

Read out of it:

- `counts.single_channel_source` and `counts.rgb_equal` — the two kinds. The first is
  visible in the manifest (`subsampling == unknown(-1)`); the second is a greyscale
  picture inside a three-channel container, which only the pixels reveal.
- `rows_after` — the corpus size from now on. **There is no backfill**, so every later
  count must use this number.
- `supercategory_effect` — which COCO supercategories lose rows, and what share. If one
  loses noticeably more than the rest, say so wherever T-real is described.

Then rebuild the whole matched set from the colour-only reference. One seeded operation,
not a patch of the 13: removing rows from the reference changes which real every
generation draws, so the set is rebuilt in full.

```python
!cd {DBP} && python scripts/p1_match_generations.py \
    --gen-manifest reports/p1_gen_manifest.csv \
    --reals-manifest reports/p1_reals_manifest_t_real.csv
```

Expect **0 failures** and every eligible generation matched. In
`reports/p1_gen_matched_manifest.json` check:

- `failures: []` — the point of the exercise;
- `reference_excluded` — non-zero for `excluded_reason` and/or
  `unmappable_subsampling`, and logged;
- `pairing_counts` — a `sampled_after_exclusion` entry for each generation whose exact
  partner was excluded. Those generations are kept, with a seeded colour partner;
- `quality_min` / `quality_max` still inside T-real's observed 80–96.

---

## Step 7 — publish the outputs

Use **Save Version → output → dataset** (CLAUDE.md §6a condition (d)). The Kaggle API is
not used; if it ever is, its token comes from Kaggle Secrets only.

**Do not copy the images anywhere.** They already sit under `/kaggle/working`, so the
save picks them up as they are. An earlier version of this step copied them into
`/kaggle/working/p1_data`, which would duplicate ~14 GB and exceed Kaggle's 20 GB output
cap.

Take the inventory instead, and note the extensions: the matched variants are **JPEG**,
which is the whole point of that step. Counting `*.png` there reports 0 and looks like a
failure.

```python
import pathlib, shutil
for name, pattern in (("reals", "*.png"), ("pairs", "*.png"),
                      ("generated", "*.png"), ("gen_matched", "*.jpg")):
    d = pathlib.Path(DBP) / "outputs" / "p1" / name
    print(f"{name:12s}", sum(1 for _ in d.rglob(pattern)) if d.exists() else 0, pattern)
print("pairs manifest rows:",
      sum(1 for _ in open(pathlib.Path(DBP) / "reports" / "p1_pairs_manifest.csv")) - 1,
      "(expect reals x 12 distinct members)")
shutil.make_archive("/kaggle/working/p1_build_reports", "zip",
                    str(pathlib.Path(DBP) / "reports"))
!du -sh /kaggle/working
```

Download `p1_build_reports.zip` to `F:\thesis-2\kaggle_raw\` — **never** into
`REPO_ROOT`, which is a protected-set change and fails the integrity check (I-032).

The images stay on Kaggle as a dataset. **No image data is ever committed**
(CLAUDE.md §2); only the manifests and summaries are, under
`thesis_dbp/reports/kaggle_<date>/`.

---

## Checklist

- [ ] Both real corpora built; `kept` recorded for each, shortfalls explained.
- [ ] Compression table recorded for both, and copied into their data cards.
- [ ] Near-duplicate groups recorded.
- [ ] All distinct members complete in B5; `sd_ft_mse` absent from the pairs manifest.
- [ ] `p1_pairs_manifest.csv` row count = reals × distinct members.
- [ ] B6 safety-checker state and discarded count recorded.
- [ ] `p1_match_generations.py` run; `failures` empty, `resized: false`,
      `png_variant_kept: true`, and the matched qualities inside T-real's observed range.
- [ ] T-real built from `train2017` only — no `val2017` path in its manifest (I-024).
- [ ] T-ctrl: `crops_per_image: 2`, `split_grouping_column: source_image_id` recorded.
- [ ] `p1_rescan_greyscale.py` run; `rows_after` recorded as the corpus size, kinds and
      supercategory effect recorded, nothing deleted (D-039).
- [ ] Matched set rebuilt after the re-scan: `failures: []`, `pairing_counts` recorded.
- [ ] `gen_matched` counted as `*.jpg`, not `*.png`.
- [ ] Long stages run through **Save & Run All (Commit)**, not an interactive session
      (I-035): a Quick Save taken mid-run captures a partial tree, and anything run after
      it is lost when the session ends.
- [ ] Output dataset published through the notebook UI; reports zipped for
      `F:\thesis-2\kaggle_raw\`.
