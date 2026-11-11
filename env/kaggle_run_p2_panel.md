
# Kaggle run — P2-B1b: freeze the evaluation panel

**CPU only, internet on.** No accelerator quota is spent here, which matters because
this session re-downloads four evaluation corpora.

**Why this session exists.** The panel's *selection* was frozen in Session A; its
*pixels* were not. Each panel row recorded a path that only meant something in the
session that built its view — `/tmp/e7x/DDA-COCO/...` for E7, an absolute
`/kaggle/working/.../eval/e3_raw_jpeg_matched/...` for E3 — so **7,750 of the 12,150
rows were unreachable in a fresh session**, and the first thing to discover that would
have been a GPU run. This session copies the panel into one directory, verified against
the hashes the manifests already record, and publishes it as a dataset.

**Afterwards, every later session needs exactly two inputs:** the corpus and
`dbp-p2-panel`. No evaluation corpus is ever re-downloaded again.

> **The panel dataset is PRIVATE and stays private.** It contains WildRF
> (research-only, `NOASSERTION`) and Chameleon (*"Commercial use in any form is
> prohibited"*). CLAUDE.md §6c. Do not tick "Public" on the upload form.

**Inputs to attach:** the code dataset at the commit Step 0 prints, Session A's output
(for `reports/` and the rebuilt `gen_matched`), `awsaf49/coco-2017-dataset`,
`adeelshah29/p1-data`, the pairs notebook output, and your Chameleon dataset.

---

## Step 0 — stage, and confirm Session A's artefacts came with it

```python
import csv, json, pathlib
R = DBP / "reports"
print("commit :", (DBP / "COMMIT.txt").read_text().strip())

for name, want in (("p2_splits.csv", 55788),
                   ("p2_panel_e7_dda_coco.csv", 3500),
                   ("p2_panel_e8_evalgen.csv", 1250),
                   ("p2_panel_e3_matched.csv", 1500),
                   ("p2_panel_e4_wildrf_matched.csv", 1500),
                   ("p2_panel_e5_chameleon_pipeline.csv", 4000),
                   ("p2_panel_t_ctrl_diagnostic.csv", 400)):
    path = R / name
    n = sum(1 for _ in csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else 0
    print(f"{name:40s} {n:6d} (expect {want}){'' if n == want else '   <== WRONG'}")
    assert n == want, name
```

All seven must match. If a panel CSV is missing, Session A's output is not attached —
these are selections drawn under seed 2026 and they must not be re-drawn, because the
budget and the pre-registration describe *these* rows.

Re-stage the corpus root exactly as Session A did (`reals`, `generated` from `p1-data`;
`pairs` from the notebook output; `gen_matched` from Session A's `/kaggle/working/gm`).

---

## Step 1 — re-download the four evaluation corpora

Allow-listed hosts only (§6, §6a). Each goes to `/tmp`, never `/kaggle/working`, and
never to the local machine — §6a(a).

```python
import os, pathlib, subprocess
for key in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR"):
    os.environ[key] = f"/tmp/{key.lower()}"
    os.makedirs(os.environ[key], exist_ok=True)

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    print("Kaggle Secrets unavailable:", type(exc).__name__, exc)
print("HF_TOKEN present:", bool(os.environ.get("HF_TOKEN")))
```

Fetch each corpus with the **same route P1 used** — see `kaggle_run_p1_eval.md` Steps
2–6 for the exact snapshot ids, revisions and `gdown` ids. `gdown` is installed in the
session only, pinned, and never added to `requirements/` (§6a(c)):

```python
!python -m pip install --quiet "gdown==5.2.0"
```

| view | source | extract to | re-derive? |
|---|---|---|---|
| E7 DDA-COCO | HF `Junwei-Xi/DDA-COCO` @ the pinned revision | `/tmp/e7x` | **no** |
| E8 EvalGEN | HF `Junwei-Xi/EvalGEN` @ the pinned revision | `/tmp/e8x` | **no** |
| E3 | HF `zr-zhang/MLLM-Generated-Image-Detection-Dataset` | `/tmp/e3` | **yes**, 1,360 rows |
| E4 WildRF | Google Drive via `gdown` | `/tmp/wildrf/WildRF` | **yes**, 495 rows |
| E5 Chameleon pipeline | Session A's output, attached | — | no, if attached |

Which views need re-deriving was **measured**, not assumed: E7's and E8's panel rows
point at source files, so extracting the archive is enough. E3's matched view re-encoded
1,360 of its 2,178 rows and E4's re-encoded 495 of 2,503, and those images only ever
existed in the P1 eval session. E5's pipeline JPEGs exist in Session A's output and can
be **attached** instead of rebuilt — `resolve_source` finds them by basename under the
root you give for that view, which is the only correct resolution there, because an E5
pipeline row's `relative_path` still points at the untouched Chameleon source: a
different image whose hash would not match.

Copy the exact `repo_id`, `revision` and `gdown` FILE_ID from
`env/kaggle_run_p1_eval.md` Steps 2–6. They are recorded there and are not retyped here.

On a 401/403 from a gated repo, mark that view `BLOCKED-ACCESS` and stop — do not look
for a mirror (§6).

---

## Step 2 — re-derive the two matched views

E3's and E4's JPEG-matched views were derived in the P1 eval session and their images
are gone. The matcher is seeded by `blake2b(salt + ':' + image_id)`, so re-running it
reproduces the same bytes — and Step 3 proves that rather than assuming it.

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e3_raw --root /tmp/e3 \
    --real-glob "images/Raw/*/real/*" --fake-glob "images/Raw/*/*-fake/*" \
    --field subset --field source_label --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e4_wildrf_test --root /tmp/wildrf/WildRF \
    --real-glob "test/*/0_real/*" --fake-glob "test/*/1_fake/*" \
    --field platform --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

Copy the exact glob and `--field` arguments from `kaggle_run_p1_eval.md`; the ones above
are the shape, not a substitute for the recorded command.

E7 and E8 need **nothing** here: their panel rows resolve against the extracted
archive, so Step 1 is the whole job for them.

E5's pipeline-matched view is already in Session A's output. Point `--source` at it in
Step 3 and skip the 15-minute rebuild:

```python
import pathlib
E5P = next(p for p in pathlib.Path("/kaggle/input").rglob("e5_chameleon_pipeline_matched")
           if p.is_dir())
print(E5P, sum(1 for _ in E5P.glob("*.jpg")), "jpgs (expect 11170)")
```

If it is not attached, rebuild it as in `kaggle_run_p1_close.md` Step 4.

---

## Step 3 — materialise, with every file verified

```python
!cd {DBP} && python scripts/p2_materialise_panel.py \
    --panel-root /kaggle/working/panel \
    --source e7_dda_coco=/tmp/e7x \
    --source e8_evalgen=/tmp/e8x \
    --source e3_matched=/tmp/e3 \
    --source e4_wildrf_matched=/tmp/wildrf/WildRF \
    --source e5_chameleon_pipeline={E5P}     --source t_ctrl_diagnostic=/kaggle/working/p1 \
    --luma-subsets "e7_dda_coco:FLUX.1,real"
```

Every row is re-hashed and compared with the `sha256` its manifest records. **On any
mismatch the run stops, prints both hashes, and does not write the digest** — the
recorded hash is never overwritten. A re-derived matched view that differs means the
encoder changed, and I-010 measured it as identical across libjpeg versions, so a
mismatch is a finding, not a nuisance.

Expect:

```
panel: 12150 rows under /kaggle/working/panel
panel digest: <64 hex chars>
luma companion: 1000 rows (not counted in the panel)
```

`--source e7_dda_coco=/tmp/e7x`, not `/tmp/e7x/DDA-COCO`: `extract_members` works out
the archive's prefix and strips it, so the extracted tree starts at the panel's own
`relative_path`. The two archives disagree on this — DDA-COCO wraps its tree in a
directory the manifest does not record, while EvalGEN's top-level directory *is* part of
the recorded path — which is why the prefix is measured per archive instead of assumed.

**Write the digest down.** Session B asserts it before any GPU work starts.

**Then keep a hash-bearing copy of the panel list**, because the next script overwrites
that filename with a `root` column in place of `sha256`:

```python
import shutil
shutil.copy2(R / "p2_panel_images.csv", R / "p2_panel_images_frozen.csv")
```

The digest is defined over `image_id` + `sha256`, so the frozen copy is what every later
session verifies against, and `p2_panel_images.csv` is what resolves paths.

The luma companion is E7's FLUX.1 and real subsets as greyscale-in-RGB **PNG for both
classes** — lossless and identical either side, so the transform adds no
label-correlated difference, and taking luma discards the chroma planes where the
4:2:0-versus-4:4:4 shortcut lives. Pre-registration §8 requires FLUX.1 to be reported as
a pair, standard / luma-only, with the luma-only number entering M1 and M2.

---

## Step 4 — rebuild the image lists against the frozen panel

```python
!cd {DBP} && python scripts/p2_build_image_lists.py \
    --root /kaggle/working/p1 \
    --pairs-manifest reports/p1_pairs_manifest.csv \
    --panel-list reports/p2_panel_images.csv
```

With `--panel-list` the panel rows carry paths relative to the **panel** root and are
marked `root=panel`; corpus rows are marked `root=corpus`. That column is how the
extraction scripts resolve two directories from one list, and without it Session B reads
panel paths against the corpus root.

Expect `D-039: 89 flagged real(s) and their 1068 reconstruction(s) excluded`,
`p2_s1_images.csv 54631`, `p2_panel_images.csv 12150`, `p2_probe_images.csv ~17941`.

---

## Step 5 — persist, and publish the panel privately

```python
import csv, json, pathlib, shutil
R, PANEL = DBP / "reports", pathlib.Path("/kaggle/working/panel")

d = json.loads((R / "p2_panel_digest.json").read_text(encoding="utf-8"))
print("rows  :", d["rows"], "(expect 12150)")
print("digest:", d["panel_digest"])
assert d["rows"] == 12150

on_disk = sum(1 for p in PANEL.rglob("*") if p.is_file())
listed = sum(1 for _ in csv.DictReader((R / "p2_panel_images.csv").open(encoding="utf-8")))
luma = sum(1 for _ in csv.DictReader((R / "p2_panel_images_luma.csv").open(encoding="utf-8")))
print(f"files on disk {on_disk} | listed {listed} + luma {luma} = {listed + luma}")
assert on_disk == listed + luma, "the panel directory and its lists disagree"

for name in ("p2_s1_images.csv", "p2_probe_images.csv"):
    rows = list(csv.DictReader((R / name).open(encoding="utf-8")))
    roots = sorted({r["root"] for r in rows})
    print(f"{name:24s} {len(rows):6d} rows, roots {roots}")
    assert "root" in rows[0], f"{name} predates the root column -- re-run Step 4"

shutil.make_archive("/kaggle/working/p2_panel_reports", "zip", str(R))
!du -sh /kaggle/working/panel /kaggle/working
```

Then **Save Version with output saving on**, wait for it to finish, and create the
dataset from `/kaggle/working/panel`:

- slug **`dbp-p2-panel`**
- visibility **Private** — this is not a preference. WildRF is research-only and
  Chameleon prohibits commercial use in any form (§6c).

Download `p2_panel_reports.zip` to `F:\thesis-2\kaggle_raw\`, never into `REPO_ROOT`
(I-032).

---

## Checklist

- [ ] Step 0: all seven Session A artefacts present at their exact counts.
- [ ] Four evaluation corpora re-downloaded from their recorded sources; nothing mirrored.
- [ ] E3 and E4 matched views re-derived; E5 pipeline view present.
- [ ] `p2_materialise_panel.py` → **12,150 rows**, **0 hash mismatches**, digest printed
      and written down.
- [ ] Luma companion: **1,000 rows**, greyscale-in-RGB, not counted in the panel.
- [ ] Image lists rebuilt with `--panel-list`; every row carries a `root`.
- [ ] `dbp-p2-panel` created and **Private**.
- [ ] Reports zip downloaded outside `REPO_ROOT`.
