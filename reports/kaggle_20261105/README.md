# Kaggle results of 2026-11-05 (P4 steps 1 and 2)

Both JSON files are **transcribed from the cell output the user pasted** on 2026-11-05.
Each session printed its record with `json.dumps(..., indent=2)`, which is how the scripts
write the file, so the text should be byte-identical to the file on Kaggle. That equality
is **NOT-VERIFIED**: the original files were not hashed, and the session archives are not
in `F:\thesis-2\kaggle_raw\`.

| file | session | code | outcome |
|---|---|---|---|
| `p4_weights/p4_weights.json` | step 1, weights (`env/kaggle_run_p4_weights.py`), CPU | `bffdfca` | every file present and hash-checked; DDA strict load; CO-SPY artifact file holds the full VAE (248 keys), **equal to the pinned `sd15` VAE (max abs diff 0.0)**; FIRE holds the full VAE, equal to `sd15` cast to fp16 (0.0). Reported `ok: false` only because code `bffdfca` treated the CO-SPY VAE keys as a stop; the user ruled on 2026-11-05 that this is not a stop. |
| `p4_dda_equivalence.json` | step 2, the isolated DDA check (`env/kaggle_run_p4_dda_equivalence.py`), CPU, no Secrets, code-only bundle + public COCO | `bffdfca` | **PASS**: DDA's own code and `dbp.baselines.dda` give identical logits on both images (max abs diff 0.0, tolerance 1e-3). |

**Not recorded by step 2** (user rule of 2026-11-05: record the URLs and sha256 of
everything DDA's code downloads): the session ran code `bffdfca`, which did not log the
DDA archive's or `torch.hub`'s downloads. Their URLs and sha256 values are
**NOT-VERIFIED**. The check script now records them (`downloads` in its output); a re-run
of step 2 would fill them in.
