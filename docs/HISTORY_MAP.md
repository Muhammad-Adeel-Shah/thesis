# History map — where each cited commit lives

Written 2026-11-15 (user request, review of `2d56cbd`). Every value below comes from a
command run that day; the command is given beside it.

## 1. Three eras of history

| Era | Commits | Where it lives now |
|---|---|---|
| **A. Before 2026-11-02** | the original development history (183 commits; the log names its tags `p3-prereg-frozen` and `p4-baselines-frozen`) | **Nowhere in git.** Removed with the old `.git` when the repository was re-rooted at `thesis_dbp/` (implementation log, 2026-11-05 entry and the snapshot table). The code of the runs survives as Kaggle code zips (section 4). |
| **B. Pre-release** | 15 commits, `db2aca9` "initial release" (2026-11-02) to `d5e56dc` (2026-11-11) | Local branch `archive/pre-release` and the bundle (section 3). |
| **C. Release** | `17e9c6e` "Initial release: image module" and its successors on `master` | The private GitHub repository, branch `master`, tags `v1.0-images` and `p3b-vitl-prereg-frozen`. |

## 2. The release root

`17e9c6e832b32f7905e92be08ca6d518c8b857b8` is a root commit, with no parent. Its tree is the tree of
`d5e56dc4e7cdb5f51f2301492f98dd58012c3285` plus 43 lines appended to
`docs/IMPLEMENTATION_LOG.md`, which record the re-root itself. No other file differs.
Source: `git diff --stat d5e56dc 17e9c6e` → `1 file changed, 43 insertions(+)`.

So any file at `d5e56dc`, other than the log, is byte-identical at `17e9c6e`.

## 3. The pre-release history (era B)

- **Branch `archive/pre-release`:** tip `d5e56dc4e7cdb5f51f2301492f98dd58012c3285`, 15 commits
  (`git rev-list --count archive/pre-release`).
- **Bundle:** `F:/thesis-2/git_archive/thesis_dbp_history_d5e56dc.bundle`.
  - 25,592,718 bytes;
  - sha256 `a8db6c3159b8766f8fac1539d2218677560dbe6fa5bf9a76c276b2f1d36f7b07` (`sha256sum`);
  - `git bundle verify`: "The bundle records a complete history".
  - Its refs (`git bundle list-heads`):
    - `refs/heads/master` → `d5e56dc`;
    - `refs/remotes/origin/master` → `47551d4`;
    - `refs/tags/p3b-vitl-prereg-frozen`: annotated tag object `43ad831c…`, pointing at
      commit `6b6f7d248079c38ca836641057f1da0578ca66bc`.
- **Name collision.** The *old* `p3b-vitl-prereg-frozen` (on `6b6f7d2`, era B) and the
  *release* `p3b-vitl-prereg-frozen` (on `17e9c6e`, era C) share a name. The release tag is
  the one in the repository. Wherever the old one is published, it is named
  `archive/p3b-vitl-prereg-frozen`.
- **To restore era B anywhere:**
  `git fetch <bundle> 'refs/heads/master:refs/heads/archive/pre-release'`.
- **Published to GitHub:** NOT YET. Pending a user decision. 14 of the 15 commits carry a
  co-author trailer, which the release history was created to leave out.

## 4. Era A: the Kaggle code zips

Kaggle runs take their code from a zip built from git's bytes (`scripts/build_code_zip.py`),
stamped with `thesis_dbp/COMMIT.txt` (short and full hash).
- 32 such zips are kept in `F:/thesis-2/kaggle_raw/` as `dbp-code-<hash>.zip`, and every
  `COMMIT.txt` matches its file name.
- 5 of those commits exist in git today: `2d56cbd`, `34a399a`, `8800801`, `9ba3dde` and
  `d5e56dc`.
- The other 27 are era A: `042f6d9`, `0af33cf`, `19f6d5d`, `1bc18e2`, `218da57`, `26775f6`,
  `28df98e`, `29e66f5`, `2dcec9a`, `2f84225`, `33f43d8`, `408c1f5`, `4dfe54f`, `6782139`,
  `692ab48`, `69cc811`, `8171516`, `860521c`, `8618a24`, `9a70325`, `a26b9e8`, `a7f4256`,
  `a8e50c0`, `bffdfca`, `c9b8c0b`, `e2561f9`, `fcc2ab4`.
- For an era-A run, its zip is the surviving record of the exact code it ran.

## 5. For the paper's reproducibility appendix

> Commit hashes cited in this thesis and its reports refer to the project's development
> history. Hashes from 2 November 2026 onward resolve in the archived pre-release history
> (`d5e56dc` and earlier, to `db2aca9`), preserved as a git bundle (sha256
> `a8db6c3159b8766f8fac1539d2218677560dbe6fa5bf9a76c276b2f1d36f7b07`), or in the release
> history (`17e9c6e` onward). The release root `17e9c6e` holds the same files as `d5e56dc`,
> apart from the implementation log. Hashes before 2 November 2026 identify code that no
> longer exists as git history. For the Kaggle runs, the exact code is preserved as the
> uploaded code archive, stamped with that commit hash.

Any public statement about these archives must respect the dataset licences in CLAUDE.md
section 6c. The repository and every archive stay private.
