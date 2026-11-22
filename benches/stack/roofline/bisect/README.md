# Bisect of the single-GPU slowdown, 7631ceb → 4485588

**Question.** The retimed table (`../SUMMARY.md`) is 5–90% slower than the old
one, which was measured at 7631ceb. Which commit made it slower?

**Answer.** No commit slows the pairwise (`k = 1`) runs. Every commit, 7631ceb
included, runs at the same speed in today's environment. Most of the `k = 1`
slowdown at small `chi` comes from the environment: every `src` call runs
`shutil.disk_usage(tempfile.gettempdir())` to size its disk tier, and our jobs
put `TMPDIR` on Lustre, where that `statvfs` takes 25 ms in the median and up
to 290 ms. The only code change above 15% is **d2287df** (opt_einsum `dp` paths
by default), which slows `k = 4` by 18% at `chi = 512` and by about 45% at
`chi = 64`.

## Method

The worktrees live outside the repository, in `../bisect-wt/<commit>` next to the clone
(`git worktree add --detach`, plus an untracked `_version.py` stub, since
`hatch-vcs` only writes it at install time). Each commit ran its **own**
`benches/stack/bench_depth.py` with `PYTHONPATH=<worktree>/src` and the
repository's venv (the package is `src_method` up to 4a2d6a3, `portc` at
4485588):

```bash
python benches/stack/bench_depth.py evolve --n-sites 50 --chis 64 --chis 512 \
  --steps-per-sweep 1 --steps-per-sweep 4 --reference-chi 64 --quimb-max-chi 0
```

Every script had all these flags, so none was substituted. Each commit ran
twice in a row and the second run is kept (`<commit>.md`). One A100 of an
exclusive Booster node, debug QoS, `TMPDIR=$CINECA_SCRATCH/tmp` as in the
Leonardo docs. Infidelities are against a `chi = 64` reference, so they are not
comparable with the paper table. They are identical across commits.

Environment: the venv of `../env.txt` after the GPU extra was restored (CuPy
14.2.0 with the CUDA wheels of `uv.lock`, quimb 1.16.0, autoray 0.11.0).

## Wall time of the 80 steps (s)

| commit | date | change | χ=64 k=1 | χ=64 k=4 | χ=512 k=1 | χ=512 k=4 |
|---|---|---|---|---|---|---|
| old table | | | 6.87 | 2.89 | 33.9 | 19.1 |
| 7631ceb | 10-13 | refresh Leonardo benchmarks (old table) | 10.7 | 3.48 | 41.5 | 20.8 |
| fb62638 | 10-14 | split the SRC sweep among GPUs | 11.2 | 3.95 | 38.5 | 20.2 |
| c91f31f | 10-14 | split a batch that runs out of memory | 10.5 (10.7) | 4.01 (3.94) | 39.7 (37.6) | 20.1 (20.3) |
| a996df3 | 10-15 | array API for backend dispatch | 10.4 (*16.8*) | 4.11 (3.97) | 37.6 (42.0) | 20.4 (*24.5*) |
| bd78e23 | 10-15 | oversample sketches | *20.0* | 4.79 | *45.1* | 22.3 |
| d2287df | 10-27 | **opt_einsum `dp` paths by default** | 11.1 | **5.05** | 41.6 | **24.6** |
| 3f7114a | 10-29 | non-blocking stream per rank | 11.2 | 6.21 | 37.3 | 24.5 |
| 4a2d6a3 | 11-07 | device SVDs, distributed CholeskyQR | 11.1 | 5.01 | 37.7 | 24.6 |
| 4485588 | 11-20 | `main` | *18.8* | 5.87 | *48.6* | 25.9 |

c91f31f and a996df3 were timed again in a later job (65160760). The table shows
that run, with the first in brackets (`c91f31f_first.md`, `a996df3_first.md`).
*Italic* cells are outliers that disagree with their neighbours and, for
a996df3, with its own repeat. They come with Lustre stalls: the same job's
`chi = 64` reference once took 202 s instead of 11 s, and the warm-ups vary from
14 to 80 s. Treat single values within about ±15% as noise. These runs are
Lustre-sensitive, which is the point below.

**First commit slower than 7631ceb by more than 15%:**

- `k = 1`: none. χ=64 stays at 10.4–11.2 s and χ=512 at 37–42 s whenever a run
  is not stalled.
- `k = 4, chi = 512`: **d2287df**, 24.6 s against 20.8 (+18%), and every later
  commit keeps it (24.5–25.9 s). The 24.5 s at a996df3 did not survive its
  repeat (20.4 s), and bd78e23 is +7%.
- `k = 4, chi = 64`: bd78e23 is the first above +15% (4.79, +38%), but its run
  is one of the stalled ones (its `k = 1` cells are the two outliers). From
  d2287df on it is consistently 5.0–6.2 s (+45–78%). Within the noise, the step
  is at bd78e23 or d2287df. The `dp` result at `chi = 512` and the `auto`
  rerun in `../SUMMARY.md` point to d2287df.

## What the profiles show

`cprofile_<commit>.txt`: n = 50, `chi = 64`, `k = 1`, 80 steps under cProfile,
after one unprofiled warm-up, at c91f31f, a996df3 and 4485588 (top 25 by
cumulative time). The three are alike. The kernels and opt_einsum take 10.2–10.8
s, and **`posix.statvfs` takes 9.9–11.7 s** (80 calls, 124–146 ms each), called
from `resolve_budgets` → `shutil.disk_usage(tempfile.gettempdir())`. No commit
changes that call. It has been there since 7631ceb.

`tmpdir_check.txt` isolates it on one node. `statvfs` of
`$CINECA_SCRATCH/tmp` takes 25 ms in the median and 288 ms at most, and of
`/tmp` a few µs. `main`'s `chi = 64, k = 1` run takes 8.2–8.4 s with
`TMPDIR=/tmp` against 10.8–11.0 s with Lustre (−24%), and its reference takes
8.5 s against 10.9 s. `k = 4` makes 4× fewer calls and changes less (4.3–4.6
against 4.8 s).

## Conclusions

1. **The `k = 1` slowdown is environmental.** With `TMPDIR` on Lustre, every
   `src` call pays a Lustre `statvfs`, about 30 ms per sweep on a quiet file
   system and over 100 ms on a busy one. That accounts for 2.6 s (quiet) to 10 s
   (busy) of the 80-sweep runs. It inflates `k = 1` most, because it is a
   per-sweep fixed cost, so it also inflates the measured fusion speedups. The
   rest of the gap to the old table (8.3 s on `/tmp` against 6.87 s at `chi =
   64`; 37–42 s against 33.9 s at `chi = 512`) is the same for every commit, so
   it comes from the software stack or the nodes, not from `portc`. The old
   stack (its CuPy and CUDA wheels) was not rebuilt to separate those.
2. **The `k = 4` slowdown is d2287df**, the switch to `dp` paths, as the `auto`
   rerun of `../SUMMARY.md` found from the other side.
3. **For the paper table**, rerun with `TMPDIR` on node-local `/tmp`, or pass
   `Resources(scratch_dir=...)` on a local disk. The Leonardo docs page
   recommends Lustre for `TMPDIR`, which is right for large spills but costs
   every small `src` call a metadata round-trip. A library-side fix is to query
   the disk only when the plan needs the disk tier. That is outside this
   branch.

## Files

| file | content |
|---|---|
| `<commit>.md` | the kept (second) run of each commit |
| `c91f31f_first.md`, `a996df3_first.md` | first timed runs of the two re-timed commits |
| `cprofile_<commit>.txt` | cProfile, top 25 by cumulative time |
| `tmpdir_check.txt` | `statvfs` and run time with `TMPDIR` on `/tmp` and on Lustre |
| `job.sh`, `profile_k1.py` | the bisect job and the cProfile driver |
| `logs/` | Slurm logs: 65159229, 65159230 (bisect), 65160760 (re-time), 65160617 (cProfile) |
