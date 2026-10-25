# Internal working notes

The working files of the competition (30 July – 5 August 2026), kept as they were written, in
Spanish. [`PLAN.md`](PLAN.md) is the solution plan and how it evolved;
[`PROJECT_STATE.md`](PROJECT_STATE.md) is the state at the close (final public/private table
and the v1 → v8 progression, local CV against the public leaderboard);
[`DECISIONS.md`](DECISIONS.md) records 15 numbered decisions (`ROG-001` to `ROG-015`), with the
measured negatives given as much space as the wins. The [root README](../../README.md) is the
summary; these files are the evidence behind it.

**The most relevant entries.** `ROG-001` is the reformulation everything rests on: TVT equals a
geological surface evaluated along the trajectory plus a per-well constant (residual 0.0065 ft,
the data's own rounding). `ROG-004`, the anisotropic metric for the surface, took it from 24.2
to 12.9 ft and is the own contribution with the largest impact. `ROG-006` replaces a fixed blend
with per-point inverse-variance fusion (9.978 → 9.145 ft on 150 wells), with most of the gain in
the most isolated third of wells, which is the regime of the hidden test. `ROG-007` closes, by
three independent routes, the idea that the pre-PS prefix predicts post-PS behaviour.
`ROG-009` and `ROG-011` set the anti-noise rules: an improvement must hold on two validation
subsets, and the final choice is measured on all 770 wells, because a 150-well subset picked the
wrong model. `ROG-014` recalibrates the constants inherited from the public particle filter,
the best single finding of the project (public score 9.072 → 8.893). `ROG-015` reads the
shakeup: the simplest model (v8) degraded least from public to private (+0.09), and every
added layer bought public score that the private leaderboard took back.

**Reading notes.** `ROG-013` records that, on the last day, a fork of a public pipeline built
on third-party artifacts was submitted in parallel to my own models. It is not my work and is
labelled as such in the README ("the public route"); it ended below every model of my own on
the private leaderboard. `ROG-010` (v7 as main submission) was superseded the next day by v11 (`ROG-014`),
the model selected for the final ranking. These files give the rank at the
close (1,082nd of 6,191, +2,299 places from 3,381st on the public board); the README uses the
later figure from the Kaggle profile (1,075th of 6,125, after withdrawn teams).
`PROJECT_STATE.md` still names `kernel/rogii_v8.py` as the submission file; the repository
ships the later `kernel/rogii_v11.py`.
