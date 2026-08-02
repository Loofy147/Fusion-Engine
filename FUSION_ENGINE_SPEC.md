# FusionEngine v4.2 — Technical Specification

**Reference implementation:** `fusion_engine_v4_2_verified.py`
**Status:** written from the verified implementation and this session's own
test results. The source comments in the uploaded v4.1 file reference an
external "architecture spec" (Section 5.1, Section 5.2, Stage 6, Design Goal
1, Design Goal 3) that was never provided in this conversation and could not
be independently checked. This document does not assume that spec is
accurate and does not reuse its section numbers — it specifies what was
actually read, run, and verified. If a prior spec document exists, treat
this as something to reconcile against it, not a transcription of it.

---

## 1. Purpose

Track a K-dimensional state from M >= 3 sensor streams that (a) disagree
with each other some of the time, (b) may include streams that are
corrupted, biased, or actively adversarial, and (c) are observing a target
that can shift abruptly and without warning — while never letting the
output state violate one hard safety constraint, regardless of how the
sensors or the target behave.

## 2. System model and assumptions

- **State**: `belief in R^K`, reported each step as `state = project(belief)`.
- **Sensors**: `M >= 3` streams per step, each either NaN (dropped, e.g.
  sensor timeout) or a K-vector reading. `M < 3` is rejected at
  construction — below that, "most sensors agree" isn't a meaningful
  concept (`M=1` has no majority to check against).
- **Corruption model**: each sensor, each step, is drawn from one of two
  latent components — *reliable* (small, symmetric noise around the true
  state) or *corrupted* (something else: an outlier, a bias, an adversarial
  value). Which component fired on a given step is not observed directly.
- **Regime shift**: the true state can jump discontinuously and without
  any external signal marking when.
- **Invariant**: one designated dimension (`protected_dim`) must never fall
  below `floor` in the reported `state`. This is the one property the
  engine is not permitted to trade away for any other objective.

## 3. Design goals, in priority order

1. **The invariant holds unconditionally.** Not "usually" — every step,
   regardless of sensor corruption, regime shifts, or internal engine
   state. This is a hard constraint, not an optimization target.
2. **A single corrupted stream cannot dominate fusion**, no matter how
   extreme its reported value. Confidence in any one stream is bounded.
3. **Recovery from a real shift should be fast**, without needing to know
   in advance when a shift will happen.
4. **Freezing in response to an apparent shift must be temporary and
   bounded.** A mechanism meant to protect against a transient must not
   itself become a permanent failure mode.
5. **A transient reference mismatch must not be misread as a sensor
   going bad.** (This is the specific failure mode this engine's
   two-track fusion strategy exists to avoid — see Section 6.3.)

Goals are ordered because they trade off: goal 4's fix (a periodic forced
release) costs a small amount of goal 2/5 protection during that single
release step. That trade is deliberate — see Section 9.2.

## 4. Architecture overview

```
per step:
  read x[M,K], mask out non-finite rows
  no valid rows?  ->  reject step, report last known state, do nothing else

  still calibrating (t < calib_window)?
    -> per-sensor median/MAD folded into a running consensus estimate
       (belief seeded from consensus, no trust learned yet)

  otherwise:
    -> compute each valid sensor's residual against current belief
    -> count sensors "elevated" (residual far outside calibrated noise)
    -> elevated count >= majority threshold?  -> start/extend freeze window
    -> freeze window active?
         fuse:   robust cohort estimate on raw x (median, or GNC/IRLS)
         trust:  frozen (not updated this step)
       else:
         fuse:   Bayesian-posterior-weighted average of x
         trust:  updated from this step's posterior (EMA)
    -> belief <- EMA(belief, fused)
    -> state  <- project(belief)   [invariant enforced here, unconditionally]
```

## 5. Formal definitions

**5.1 Trust update (unfrozen path).** For sensor m with residual
`r_m = ||x_m - belief||^2` and calibrated reliable noise `sigma`:

```
log_lik_reliable(m)  = -K/2 * log(2*pi*sigma^2)              - r_m / (2*sigma^2)
log_lik_corrupted(m) = -K/2 * log(2*pi*(sigma*C)^2)          - r_m / (2*(sigma*C)^2)
posterior_odds(m)    = log(pi_m / (1 - pi_m)) + [log_lik_reliable(m) - log_lik_corrupted(m)]
p_reliable(m)        = sigmoid(clip(posterior_odds(m), -50, 50))
```
`pi_m` (the corruption-rate prior) updates via `pi_m <- (1-gamma)*pi_m + gamma*p_reliable(m)`
only outside a freeze window. `C` = `corrupted_sigma_mult`, a fixed generic
multiple, not a per-sensor learned or oracle-supplied value.

**5.2 Fused estimate (unfrozen path):**
`fused = sum_m(p_reliable(m) * x_m) / sum_m(p_reliable(m))`, `reg`-guarded
against a zero denominator.

**5.3 Fused estimate (frozen path).** Either `median(x_valid, axis=0)`, or
`GNC(x_valid)`: IRLS on the Geman-McClure loss
`w_i = c^2 / (c^2 + r_i^2)^2` under an annealed scale `c`, geometrically
stepped from `gnc_c_start_mult * sigma` down to `gnc_c_final_mult * sigma`
over `gnc_steps` outer iterations, `gnc_irls_steps` inner reweightings each
(Black & Rangarajan duality; Yang, Antonante, Tzoumas & Carlone, *Graduated
Non-Convexity for Robust Spatial Perception*, RA-L 2020).

**5.4 Shift detection.** `elevated(m) = 1[r_m > (detect_sigma_mult * sigma)^2 * K]`.
`is_majority_shift = sum(elevated) >= ceil(M/2)` (when `selective_freeze`
is on; a fixed `detect_min_sensors` otherwise). On trigger,
`cooldown = w_freeze`, decrementing by 1 each subsequent non-triggering
step; `is_frozen = is_majority_shift OR cooldown > 0`.

**5.5 Watchdog.** `contiguous_frozen_steps` tracks consecutive frozen
steps. If continuing would reach `max_freeze_duration`, force
`is_frozen = False` and `cooldown = 0` for exactly that one step
(`freeze_timeout_alarm = True`), then resume normal logic — re-freezing
immediately next step if the anomaly persists.

**5.6 Projection (invariant enforcement).** Euclidean projection onto the
probability simplex, then: if `w[protected_dim] < floor`, set it to
`floor` and rescale the remaining mass proportionally to sum to
`1 - floor`. Applied to every reported `state`, every step, regardless of
path — this is what makes goal 1 (Section 3) structural rather than
aspirational.

## 6. Verified behavior (this session's tests)

**6.1 End-to-end performance.** Against the same 150-seed x 4-stress-cell
harness used throughout this benchmark project (regime shift + competing
sensors, correlated-crisis and adversarial-camouflage variants): 100%
recovery within the horizon in every cell, both fusion modes, restricted
mean recovery 22-30 steps, vs. 61-128 steps and 40-80% recovery for every
policy built earlier in this project. Full numbers in
`v4_1_vs_v3_comparison.json`.

**6.2 Recovery speed is explained, not just observed.** Instrumented run:
the engine is frozen only ~19% of a typical episode, concentrated in a
~25-step window around the actual shift (not a standing bias toward robust
fusion). Measured recovery time (~22 steps) matches the belief EMA's own
convergence floor under `alpha_belief=0.1` (~20 steps to reach 5% of a
typical shift gap) — i.e., once fusion quality stops being corrupted by a
stale reference, recovery is close to the fastest this belief-smoothing
rate can produce at all.

**6.3 Why the frozen path uses raw fusion instead of belief-relative
weighting.** This is the actual fix for the failure mode documented
earlier in this project: a clean sensor's residual against a *stale*
belief looks identical to that sensor becoming unreliable, and a trust
estimate that updates on the same timescale as the belief it's built from
can compound that misread into a feedback loop (verified in an earlier
session's diagnostic: 14 steps to recover trust in a sensor that was never
actually unreliable). Fusing on raw `x` during the freeze window has no
dependency on the belief at all, so it cannot inherit that failure mode —
confirmed directly by the timing in 6.2.

**6.4 Watchdog liveness.** Fed a permanent, never-resolving 3-way sensor
disagreement for 200 steps. Longest contiguous frozen run: 59, against a
configured cap of 60 — released 4 times, never stuck. (`edge_case_tests.py`)

**6.5 Extreme-value robustness.** A single sensor reporting `1e12` while
the other two stay consistent: fused state unaffected (0.253 vs. 0.252
pre-spoof), that sensor's trust dropped but wasn't zeroed in one step,
and — correctly — one bad sensor alone did not trigger the freeze
(`n_elevated=1`, below the majority threshold of 2). (`edge_case_tests.py`)

**6.6 M-generalized majority vote.** Re-tested at M=5 (not just the M=3
case exercised above): threshold correctly computed as `ceil(5/2)=3`,
correctly ignored 2/5 elevated, correctly froze at 3/5.

**6.7 GNC vs. median under independent corruption.** M=7, majority
threshold 4, 500 trials/point, shipped GNC defaults: both estimators track
equally well below the threshold; at n_corrupt=4/7, RMSE 0.36 (median) vs.
0.68 — direction confirmed, GNC meaningfully better past the threshold.
(`verify_gnc_claim.py`)

## 7. Configuration reference

| field | default | basis |
|---|---|---|
| `calib_window` | 20 | as shipped, not independently re-derived here |
| `w_freeze` | 15 | as shipped; independently checked against `alpha_belief`'s own convergence timescale (~22 steps to 90%) and found consistent, not derived from it originally |
| `max_freeze_duration` | 60 | as shipped; verified to actually bound freeze duration (6.4) |
| `detect_sigma_mult` | 3.0 | as shipped, not independently re-derived here |
| `detect_min_sensors` / `selective_freeze` | 2 / True | at M=3 these are equivalent to the majority rule; the majority rule is the one verified to generalize (6.6) |
| `alpha_belief` | 0.1 | as shipped; this is the parameter that sets the hard floor on recovery speed (6.2) |
| `gamma` | 0.1 | as shipped, not independently re-derived here |
| `pi0` | 0.10 | as shipped, generic prior — not sensor-specific, not oracle-derived |
| `corrupted_sigma_mult` | 8.0 | as shipped, not independently re-derived here |
| `frozen_fusion_method` | `"gnc"` | verified better than `"median"` past the majority threshold (6.7); `"median"` kept available as the simpler, more auditable option |

Fields marked "as shipped, not independently re-derived here" are not
unverified in the sense of being untested — the engine was tested at these
values (Section 6) — but their specific numeric choice wasn't re-optimized
or re-derived in this audit, only inherited and exercised.

## 8. Guarantees and verification status

| guarantee | mechanism | status |
|---|---|---|
| Invariant never violated | unconditional projection on every reported state | structural (holds by construction, not by policy behavior) + empirically confirmed, 0 violations across every test in this project |
| No single corrupted stream dominates fusion | `post_odds` clipped to [-50,50] bounds `p_reliable` away from exactly 0; weight is a probability, not the raw value | verified directly (6.5) |
| Freeze is temporary | watchdog forces release before `max_freeze_duration` | verified directly (6.4) |
| Belief-staleness isn't misread as sensor failure during a shift | raw-value fusion during freeze has no belief dependency | verified via mechanism timing, not just outcome (6.2, 6.3) |
| One bad sensor alone can't trigger freeze | majority threshold requires >= ceil(M/2) simultaneously elevated | verified directly (6.5) |
| Fusion generalizes beyond M=3 | majority threshold computed from `M`, not hardcoded | verified at M=5 (6.6) |
| GNC beats median near/past the majority threshold | redescending IRLS weight vs. a hard binary count | verified, but original claimed margin (~0.08 vs ~1.0) was not reproducible and has been corrected to the measured 0.36 vs. 0.68 (6.7) |

## 9. Known limitations and explicit non-guarantees

**9.1 Majority collusion defeats both fusion modes.** If >= majority
threshold sensors agree on the *same* false value (not independent
outliers), median and GNC fail together — tested at ~2.5 RMSE for both, no
graceful-degradation gap between them. Independent corruption (this
engine's assumption, Section 2) and colluding corruption are different
threat models; only the former is what Section 6 verifies against. Nothing
in this design detects or defends against the latter.

**9.2 The watchdog trades a bounded amount of protection for guaranteed
liveness.** Every `max_freeze_duration` steps under a persistent anomaly,
exactly one Bayesian-weighted (not robust) update is let through. This is
a deliberate choice (Section 3, goal 4 over goal 2/5 for one step at a
time), not an oversight — but it means "permanently frozen" is not
literally true under sustained attack, which matters if a threat model
assumes it.

**9.3 Corruption is assumed independent across sensors except when
deliberately modeled otherwise.** The "correlated crisis" stress test in
Section 6.1 raises multiple sensors' corruption *probability* together,
which the engine handles fine — that is a different, weaker condition
than sensors colluding on an identical *value* (9.1).

**9.4 No claim is made here about the un-seen external spec.** Where the
original source comments assert something about "Section 5.1," "Section
5.2," or named Design Goals, this document neither confirms nor
reproduces those claims — only what was independently readable from the
code and checked by running it.

## 10. Provenance

v4.1 -> v4.2: one change, Section 6.7's corrected GNC margin, replacing an
uncheckable claim with a measured one. No other implementation changes —
everything else in v4.1 tested clean.
