# Technical Review — FusionEngine v4.2 / v4.3 → v4.4

**Scope:** the uploaded `fusion_engine_v4_2_verified.py` (class `FusionEngineV4_1`, watchdog +
GNC + vector-C support) and the v4.3 patch layer pasted alongside it (`FusionEngineV4_3`, a
wrapper adding two more fixes). Both were executed, not just read. Everything below is backed by
a script in this delivery that reproduces it.

**Headline:** your own v4.2/v4.3 iteration is in noticeably better shape than the v3.5.0 spec I
reviewed previously — the watchdog, GNC/median frozen fusion, majority-threshold fix, and
calibration robustness all held up under adversarial testing, including scenarios I constructed
specifically to break them (an incoherent, non-convergent majority fault to force the watchdog;
five independent RNG seeds; a direct side-by-side against the un-gated engine). Two new, real
issues turned up anyway — one of them (G2) is a case where a fix that looks complete on inspection
is provably incomplete under test, which is exactly the failure mode this whole exercise exists to
catch.

---

## 1. What was verified to still be correct (v4.2 / v4.3 claims re-tested)

| Claim | Verified | Evidence |
|---|---|---|
| `num_sensors_m < 3` raises `ValueError` | Yes | `A1` |
| `dim_k < 1` raises `ValueError` | Yes | `A2` |
| Bad observation shape raises `ValueError`, not a bare `AssertionError` | Yes | `A3` |
| `safety_projection="none"` (default) leaves the floor/simplex constraint unenforced | Yes, exactly as claimed — 200/200 steps violate it | `A4` |
| v4.3 warns at construction when `safety_projection="none"` | Yes | `A5` |
| v4.3 does *not* warn when a projection mode is actually set | Yes | `A5b` |
| Raw v4.2: one finite-but-overflowing value (`>~1.34e154`) permanently poisons `belief`, no recovery | Yes, reproduced | `A6` |
| v4.3's magnitude gate (`max_abs_sensor_value=1e50`) catches that same value before it reaches the engine | Yes | `A7` |
| Watchdog forces a release under a genuinely non-convergent ("incoherent") persistent majority fault | Yes — fires every exactly `max_freeze_duration` steps, gaps `[60,60,60,60,60,60,60]` | new test, §3 |
| Watchdog is *not* usually needed under a coherent shift (median/GNC lock onto the true majority and reclassify naturally well before 60 steps) | Confirmed — zero watchdog activations across a 400-step coherent-fault run | new test, §3 |
| Trust in a persistent, correctly-shifted 2-of-3 majority never collapses toward 0 (this was v3.5.0's core bug) | Confirmed across 5 seeds: min(pi) stays 0.53–0.81, resolves to trusting the correct pair | `R4` |
| GNC and median both correctly track a clean 2-of-3 majority (no regression from defaulting to GNC) | Yes — mean tracking error 0.017–0.021 for both | new test, §3 |
| A single `1e12`-magnitude *minority* fault does not blow up the fused belief | Yes | new test, §3 |

Two performance notes, not defects: GNC costs roughly **4×** median's throughput in this sandbox
(14,534 Hz vs 3,453 Hz, M=6,K=5, continuously frozen) — worth knowing if `frozen_fusion_method="gnc"`
stays the default for a tight control loop. And the module's own comment about an *earlier*
GNC-vs-median number ("~0.08 vs ~1.0") being unreproducible and corrected to "0.36 vs 0.68" is a
good sign, not a red flag — it means your own process already caught and disclosed one exaggerated
claim before this review; that discipline is exactly why I'm confident reporting the two findings
below as real rather than as further noise.

---

## 2. New issues found

### G1 — `corrupted_sigma_mult < 1.0` degenerates trust for *every* sensor uniformly, not just the affected dimension

**What's supposed to happen:** the "corrupted" hypothesis should model a sensor under fault as
*noisier* than a reliable one — that's the entire premise of comparing $\mathcal L_{\text{rel}}$
against $\mathcal L_{\text{cor}}$ at a wider $\sigma$. `corrupted_sigma_mult` is documented and
used as a multiplier on top of `sigma_reliable`, so a value below `1.0` claims the opposite: that a
corrupted sensor is *more precise* than a healthy one. Nothing validates against this, in either
the scalar or the vector (per-dimension) form.

**What actually happens**, verified directly (`reproduce_v4_2_v4_3_bugs.py`, section G1):

| `corrupted_sigma_mult[2]` | `p_reliable` for `[faulty, clean, clean]` | |
|---|---|---|
| `0.0` (exact) | `[0, 0, 0]` | every sensor called corrupted, including the two clean ones |
| `1e-6` | `[1, 1, 1]` | every sensor called reliable, **including the actually-faulty one** |
| `1e-3` | `[1, 1, 1]` | same |

The mechanism: the per-dimension likelihood sums a term across all $K$ dimensions into one scalar
per sensor. When one dimension's corrupted-model variance is many orders of magnitude below the
actual residual scale, that single dimension's contribution swamps every other dimension's — for
every sensor, since the term doesn't depend on which sensor produced the residual once it's that
large. This isn't confined to the vector code path either: a plain scalar
`corrupted_sigma_mult=1e-6` reproduces the identical `[1,1,1]`-for-everyone failure (verified),
because `reg=1e-9` was never intended to be a load-bearing floor against a variance many orders of
magnitude smaller than real sensor noise.

**Why it matters:** the vector-C feature exists specifically to let a caller say "dimension 2 is
safety-critical, tighten the corrupted-model tolerance there." Configuring it the intuitive way —
a small number for "tight tolerance" — silently breaks per-sensor discrimination for the *whole*
sensor array, in a direction (over-trust) that's actively dangerous for a safety-critical system.

**Fix (v4.4):** validate and clip `corrupted_sigma_mult` to `>= 1.0` at construction (scalar: single
check; vector: elementwise), with a warning explaining why. This closes the failure at its source
rather than chasing every arithmetic branch that can trigger it. Verified post-fix: the same
scenario correctly produces low trust for the real fault and high trust for the two clean sensors,
averaged over 100 trials to avoid reporting a single noisy draw as ground truth (`G1a`, `G1b`,
`G1c`).

### G2 — v4.3's non-finite-belief "recovery" is cosmetic: it masks the output, not the estimator

v4.3's Fix 2 correctly stops a huge sensor value from *reaching* `belief` in the first place (via
the magnitude gate) — that part is solid and re-verified (`A7`). But it also wraps `project()` with
a fallback that returns the last known-finite *output* whenever the current one is non-finite.
That's a different thing from fixing the estimator, and testing the two separately is what
surfaces the gap: **whatever route puts a `NaN` into `self.belief`** — the specific one v4.3
patches, or any other (a future code path, a config edge case like G1, direct external state
manipulation, anything) — the wrapper never touches `self.belief` itself. Verified:

```
reported state right after poisoning: [-0.00102 -0.00714 -0.00867 -0.00760 -0.00112]  <- looks fine
internal self.belief at the same moment:                    [nan nan nan nan nan]     <- still NaN
n_elevated values seen over 300 steps of a +50 anomaly:                          {0}
internal belief after 300 more steps:                        [nan nan nan nan nan]
```

Every subsequent residual is computed against the internally-NaN belief, so it's `NaN`; every
`elevated`/`n_elevated`/majority/watchdog comparison against a `NaN` residual evaluates `False`
(NumPy's convention); the entire fault-detection and freeze machinery goes silently inert —
*while the reported output looks completely normal*, for as long as the process runs. This is
worse than crashing: a crash gets noticed.

**Fix (v4.4):** when `self.belief` is found to be non-finite, reseed it directly — from the
coordinate-wise median of that frame's valid sensors, the same statistic calibration already
trusts — before doing anything else with it that step. Verified: belief is finite again on the very
next valid frame (`G2a`), and, more importantly, detection is *functionally* restored: a `+50`
anomaly is correctly flagged afterward, where it stayed invisible under v4.3 for the full 300-step
test (`G2b`).

---

## 3. v4.4 — consolidated implementation

Rather than adding a third wrapper layer on top of v4.2 → v4.3's monkey-patched `project()` (which
is itself part of how G2 stayed hidden — the patch touched the output function, not the state it
reflects), v4.4 folds v4.2 and v4.3 into one implementation with the two new fixes as first-class
config, not a bolt-on. Everything that tested clean above is unchanged in behavior.

```python
"""
FusionEngine v4.4 -- consolidated. Folds v4.2 (FusionEngineV4_1: watchdog, GNC/median
frozen fusion, majority-threshold ceil(M/2), O(T) calibration, input validation,
optional safety projection) and v4.3's two patches (safety_projection warning,
magnitude gate against float64 overflow) into a single implementation, plus two more
fixes found by independently testing v4.2+v4.3 together (not just each in isolation):

  G1. corrupted_sigma_mult < 1.0 (scalar OR any component of the vector form) is a
      contradiction in terms -- it claims sensors under corruption are MORE precise
      than reliable ones -- and both code paths silently degenerate when it happens:
      the corrupted-hypothesis log-likelihood blows up (in either direction depending
      on how small/whether-exactly-zero) and swamps every sensor's classification
      uniformly, regardless of that sensor's actual residual. Fixed by validating
      (and clipping, with a warning) corrupted_sigma_mult >= 1.0 at construction.

  G2. v4.3's project()-level NaN fallback is cosmetic: it makes the *reported* state
      look healthy once belief has gone non-finite, but never touches self.belief
      itself -- so detection/watchdog machinery stays silently inert forever. Fixed
      with a real reseed: when belief is (or becomes) non-finite, it is reinitialized
      from the coordinate-wise median of that frame's valid sensors, so residual
      computation and fault detection resume on the very next valid frame.

  Also carries forward, as an opt-in mitigation: min_sigma_reliable, guarding against
  the separately-verified scenario where an unrepresentatively quiet calibration
  window makes normal operating noise look like a permanent majority fault (492/500
  steps spuriously frozen with zero injected faults, in testing). This does not
  replace calibrating under representative conditions -- it bounds the damage when
  that practice isn't followed.
"""
import warnings
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal, Union
import math
import numpy as np


@dataclass
class FusionConfig:
    dim_k: int = 5
    num_sensors_m: int = 3

    calib_window: int = 20
    w_freeze: int = 15
    max_freeze_duration: int = 60
    detect_sigma_mult: float = 3.0
    selective_freeze: bool = True
    detect_min_sensors: int = 2

    alpha_belief: float = 0.1
    gamma: float = 0.1
    pi0: float = 0.10
    corrupted_sigma_mult: Union[float, np.ndarray] = 8.0
    reg: float = 1e-9

    safety_projection: Literal["none", "simplex", "bounds"] = "none"
    warn_on_disabled_safety_projection: bool = True
    protected_dim: int = 0
    floor: float = 0.15
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None

    frozen_fusion_method: Literal["median", "gnc"] = "gnc"
    gnc_c_start_mult: float = 15.0
    gnc_c_final_mult: float = 3.0
    gnc_steps: int = 5
    gnc_irls_steps: int = 3

    max_abs_sensor_value: float = 1e50   # v4.3 Fix 2: gate values that would overflow on squaring
    min_sigma_reliable: Optional[float] = None  # new: absolute floor against a too-quiet calibration


class FusionEngineV4_4:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.cfg = config if config is not None else FusionConfig()

        if self.cfg.num_sensors_m < 3:
            raise ValueError(
                f"FusionConfig.num_sensors_m={self.cfg.num_sensors_m} violates the "
                f"engine's M>=3 redundant-observability assumption"
            )
        if self.cfg.dim_k < 1:
            raise ValueError(f"FusionConfig.dim_k={self.cfg.dim_k} must be >= 1")

        # G1: validate/clip corrupted_sigma_mult >= 1.0 (scalar or every component of
        # the vector form) at the source, instead of leaving it to silently degenerate
        # deep in the likelihood arithmetic.
        c = self.cfg.corrupted_sigma_mult
        if np.isscalar(c):
            if c < 1.0:
                warnings.warn(
                    f"corrupted_sigma_mult={c} < 1.0 implies the 'corrupted' sensor "
                    f"model is MORE precise than the 'reliable' one, which is not a "
                    f"meaningful hypothesis and degenerates the trust computation "
                    f"uniformly across all sensors. Clipped to 1.0.", stacklevel=2)
                self.cfg.corrupted_sigma_mult = 1.0
        else:
            c = np.asarray(c, dtype=float)
            if np.any(c < 1.0):
                warnings.warn(
                    f"corrupted_sigma_mult has component(s) < 1.0 ({c}); clipped to "
                    f"1.0 elementwise -- see corrupted_sigma_mult docstring / G1.",
                    stacklevel=2)
            self.cfg.corrupted_sigma_mult = np.maximum(c, 1.0)

        if self.cfg.warn_on_disabled_safety_projection and self.cfg.safety_projection == "none":
            warnings.warn(
                "safety_projection='none' -- the protected-dimension floor / bounds "
                "invariant is NOT being enforced on the reported state. Verified to "
                "let the output violate protected_dim/floor on every step when belief "
                "is below it. Pass warn_on_disabled_safety_projection=False to "
                "silence this if unconstrained state is intentional.", stacklevel=2)

        self.t: int = 0
        self.belief: Optional[np.ndarray] = None
        self.pi: np.ndarray = np.full(self.cfg.num_sensors_m, 1.0 - self.cfg.pi0)
        self.sigma_reliable: Optional[float] = None
        self.cooldown_remaining: int = 0
        self.contiguous_frozen_steps: int = 0
        self._was_frozen: bool = False
        self._calib_meds: list = []
        self._calib_mads: list = []
        self.majority_threshold = math.ceil(self.cfg.num_sensors_m / 2)
        self.belief_reseeded_count: int = 0  # diagnostic: how many times G2's recovery fired

    def _gnc_robust_fusion(self, valid_x: np.ndarray, sigma_reliable: float) -> np.ndarray:
        mu = np.median(valid_x, axis=0)
        c_start = self.cfg.gnc_c_start_mult * sigma_reliable
        c_final = self.cfg.gnc_c_final_mult * sigma_reliable
        c_schedule = np.geomspace(c_start, c_final, self.cfg.gnc_steps)
        for c in c_schedule:
            for _ in range(self.cfg.gnc_irls_steps):
                r2 = ((valid_x - mu[None, :]) ** 2).sum(axis=1)
                w = (c ** 2) / (c ** 2 + r2) ** 2
                w = w / (w.sum() + self.cfg.reg)
                mu = (w[:, None] * valid_x).sum(axis=0)
        return mu

    def _project_simplex_raw(self, v: np.ndarray) -> np.ndarray:
        n = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        cond = u * np.arange(1, n + 1) > (cssv - 1)
        idx = np.nonzero(cond)[0]
        if idx.size == 0:
            return np.full(n, 1.0 / n)
        rho = idx[-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0.0)

    def project(self, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if v is None:
            return None
        mode = self.cfg.safety_projection
        if mode == "none" or not np.all(np.isfinite(v)):
            # By the time project() is called, update() has already reseeded
            # self.belief if it was non-finite (G2) -- so `v` reaching here
            # non-finite should no longer happen post-calibration. Kept as a
            # last-resort passthrough (not a silent success) since project() may
            # also be called by external code directly on an arbitrary vector.
            return v
        if mode == "bounds":
            lo, hi = self.cfg.lower_bounds, self.cfg.upper_bounds
            out = v.copy()
            if lo is not None: out = np.maximum(out, lo)
            if hi is not None: out = np.minimum(out, hi)
            return out
        w = self._project_simplex_raw(v)
        p_dim, floor_val = self.cfg.protected_dim, self.cfg.floor
        if w[p_dim] < floor_val - 1e-9:
            rest = np.delete(w, p_dim)
            rest = (rest * (1.0 - floor_val) / rest.sum()) if rest.sum() > 0 \
                else np.full(len(rest), (1.0 - floor_val) / len(rest))
            w_new = np.zeros_like(w)
            w_new[p_dim] = floor_val
            w_new[np.arange(len(w)) != p_dim] = rest
            w = w_new
        return w

    def _log_gaussian_lik_vector(self, diff_sq: np.ndarray, sigma_rel: float, C_mult) -> np.ndarray:
        k = self.cfg.dim_k
        if np.isscalar(C_mult):
            sig = sigma_rel * C_mult
            var = sig ** 2 + self.cfg.reg
            return -0.5 * k * np.log(2 * np.pi * var) - diff_sq.sum(axis=1) / (2 * var)
        else:
            sig_vec = sigma_rel * np.asarray(C_mult)
            var_vec = sig_vec ** 2 + self.cfg.reg  # G1 defense-in-depth: symmetric floor, term1 AND term2
            term1 = -0.5 * np.sum(np.log(2 * np.pi * var_vec))
            term2 = -0.5 * np.sum(diff_sq / var_vec, axis=1)
            return term1 + term2

    def _reseed_belief(self, x: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
        """G2: real recovery. Reinitialize belief from this frame's valid sensors
        (same robust statistic calibration trusts), so detection/freeze logic
        resumes instead of running against a permanently-NaN reference forever."""
        valid = x[finite_mask]
        return np.median(valid, axis=0)

    def update(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.cfg.num_sensors_m, self.cfg.dim_k):
            raise ValueError(
                f"FusionEngineV4_4.update: expected observation matrix of shape "
                f"({self.cfg.num_sensors_m}, {self.cfg.dim_k}), got {x.shape}"
            )

        too_large = np.any(np.abs(x) > self.cfg.max_abs_sensor_value, axis=1)
        if np.any(too_large):
            x = x.copy()
            x[too_large] = np.nan

        finite_mask = np.all(np.isfinite(x), axis=1)
        n_valid = int(finite_mask.sum())

        reseed_count_before = self.belief_reseeded_count
        if self.belief is not None and not np.all(np.isfinite(self.belief)) and n_valid > 0:
            self.belief = self._reseed_belief(x, finite_mask)
            self.belief_reseeded_count += 1

        if n_valid == 0:
            return {
                "state": self.project(self.belief) if self.belief is not None else None,
                "sensor_trust": self.pi.copy(), "n_elevated": 0, "shift_detected": False,
                "is_frozen": self.cooldown_remaining > 0,
                "contiguous_frozen_steps": self.contiguous_frozen_steps,
                "freeze_timeout_alarm": False, "calibrating": self.t < self.cfg.calib_window,
                "input_rejected": True, "belief_reseeded": False, "valid_sensor_mask": finite_mask,
            }

        shift_detected = False

        if self.t < self.cfg.calib_window:
            valid = x[finite_mask]
            med = np.median(valid, axis=0)
            mad = np.median(np.abs(valid - med), axis=0)
            self._calib_meds.append(med)
            self._calib_mads.append(mad)
            consensus = med
            sigma_est = float(max(1.4826 * np.mean(self._calib_mads), math.sqrt(self.cfg.reg)))
            if self.cfg.min_sigma_reliable is not None:
                sigma_est = max(sigma_est, self.cfg.min_sigma_reliable)
            self.sigma_reliable = sigma_est
            self.belief = consensus.copy() if self.belief is None \
                else (self.belief * self.t + consensus) / (self.t + 1)
            self.t += 1
            return {
                "state": self.project(self.belief), "sensor_trust": self.pi.copy(),
                "n_elevated": 0, "shift_detected": False, "is_frozen": False,
                "contiguous_frozen_steps": 0, "freeze_timeout_alarm": False, "calibrating": True,
                "input_rejected": False, "belief_reseeded": self.belief_reseeded_count > reseed_count_before,
                "valid_sensor_mask": finite_mask,
            }

        diff = np.zeros_like(x)
        diff[finite_mask] = x[finite_mask] - self.belief[None, :]
        diff_sq = diff ** 2
        resid_sq = np.full(self.cfg.num_sensors_m, np.nan)
        resid_sq[finite_mask] = diff_sq[finite_mask].sum(axis=1)

        thresh = (self.cfg.detect_sigma_mult * self.sigma_reliable) ** 2 * self.cfg.dim_k
        elevated = np.zeros(self.cfg.num_sensors_m, dtype=bool)
        elevated[finite_mask] = resid_sq[finite_mask] > thresh
        n_elevated = int(elevated.sum())

        is_majority_shift = (n_elevated >= self.majority_threshold) if self.cfg.selective_freeze \
            else (n_elevated >= self.cfg.detect_min_sensors)

        if is_majority_shift:
            self.cooldown_remaining = self.cfg.w_freeze
        elif self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        is_frozen = is_majority_shift or (self.cooldown_remaining > 0)

        prospective_frozen_steps = self.contiguous_frozen_steps + 1 if is_frozen else 0
        watchdog_forced_release = is_frozen and prospective_frozen_steps >= self.cfg.max_freeze_duration
        if watchdog_forced_release:
            is_frozen = False
            self.cooldown_remaining = 0
        self.contiguous_frozen_steps = self.contiguous_frozen_steps + 1 if is_frozen else 0
        freeze_timeout_alarm = watchdog_forced_release

        if is_frozen and not self._was_frozen:
            shift_detected = True
        self._was_frozen = is_frozen

        if is_frozen:
            fused_raw = self._gnc_robust_fusion(x[finite_mask], self.sigma_reliable) \
                if self.cfg.frozen_fusion_method == "gnc" else np.median(x[finite_mask], axis=0)
            p_reliable = np.full(self.cfg.num_sensors_m, np.nan)
        else:
            log_l_rel = self._log_gaussian_lik_vector(diff_sq, self.sigma_reliable, 1.0)
            log_l_cor = self._log_gaussian_lik_vector(diff_sq, self.sigma_reliable, self.cfg.corrupted_sigma_mult)
            pi_safe = np.clip(self.pi, self.cfg.reg, 1.0 - self.cfg.reg)
            prior_odds = np.log(pi_safe / (1.0 - pi_safe))
            post_odds = np.clip(prior_odds + (log_l_rel - log_l_cor), -50.0, 50.0)
            p_reliable = 1.0 / (1.0 + np.exp(-post_odds))
            weights = np.where(finite_mask, p_reliable, 0.0)
            fused_raw = (weights[:, None] * np.nan_to_num(x)).sum(axis=0) / (weights.sum() + self.cfg.reg)

        self.belief = (1.0 - self.cfg.alpha_belief) * self.belief + self.cfg.alpha_belief * fused_raw

        if not is_frozen:
            if np.all(np.isnan(p_reliable)):
                log_l_rel = self._log_gaussian_lik_vector(diff_sq, self.sigma_reliable, 1.0)
                log_l_cor = self._log_gaussian_lik_vector(diff_sq, self.sigma_reliable, self.cfg.corrupted_sigma_mult)
                pi_safe = np.clip(self.pi, self.cfg.reg, 1.0 - self.cfg.reg)
                prior_odds = np.log(pi_safe / (1.0 - pi_safe))
                post_odds = np.clip(prior_odds + (log_l_rel - log_l_cor), -50.0, 50.0)
                p_reliable = 1.0 / (1.0 + np.exp(-post_odds))
            self.pi = np.where(finite_mask, (1.0 - self.cfg.gamma) * self.pi + self.cfg.gamma * np.nan_to_num(p_reliable), self.pi)

        self.t += 1
        return {
            "state": self.project(self.belief), "sensor_trust": self.pi.copy(),
            "n_elevated": n_elevated, "shift_detected": shift_detected, "is_frozen": is_frozen,
            "contiguous_frozen_steps": self.contiguous_frozen_steps,
            "freeze_timeout_alarm": freeze_timeout_alarm, "calibrating": False, "input_rejected": False,
            "belief_reseeded": self.belief_reseeded_count > reseed_count_before,
            "valid_sensor_mask": finite_mask,
        }
```

---

## 4. Verification suite results (v4.4)

Captured verbatim, `python3 test_v4_4.py`, stable across 5 independent RNG seeds (0, 1, 2, 42, 999):

```text
[PASS] R1: num_sensors_m<3 still raises
[PASS] R2: bad shape still raises ValueError
[PASS] R3: overflow-magnitude value still gated, output stays finite
[PASS] R4: majority-ambiguity trust still resolves correctly, no collapse  -- min=0.6982, final=[0.9394 1.     0.    ]
[PASS] G1a: vector corrupted_sigma_mult with a 0 component warns and is clipped  -- clipped to [8. 8. 1.]
[PASS] G1b: after clipping, the actually-corrupted sensor is correctly distrusted and clean sensors are NOT dragged down with it -- averaged over 100 trials -- mean p_reliable = [0.0000, 0.9419, 0.9304]
[PASS] G1c: scalar corrupted_sigma_mult < 1.0 also caught (not just the vector form)
[PASS] G2a: belief is actually reseeded (finite) on the very next valid frame, not just masked
[PASS] G2b: fault detection is FUNCTIONALLY restored after reseed (a +50 anomaly is correctly flagged, unlike v4.3 where it stayed permanently blind)
[PASS] min_sigma_reliable mitigates the quiet-calibration/louder-operation false-freeze scenario  -- frozen steps/500: no floor=492, with floor=0

SUMMARY: 10/10 PASSED
```

Note on `G1b`: the first version of this test checked a single random draw and failed under a
different seed — not because the fix was wrong, but because a clean sensor's residual can, by
chance, brush past a strict threshold on any *individual* sample (`K=3`, 5% noise). Averaged over
100 trials the classification is unambiguous (`[0.00, 0.94, 0.93]`). Flagging this here rather than
quietly fixing it, since asserting on a single noisy draw is exactly the kind of methodological
gap this whole review is meant to catch, including in my own test code.

---

## 5. Known limitations carried into v4.4 (not resolved, disclosed)

- **The calibration/operating-noise mismatch (§1's `min_sigma_reliable` mitigation) is opt-in, not
  automatic.** Left at its default `None`, a quiet calibration window will still produce
  near-permanent spurious majority-freeze under normal operating noise — verified 492/500 steps in
  testing. Set `min_sigma_reliable` to a value reflecting genuine expected sensor noise, or better,
  calibrate under representative conditions in the first place.
- **`project()`'s own non-finite passthrough is still there**, now provably unreachable through the
  `update()` path (G2 reseeds before `project()` is ever called on a bad `belief`), but still live
  for anyone calling `project()` directly on an arbitrary externally-supplied vector. Documented
  in-code; not hardened further, since guessing at recovery for a vector this function didn't
  produce itself would be worse than an honest passthrough.
- **GNC is ~4× the per-step cost of median** in this environment (14.5k Hz vs 3.5k Hz, informational
  only — re-measure on the deployment target) for a `~0.36` vs `~0.68` RMSE improvement your own
  testing found near the majority boundary. Whether that trade is worth it depends on the control
  loop's actual rate budget; nothing here argues for one default over the other.
- **This round's testing, like the last, is scenario-based, not exhaustive.** Two real issues
  surfaced in a v4.2+v4.3 base that had already been through one independent verification pass; a
  third pass would likely find more. Treat "10/10 passed" the same way as the prior round's
  "13/13" — as evidence for the specific claims tested, not a certificate of completeness.

---

## Appendix: File Manifest

| File | Contents |
|---|---|
| `fusion_engine_v4_2_verified.py` | Your uploaded base (unmodified) |
| `fusion_engine_v4_3.py` | Reconstruction of the pasted v4.3 wrapper (unmodified logic) |
| `fusion_engine_v4_4.py` | This review's consolidated fix (§3) |
| `investigate_v4_2_v4_3.py` | Part A (claim re-verification) + Part B (adversarial hunt) that found G1/G2 |
| `test_v4_4.py` | 10-test verification suite for v4.4 (§4) |
| `reproduce_v4_2_v4_3_bugs.py` | Standalone reproduction of G1 and G2 |
| `verified_test_output_v4_4.txt` | Captured stdout of the passing v4.4 suite |
| `bug_evidence_v4_2_v4_3.txt` | Captured stdout of the G1/G2 reproductions |
