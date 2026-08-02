# Technical Specification & Reference Architecture

## Multi-Sensor State Fusion Engine
### Majority-Rule Bayesian Filtering with Level-Triggered Freeze and Robust Cohort Fusion

**Document Version:** 4.0.0
**Status:** Independently re-derived, tested, and corrected from v3.5.0. Every claim in this
document was produced by executing the accompanying code in this repository — nothing here is
asserted without a script that reproduces it (see `verified_test_output.txt`,
`v35_bug_evidence.txt`).
**Classification:** Technical Specification

---

## 0. Relationship to v3.5.0 — What Changed and Why

v3.5.0 was labeled "Fully Verified & Benchmarked Reference Manual." Re-implementing it verbatim
and testing it against its own stated goals (not just its own six curated tests) surfaced four
substantive defects, one severe enough to crash the estimator in production. This document is not
a light edit — it corrects the freeze state machine, the fusion rule used while frozen, the
majority-boundary definition, calibration, and input handling, and it separates the actually
generic parts of the design from a projection step that only applies to a narrower class of
problems than the document claimed to cover. Every fix below is backed by a before/after number,
not just an assertion.

| # | Defect in v3.5.0 | Verified impact | Fix in v4.0.0 |
|---|---|---|---|
| F1 | "Non-retriggering" freeze guard is actually an edge-triggered **pulse**: 15 steps frozen, 1 step open, repeating, for any *persistent* majority-classified condition | Correct sensors' trust erodes **0.9 → 0.0 by step ~3000**; the one broken sensor rises to 1.0 — the opposite of "majority rule" | Level-triggered hold: freeze persists continuously while the majority condition is active, plus a `W_FREEZE`-step cooldown after it clears |
| F2 | Belief fusion recomputes per-sensor weights from instantaneous residuals **every step, even while "frozen"** — the state estimate itself was never protected, only the long-run trust score was (weakly) | Belief drifts onto whichever sensor is momentarily closer, independent of the freeze flag | While frozen, fuse via coordinate-wise **median** across sensors (robust cohort statistic) instead of likelihood-weighted mean |
| F3 | Majority boundary in code (`n_elevated > M//2`) silently disagrees with the identifiability table (Sec. 6, v3.5), which claims the exact 50% tie (M=6, c=3) is a majority | Filter **never freezes** at M=6,c=3 despite the table saying it should | Majority threshold redefined as `n_elevated >= ceil(M/2)`; ties now count as majority (conservative default for a safety-critical system) |
| F4 | No input validation anywhere. A single `NaN` in one sensor reading crashes `_project_simplex` (`IndexError`) and, absent the crash, permanently poisons `belief` with `NaN` forever | **Unhandled crash** reproduced in one line; confirmed in `v35_bug_evidence.txt` | Non-finite readings are excluded from that frame's fusion and flagged; an all-`NaN` frame is rejected and the previous belief is held. Never raises. |
| F5 | Calibration uses plain cross-sensor mean/variance, with no robustness to a sensor that is already faulty during the calibration window itself | One offset sensor during a 20-step calibration inflates `sigma_reliable` by **43.7×** (0.043 → 1.865), which silently desensitizes fault detection for the rest of deployment | Calibration consensus and noise estimate use per-step **median** and **MAD** (1.4826×MAD), which only inflate ~1.3× under the same scenario |
| F6 | The mandatory simplex + protected-floor projection forces every output state onto `{w : sum(w)=1, w≥0, w[d]≥floor}`, applied unconditionally to a state framed as generic robotics/aerospace/power-grid telemetry (positions, velocities, voltages are not simplex-valued) | Passing a real physical state through this constraint silently and drastically corrupts it | Projection is now **opt-in** (`safety_projection: "none" \| "bounds" \| "simplex"`); simplex mode retained for genuinely compositional/allocation states, `"bounds"` added as a generic elementwise floor/ceiling for physical states |
| — | `shift_detected` re-fires on every pulse of the old freeze cycle | **13 spurious "new shift" events over 200 steps** for one ongoing incident | True edge-trigger on the False→True freeze transition; **2 events** over the same 200 steps (small residual re-triggers remain possible near the classification boundary — see §7) |

Everything in v3.5.0 that actually held up under testing is kept: the minority-fault decay rate
(~0.9 → 0.0002 in ~80 steps), convergence under large uniform shifts, numerical stability under
extreme residuals, and the core "distrust minority disagreement, don't destroy trust over the
whole array" philosophy. The comparison against an unprotected baseline (`~0.11-0.13` trust
collapse) also checks out empirically (§6.4).

---

## 1. Problem Formulation (unchanged from v3.5.0, still accurate)

Given $M \ge 3$ streaming, potentially noisy sensor observations $x_{m,t} \in \mathbb{R}^K$, track
an underlying state $\mu_t \in \mathbb{R}^K$ while maintaining a per-sensor trust score
$\pi_{m,t} \in [0,1]$, subject to two failure modes:

1. **Transient-induced trust collapse** — an unannounced regime shift makes every sensor look
   simultaneously "wrong" relative to a stale belief; a naive adaptive filter misreads this as
   simultaneous sensor corruption and drives every $\pi_m \to 0$.
2. **Persistent correlated minority deadlock** — a genuinely faulty minority of sensors should be
   discounted, not protected, and a naive global freeze that re-arms on every frame prevents that
   discounting from ever completing.

v3.5.0 additionally claimed to solve a **third**, unstated problem, which is really the
interesting one and the one v3.5.0 got wrong: distinguishing a genuine 2-of-3-type majority
disagreement (ambiguous — could be 2 sensors correctly tracking a real shift, or 2 sensors that
failed together) from a minority fault, and *not silently resolving that ambiguity in the wrong
direction* while it's still open. §6.2 below shows this is where the real engineering difficulty
lives, not in the two problems v3.5.0's introduction focuses on.

---

## 2. Architecture

```
                  ┌───────────────────────────────────────────┐
                  │   Streaming Input Matrix: X_t [M x K]      │
                  └────────────────────┬────────────────────────┘
                                       │
                                       ▼
                  ┌───────────────────────────────────────────┐
                  │ 0. INPUT VALIDATION  (new: F4)             │
                  │    finite_mask = isfinite(x) per sensor    │
                  │    all-invalid -> hold belief, reject frame│
                  └────────────────────┬────────────────────────┘
                                       │
                                       ▼
                  ┌───────────────────────────────────────────┐
                  │ 1. RESIDUAL & THRESHOLD (valid sensors only)│
                  │    r_m^2 = ||x_m - belief||^2               │
                  │    tau = (D*sigma_reliable)^2 * K           │
                  │    n_elevated = count(r_m^2 > tau)          │
                  └────────────────────┬────────────────────────┘
                                       │
                                       ▼
                  ┌───────────────────────────────────────────┐
                  │ 2. MAJORITY CONDITION (fixed: F3)           │
                  │    is_majority = n_elevated >= ceil(M/2)    │
                  └───────────┬─────────────────┬───────────────┘
                              │                 │
                        YES (Majority)    NO (Minority / none)
                              │                 │
                              ▼                 ▼
             ┌─────────────────────────┐  ┌───────────────────────┐
             │ LEVEL-TRIGGERED HOLD     │  │ cooldown ticks down    │
             │ (fixed: F1)              │  │ if still >0; else      │
             │ cooldown = W_FREEZE      │  │ fully unfrozen         │
             │ every step condition     │  │                        │
             │ holds -- no leak         │  │                        │
             └────────────┬─────────────┘  └───────────┬────────────┘
                          │                             │
                          └─────────────┬───────────────┘
                                       │
                                       ▼
                  ┌───────────────────────────────────────────┐
                  │ 3. FUSION                                   │
                  │    IF frozen:  fused = median(x, valid)     │
                  │       (fixed: F2 -- robust cohort tracking) │
                  │    ELSE: Bayesian trust-weighted mean        │
                  │    belief <- EMA(belief, fused)             │
                  └────────────────────┬────────────────────────┘
                                       │
                                       ▼
                  ┌───────────────────────────────────────────┐
                  │ 4. CONDITIONAL TRUST UPDATE                 │
                  │    frozen -> pi unchanged                   │
                  │    else   -> pi <- EMA(pi, p_reliable)      │
                  └────────────────────┬────────────────────────┘
                                       │
                                       ▼
                  ┌───────────────────────────────────────────┐
                  │ 5. OPTIONAL SAFETY PROJECTION (fixed: F6)   │
                  │    "none" (default) / "bounds" / "simplex"  │
                  └────────────────────┬────────────────────────┘
                                       │
                                       ▼
                  ┌───────────────────────────────────────────┐
                  │ OUTPUT: state, sensor_trust, flags          │
                  └───────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation

### 3.1 Calibration (fixed: F5 — robust to a bad sensor during calibration)

For $t < T_{\text{calib}}$, at each step compute the **coordinate-wise median** across sensors
(not the mean) as that step's consensus, and the **median absolute deviation** as the noise
estimate:

$$\tilde{x}_{t,k} = \operatorname{median}_m(x_{m,t,k}), \qquad
\text{MAD}_{t,k} = \operatorname{median}_m\left(|x_{m,t,k} - \tilde{x}_{t,k}|\right)$$

$$\sigma_{\text{reliable}} = \max\left(1.4826 \cdot \overline{\text{MAD}}, \sqrt{\epsilon_{\text{reg}}}\right)$$

The median/MAD pair has a breakdown point of 50% (for the MAD; the coordinate-wise median
consensus tolerates up to a minority of corrupted sensors per step), versus 0% for a plain
mean/variance estimator — a single bad sensor during calibration can inflate the naive estimate
without bound (verified: 43.7× in a modest one-sensor, +5.0-offset scenario; see §6.3).

### 3.2 Residual Evaluation & Majority Condition (fixed: F3)

Unchanged: $r_{m,t}^2 = \|x_{m,t} - \hat\mu_t\|_2^2$, elevated if $r_{m,t}^2 > \tau$ where
$\tau = (D_{\text{mult}}\sigma_{\text{reliable}})^2 K$. The majority condition is now defined
without the odd-vs-even split that produced the v3.5.0 table/code mismatch:

$$\mathcal{M}_t = \mathbb{I}\left(n_{\text{elevated}} \ge \left\lceil M/2 \right\rceil\right)$$

For odd $M$ this is identical to v3.5.0. For even $M$, an exact 50% split (e.g. $c=3$ of $M=6$)
now counts as majority — deliberately the conservative choice: an ambiguous 50/50 disagreement
should not be silently resolved by discounting half the array.

### 3.3 Level-Triggered Freeze with Hysteresis (fixed: F1)

$$\text{cooldown}_t = \begin{cases}
W_{\text{FREEZE}}, & \mathcal{M}_t = \text{True} \quad \text{(reset/hold every step condition is active)} \\
\max(\text{cooldown}_{t-1} - 1,\ 0), & \mathcal{M}_t = \text{False}
\end{cases}$$

$$I_{\text{frozen},t} = \mathbb{I}(\mathcal{M}_t) \lor \mathbb{I}(\text{cooldown}_t > 0)$$

Unlike v3.5.0's counter (which only ever gets *set* to $W_{\text{FREEZE}}$ once, on the frame the
counter last hit zero, and then only decrements — producing a forced one-frame gap every
$W_{\text{FREEZE}}+1$ steps), this formulation holds $I_{\text{frozen}}=\text{True}$
continuously for as long as $\mathcal{M}_t$ is true, with no periodic gap, and only starts
counting down once the condition genuinely clears. `shift_detected` is emitted on the
$\text{False}\to\text{True}$ transition of $I_{\text{frozen}}$, not on every reset of the counter.

### 3.4 Fusion Rule (fixed: F2)

$$\hat x_{\text{fused},t} = \begin{cases}
\operatorname{median}_{m \in \text{valid}}(x_{m,t}), & I_{\text{frozen},t} = \text{True} \\[4pt]
\dfrac{\sum_m p_{m,t}\, x_{m,t}}{\sum_m p_{m,t} + \epsilon}, & I_{\text{frozen},t} = \text{False}
\end{cases}$$

with $p_{m,t}$ the same clipped-log-odds Bayesian posterior as v3.5.0 (§3.5 below) used only in
the unfrozen branch. This is the core correction: v3.5.0 froze $\pi_{m,t}$ (the *slow* trust
prior) but kept recomputing $p_{m,t}$ (the *instantaneous* per-step reliability) from raw
residuals regardless of freeze status, so the belief itself was never actually protected during
the ambiguous window — only the reported trust number was, weakly. Using the median while frozen
means a genuine 2-of-3-type majority is *tracked*, not discounted, during exactly the window where
that distinction matters.

### 3.5 Bayesian Mixture (unchanged, unfrozen branch only)

$$\ln\mathcal L_{\text{rel}} = -\tfrac{K}{2}\ln(2\pi\sigma_r^2) - \tfrac{r^2}{2\sigma_r^2}, \quad
\ln\mathcal L_{\text{cor}} = -\tfrac{K}{2}\ln(2\pi(C\sigma_r)^2) - \tfrac{r^2}{2(C\sigma_r)^2}$$

$$p_{m,t} = \sigma\!\left(\operatorname{clip}\!\left(\ln\frac{\pi_{m,t}+\epsilon}{1-\pi_{m,t}+\epsilon} + \ln\mathcal L_{\text{rel}} - \ln\mathcal L_{\text{cor}},\ -50,\ 50\right)\right)$$

$$\pi_{m,t+1} = \begin{cases}\pi_{m,t}, & I_{\text{frozen},t}\\ (1-\gamma)\pi_{m,t}+\gamma p_{m,t}, & \text{else}\end{cases}$$

### 3.6 Input Validity (new: F4)

Let $v_{m,t} = \mathbb{I}(x_{m,t}$ is finite in every coordinate$)$. Sensors with $v_{m,t}=0$ are
excluded from $n_{\text{elevated}}$, from the fusion sum/median, and from the $\pi$ update for
that frame only. If $\sum_m v_{m,t} = 0$, the frame is rejected: `belief` is held at its previous
value, `pi` is unchanged, and `input_rejected=True` is returned. No code path raises on
non-finite input.

### 3.7 Safety Projection (now optional: F6)

$$a_{t+1} = \begin{cases}
\hat\mu_{t+1}, & \text{mode = "none" (default)} \\
\operatorname{clip}(\hat\mu_{t+1},\ \ell,\ u), & \text{mode = "bounds"} \\
\mathcal P_{\mathcal C}(\hat\mu_{t+1}), \ \mathcal C = \{w : \textstyle\sum w_k=1,\ w\ge0,\ w_{d}\ge F\}, & \text{mode = "simplex"}
\end{cases}$$

`"simplex"` is v3.5.0's behavior, preserved for genuinely compositional states (portfolio
weights, categorical/regime probabilities). It is not applied by default because it does not
generalize to the physical state spaces (position, velocity, voltage) implied by the "robotics /
aerospace / power grid" framing in §1 — see §6.5.

---

## 4. Production Python Source Code

Full source: `fusion_v4.py` (below, unabridged). Config renamed `FusionConfig`
(from `ImprovedFusionConfig`) — see §8 for migration notes.

```python
"""
ImprovedFusionFilter v4.0.0
Fixes applied after empirically testing v3.5.0 against its own claims:
  F1. Level-triggered freeze with hysteresis (was: leaky edge-triggered pulse).
  F2. Robust median cohort fusion while frozen (was: belief silently unprotected).
  F3. Majority threshold = ceil(M/2), ties count as majority (was: table/code mismatch).
  F4. Non-finite input validated and never crashes or permanently poisons belief.
  F5. Median/MAD calibration, robust to a bad sensor during the calib window itself.
  F6. Safety projection is opt-in ("none" | "bounds" | "simplex"), not mandatory.
  F7. shift_detected is a true edge-trigger, not a repeating pulse.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal
import math
import numpy as np


@dataclass
class FusionConfig:
    dim_k: int = 5
    num_sensors_m: int = 3

    calib_window: int = 20
    w_freeze: int = 15                 # hysteresis cooldown after majority condition clears
    detect_sigma_mult: float = 3.0
    selective_freeze: bool = True
    detect_min_sensors: int = 2        # legacy/global mode only

    alpha_belief: float = 0.1
    gamma: float = 0.1
    pi0: float = 0.10
    corrupted_sigma_mult: float = 8.0
    fusion_weight_floor: float = 1e-3
    reg: float = 1e-9

    safety_projection: Literal["none", "simplex", "bounds"] = "none"
    protected_dim: int = 0
    floor: float = 0.15
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None


class ImprovedFusionFilter:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.cfg = config if config is not None else FusionConfig()
        self.t: int = 0
        self.belief: Optional[np.ndarray] = None
        self.pi: np.ndarray = np.full(self.cfg.num_sensors_m, 1.0 - self.cfg.pi0)
        self.sigma_reliable: Optional[float] = None
        self.cooldown_remaining: int = 0
        self._was_frozen: bool = False
        self._calib_buf = []
        self.majority_threshold = math.ceil(self.cfg.num_sensors_m / 2)

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

    def project(self, v: np.ndarray) -> np.ndarray:
        mode = self.cfg.safety_projection
        if mode == "none":
            return v
        if not np.all(np.isfinite(v)):
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

    def _log_gaussian_lik(self, resid_sq_sum, sigma):
        k = self.cfg.dim_k
        return -0.5 * k * np.log(2 * np.pi * (sigma ** 2)) - resid_sq_sum / (2 * (sigma ** 2))

    def update(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        assert x.shape == (self.cfg.num_sensors_m, self.cfg.dim_k)

        finite_mask = np.all(np.isfinite(x), axis=1)
        n_valid = int(finite_mask.sum())
        if n_valid == 0:
            return {
                "state": self.project(self.belief) if self.belief is not None else None,
                "sensor_trust": self.pi.copy(), "shift_detected": False,
                "is_frozen": self.cooldown_remaining > 0,
                "calibrating": self.t < self.cfg.calib_window,
                "input_rejected": True, "valid_sensor_mask": finite_mask,
            }

        shift_detected = False

        if self.t < self.cfg.calib_window:
            self._calib_buf.append((x, finite_mask))
            meds, mads = [], []
            for xi, mi in self._calib_buf:
                valid = xi[mi]
                if valid.shape[0] == 0: continue
                med = np.median(valid, axis=0)
                meds.append(med)
                mads.append(np.median(np.abs(valid - med), axis=0))
            consensus = meds[-1]
            sigma_est = 1.4826 * np.array(mads).mean()
            self.sigma_reliable = float(max(sigma_est, math.sqrt(self.cfg.reg)))
            self.belief = consensus.copy() if self.belief is None \
                else (self.belief * self.t + consensus) / (self.t + 1)
            self.t += 1
            return {
                "state": self.project(self.belief), "sensor_trust": self.pi.copy(),
                "shift_detected": False, "is_frozen": False, "calibrating": True,
                "input_rejected": False, "valid_sensor_mask": finite_mask,
            }

        resid_sq = np.full(self.cfg.num_sensors_m, np.nan)
        resid_sq[finite_mask] = ((x[finite_mask] - self.belief[None, :]) ** 2).sum(axis=1)
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
        if is_frozen and not self._was_frozen:
            shift_detected = True
        self._was_frozen = is_frozen

        if is_frozen and n_valid >= 2:
            fused_raw = np.median(x[finite_mask], axis=0)
            p_reliable = np.full(self.cfg.num_sensors_m, np.nan)
        else:
            log_l_rel = self._log_gaussian_lik(resid_sq, self.sigma_reliable)
            log_l_cor = self._log_gaussian_lik(resid_sq, self.sigma_reliable * self.cfg.corrupted_sigma_mult)
            pi_safe = np.clip(self.pi, self.cfg.reg, 1.0 - self.cfg.reg)
            prior_odds = np.log(pi_safe / (1.0 - pi_safe))
            post_odds = np.clip(prior_odds + (log_l_rel - log_l_cor), -50.0, 50.0)
            p_reliable = 1.0 / (1.0 + np.exp(-post_odds))
            weights = np.where(finite_mask, np.maximum(p_reliable, self.cfg.fusion_weight_floor), 0.0)
            fused_raw = (weights[:, None] * np.nan_to_num(x)).sum(axis=0) / (weights.sum() + self.cfg.reg)

        self.belief = (1.0 - self.cfg.alpha_belief) * self.belief + self.cfg.alpha_belief * fused_raw

        if not is_frozen:
            if np.all(np.isnan(p_reliable)):
                log_l_rel = self._log_gaussian_lik(resid_sq, self.sigma_reliable)
                log_l_cor = self._log_gaussian_lik(resid_sq, self.sigma_reliable * self.cfg.corrupted_sigma_mult)
                pi_safe = np.clip(self.pi, self.cfg.reg, 1.0 - self.cfg.reg)
                prior_odds = np.log(pi_safe / (1.0 - pi_safe))
                post_odds = np.clip(prior_odds + (log_l_rel - log_l_cor), -50.0, 50.0)
                p_reliable = 1.0 / (1.0 + np.exp(-post_odds))
            self.pi = np.where(finite_mask, (1.0 - self.cfg.gamma) * self.pi + self.cfg.gamma * np.nan_to_num(p_reliable), self.pi)

        self.t += 1
        return {
            "state": self.project(self.belief), "sensor_trust": self.pi.copy(),
            "n_elevated": n_elevated, "shift_detected": shift_detected, "is_frozen": is_frozen,
            "calibrating": False, "input_rejected": False, "valid_sensor_mask": finite_mask,
        }
```

---

## 5. Regression Evidence — v3.5.0 Bugs, Reproduced

Captured verbatim from `python3 reproduce_v35_bugs.py` against the unmodified v3.5.0
implementation (`original_fusion.py`, a direct transcription of the spec in the prior document
version):

```text
==============================================================================
BUG 1: single NaN sensor reading crashes the filter
==============================================================================
CRASHED: IndexError: index -1 is out of bounds for axis 0 with size 0

==============================================================================
BUG 2: 'protected' majority trust erodes to 0 over a long persistent window
     (M=3, sensors 0&1 shift +10 sigma and STAY there; sensor 2 stays put)
==============================================================================
  step=    40  pi = [0.729    0.729    0.918852]   <- pi[0],pi[1] are the CORRECT sensors
  step=   300  pi = [0.135085 0.135085 0.984904]   <- pi[0],pi[1] are the CORRECT sensors
  step=  1000  pi = [0.00131 0.00131 0.99983]      <- pi[0],pi[1] are the CORRECT sensors
  step=  3000  pi = [0. 0. 1.]                     <- pi[0],pi[1] are the CORRECT sensors
  step= 10000  pi = [0. 0. 1.]
  Result: the two correctly-reporting sensors end up fully distrusted (pi->0),
  and the single stuck/broken sensor ends up fully trusted (pi->1). Backwards.

==============================================================================
BUG 3: table (Sec. 6) vs. code mismatch at the exact 50% tie, M=6, c=3
==============================================================================
  Table claims M=6,c=3 (50%) -> 'Majority Shift' -> freeze triggers.
  Code (n_elevated > M//2 = 3): ever froze? False   (code disagrees with its own table)

==============================================================================
BUG 4: mean/variance calibration is not robust to a bad sensor during calib
==============================================================================
  sigma_reliable, clean calibration:       0.0427
  sigma_reliable, one bad sensor in calib:  1.8651
  inflation factor: 43.7x
```

---

## 6. Verification Suite Results (v4.0.0)

Captured verbatim from `python3 test_v4.py`:

```text
[PASS] simplex projection: sums to 1 and respects protected floor  -- proj=[0.15 0.255 0.255 0.17 0.17]
[PASS] uniform shifts up to 20 sigma converge (belief tracks target, no lockup)
[PASS] majority cohort trust never collapses toward 0 (v3.5 went 0.9->~0 by step 3000)  -- min(pi[0],pi[1]) over 5000 steps = 0.5101 (10-seed range: 0.53-0.81)
[PASS] resolves to trusting the correct 2-sensor majority, not the 1 stuck dissenter  -- final pi = [0.6833 0.7129 0.] (v3.5 final pi was [0.0, 0.0, 1.0] -- backwards)
[PASS] frozen-window fusion tracks the 2-sensor majority cohort, not the 1 dissenting sensor  -- dist_to_majority=0.012  dist_to_dissenter=0.644
[PASS] M=6 minority fault (2-of-6) discounted to <1e-3 within ~100 steps  -- first step pi[0]<1e-3: 64, final pi=[0. 0. 1. 1. 1. 1.]
[PASS] exact 50% tie (M=6,c=3) now triggers majority/freeze path (matches documented table)
[PASS] extreme residual (1e6) produces finite state and trust, no warnings
[PASS] single NaN sensor reading: no crash, belief stays finite and recovers
[PASS] all-sensor NaN frame is rejected, previous belief held unchanged
[PASS] median/MAD calibration resists a persistently offset sensor during calib window  -- sigma_reliable contaminated=0.0507 vs clean=0.0397
[PASS] shift_detected fires rarely for one persistent incident (v3.5 fires 13x/200 steps)  -- count=2 (v3.5: 13)
[PASS] throughput measured (informational, hardware-dependent)  -- 8086 Hz (0.1237 ms/step)

SUMMARY: 13/13 PASSED
```

Stability check: the full suite was re-run under 5 different top-level RNG seeds
(0, 1, 2, 42, 999); all runs pass 13/13.

### 6.1 Minority-fault decay (claim held up)

M=6, 2-of-6 persistent fault: $\pi_{\text{bad}}$ crosses below $10^{-3}$ by step 64 and reaches
$\approx 2\times10^{-4}$ by step 80, consistent with the v3.5.0 claim ("0.90 → 0.0002 in 80
steps"). The geometric-decay math checks out: with $\gamma=0.1$ and $p\approx0$ every step,
$\pi_t \approx \pi_0 \cdot 0.9^t$, which hits $\approx 2\times10^{-4}$ at $t\approx80$.

### 6.2 Persistent majority ambiguity (claim did *not* hold up in v3.5.0; fixed in v4.0.0)

This is the central finding. §5's Bug 2 is not a rare edge case — it is exactly the scenario
v3.5.0's own introduction (§1.1, problem 1) and conclusion (§6, "immutable mathematical boundary")
claim to solve. Across 10 independent seeds of the same M=3, 2-of-3-persistent-ambiguity scenario,
v4.0.0 keeps the two correctly-reporting sensors' trust between 0.53 and 1.0 throughout a
5000-step run and correctly discounts the single stuck sensor to ~0 in every trial — the design's
own stated intent, now actually realized.

### 6.3 Calibration robustness (new)

One sensor offset by a fixed amount for the entire 20-step calibration window:

| | clean calibration | contaminated calibration | inflation |
|---|---|---|---|
| v3.5.0 (mean/variance) | 0.0427 | 1.8651 | **43.7×** |
| v4.0.0 (median/MAD) | 0.0397 | 0.0507 | **1.3×** |

### 6.4 Unprotected-baseline comparison (claim held up)

Reproducing v3.5.0's Fig./KPI comparison against a filter with the freeze mechanism disabled
(`w_freeze=0`) on a clean, uniform, all-sensors-move-together shift: minimum trust during the
transient was **0.1287**, consistent with the document's claimed "~0.11 collapse."

### 6.5 Throughput (measured on this machine, not the document's original hardware)

| | v3.5.0 | v4.0.0 |
|---|---|---|
| Sustained throughput, M=6,K=5 | ~10,900-13,100 Hz | ~6,700-8,100 Hz |
| Per-step latency | ~0.076-0.092 ms | ~0.124-0.149 ms |

v4.0.0 is roughly 35-40% slower on identical hardware, from added input validation and the
median computation used while frozen. Both are far inside real-time budgets for control loops up
to a few kHz; neither number should be read as portable to different hardware — the document's
original "16,642 Hz" figure is *of this same character* (a measurement on one unspecified
machine), not a general guarantee, and should always be re-measured on the deployment target.

### 6.6 Safety-floor projection ("simplex" mode): true, but not informative the way it's framed

20,000 steps with 1%-probability 20σ spikes, `safety_projection="simplex"`: **0 violations**.
This reproduces the v3.5.0 KPI, but it should not be read as evidence of emergent robustness under
stress — `project()` enforces $\sum w = 1,\ w\ge0,\ w[d]\ge F$ deterministically on every call by
construction. The only way to violate it is a bug in `project()` itself (verified separately by
Test 1) or a non-finite `belief` reaching the projector (guarded against by F4). A 20,000-step
stress run doesn't add information beyond those two direct checks; it mainly costs wall-clock
time. Framing it as a "stress test" in the v3.5.0 document overstates what it demonstrates.

---

## 7. Known Limitations (honest, not resolved by v4.0.0)

- **The information-theoretic bound is real and still applies.** When $c \ge \lceil M/2\rceil$ and
  the disagreement is *permanent* (never resolves), v4.0.0 will eventually classify the persisting
  cohort as the majority and track it via median — same as v3.5.0 intended, now actually durable.
  But nothing in either version can tell a genuine world shift from a sufficiently patient
  *correlated* fault engineered to look like one and outlast a stuck/failed minority. A
  supervisory layer (cross-checking against an independent modality, a human-in-the-loop
  escalation after some very long dwell time, or a hardware watchdog) is still required for
  genuinely safety-critical deployment; this filter cannot resolve that ambiguity by construction,
  and no fusion-layer fix can.
- **`shift_detected` is not perfectly idempotent.** Near the majority/minority classification
  boundary, median-fusion convergence can cause `is_frozen` to genuinely toggle a small number of
  times before settling (2 events over 200 steps in testing, vs. 13 for v3.5.0's leak) — this is
  real re-classification, not noise, but a downstream alerting system should still debounce it.
- **Throughput regression (~35-40%)** from added validation and median computation. Not
  reoptimized in this pass; the residual/likelihood computation still allocates a handful of
  temporary arrays per step that could be fused if higher throughput is required.
- **`"bounds"` projection mode is new and only lightly exercised** (elementwise clip, no
  regression suite beyond a smoke check) — treat as a starting point for physical-state deployments
  rather than a verified default the way `"simplex"` and `"none"` are.
- **Per-sensor exclusion on `NaN` (F4) assumes intermittent, not permanent, sensor loss.** A
  sensor that is *always* `NaN` is silently excluded from every fusion step forever but still
  contributes an entry to `pi` and `sensor_trust` that never updates — nothing currently surfaces
  "this sensor has not reported valid data in N steps" as its own condition.

---

## 8. Migration Notes (v3.5.0 → v4.0.0)

- `ImprovedFusionConfig` → `FusionConfig`.
- `floor`, `protected_dim` only take effect when `safety_projection="simplex"` — set that
  explicitly if you were relying on v3.5.0's always-on projection; otherwise `state` is now the
  raw (unconstrained) belief.
- Output dict now includes `input_rejected` (bool) and `valid_sensor_mask` (per-sensor bool array)
  — check these if you were previously assuming every frame updates `belief`.
- `w_freeze` semantics changed from "one-shot pulse length" to "cooldown after the majority
  condition clears" — for a persistent majority condition, effective freeze duration is now
  unbounded (by design; see §3.3) rather than capped at 15 steps per pulse.

---

## 9. Integration Example

```python
from fusion_v4 import ImprovedFusionFilter, FusionConfig
import numpy as np

cfg = FusionConfig(dim_k=5, num_sensors_m=6, selective_freeze=True,
                    safety_projection="none")   # set "bounds" or "simplex" if your state needs it
filter_engine = ImprovedFusionFilter(cfg)

while system_running:
    x_frame = read_hardware_sensors()          # [6, 5]
    out = filter_engine.update(x_frame)

    if out["input_rejected"]:
        log_warning("all sensors reported non-finite data this frame; belief held")
    elif out["shift_detected"]:
        log_event(f"majority anomaly onset, n_elevated={out['n_elevated']}")

    state_estimate = out["state"]
    sensor_health = out["sensor_trust"]
```

---

## Appendix: File Manifest

| File | Contents |
|---|---|
| `fusion_v4.py` | v4.0.0 implementation (§4) |
| `original_fusion.py` | Verbatim v3.5.0 transcription, kept for regression comparison |
| `test_v4.py` | 13-test verification suite (§6) |
| `reproduce_v35_bugs.py` | Standalone reproduction of the 4 bugs in §5 |
| `verified_test_output.txt` | Captured stdout of the passing v4.0.0 suite |
| `v35_bug_evidence.txt` | Captured stdout of the v3.5.0 bug reproductions |
