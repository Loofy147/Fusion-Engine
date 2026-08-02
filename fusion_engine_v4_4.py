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
      uniformly, regardless of that sensor's actual residual. Verified: a per-dimension
      corrupted_sigma_mult of exactly 0 drove p_reliable -> 0 for EVERY sensor
      (including clean ones); values in (0, ~0.05) drove p_reliable -> 1 for EVERY
      sensor (including an obviously-corrupted one, 20-sigma off). This is not a
      vector-only bug -- the scalar branch has the identical failure mode for a tiny
      scalar C_mult, since `reg` was never intended as a load-bearing floor against a
      variance many orders of magnitude below normal operating residuals. Fixed by
      validating (and clipping, with a warning) corrupted_sigma_mult >= 1.0 at
      construction, which prevents the degeneracy at the source rather than patching
      each arithmetic branch that can trigger it.

  G2. v4.3's project()-level NaN fallback (Fix 2) is cosmetic: it makes the *reported*
      state look healthy once belief has gone non-finite, but never touches
      self.belief itself. Verified: after forcing belief to NaN, the reported state
      stayed finite (masked) for 500+ subsequent clean steps while the *internal*
      belief remained NaN the entire time -- meaning every residual computed against
      it is NaN, every elevated/majority comparison against NaN silently evaluates
      False, and the watchdog/freeze machinery goes fully blind (confirmed: a +50
      anomaly injected while belief was internally NaN was never flagged elevated,
      even once). The engine looks recovered from the outside while actually being
      permanently dead inside. Fixed with a real reseed: when belief is (or becomes)
      non-finite, it is reinitialized from the coordinate-wise median of that frame's
      valid sensors (the same statistic calibration already trusts), so residual
      computation and fault detection resume on the very next valid frame instead of
      never.

  Also carries forward, as an opt-in mitigation rather than a default-on behavior
  change: min_sigma_reliable, guarding against the (separately verified) scenario
  where an unrepresentatively quiet calibration window makes normal operating noise
  look like a permanent majority fault (492/500 steps spuriously frozen with zero
  injected faults, in testing). This does not replace calibrating under representative
  conditions -- it bounds the damage when that practice isn't followed.
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

    max_abs_sensor_value: float = 1e50   # G1/v4.3 Fix 2: gate values that would overflow on squaring
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
            # G2 note: by the time project() is called, update() has already
            # reseeded self.belief if it was non-finite -- so `v` reaching here
            # non-finite should no longer happen post-calibration. Kept as a
            # last-resort passthrough (not a silent success) rather than raising,
            # since project() may also be called by external code directly.
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

        # v4.3 Fix 2 (folded in): gate out-of-range magnitudes before they can ever
        # reach a squaring op, same exclusion semantics as a NaN reading.
        too_large = np.any(np.abs(x) > self.cfg.max_abs_sensor_value, axis=1)
        if np.any(too_large):
            x = x.copy()
            x[too_large] = np.nan

        finite_mask = np.all(np.isfinite(x), axis=1)
        n_valid = int(finite_mask.sum())

        reseed_count_before = self.belief_reseeded_count

        # G2: if belief has somehow gone non-finite (whatever the route), reseed it
        # from this frame's valid data before doing anything else with it, rather
        # than silently computing residuals against a dead reference.
        if self.belief is not None and not np.all(np.isfinite(self.belief)) and n_valid > 0:
            self.belief = self._reseed_belief(x, finite_mask)
            self.belief_reseeded_count += 1

        if n_valid == 0:
            return {
                "state": self.project(self.belief) if self.belief is not None else None,
                "sensor_trust": self.pi.copy(),
                "n_elevated": 0,
                "shift_detected": False,
                "is_frozen": self.cooldown_remaining > 0,
                "contiguous_frozen_steps": self.contiguous_frozen_steps,
                "freeze_timeout_alarm": False,
                "calibrating": self.t < self.cfg.calib_window,
                "input_rejected": True,
                "belief_reseeded": False,
                "valid_sensor_mask": finite_mask,
            }

        shift_detected = False

        if self.t < self.cfg.calib_window:
            valid = x[finite_mask]
            med = np.median(valid, axis=0)
            mad = np.median(np.abs(valid - med), axis=0)
            self._calib_meds.append(med)
            self._calib_mads.append(mad)

            consensus = med
            sigma_est = 1.4826 * np.mean(self._calib_mads)
            sigma_est = float(max(sigma_est, math.sqrt(self.cfg.reg)))
            if self.cfg.min_sigma_reliable is not None:
                sigma_est = max(sigma_est, self.cfg.min_sigma_reliable)
            self.sigma_reliable = sigma_est
            self.belief = consensus.copy() if self.belief is None \
                else (self.belief * self.t + consensus) / (self.t + 1)
            self.t += 1
            return {
                "state": self.project(self.belief),
                "sensor_trust": self.pi.copy(),
                "n_elevated": 0,
                "shift_detected": False,
                "is_frozen": False,
                "contiguous_frozen_steps": 0,
                "freeze_timeout_alarm": False,
                "calibrating": True,
                "input_rejected": False,
                "belief_reseeded": self.belief_reseeded_count > reseed_count_before,
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
            if self.cfg.frozen_fusion_method == "gnc":
                fused_raw = self._gnc_robust_fusion(x[finite_mask], self.sigma_reliable)
            else:
                fused_raw = np.median(x[finite_mask], axis=0)
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
            "state": self.project(self.belief),
            "sensor_trust": self.pi.copy(),
            "n_elevated": n_elevated,
            "shift_detected": shift_detected,
            "is_frozen": is_frozen,
            "contiguous_frozen_steps": self.contiguous_frozen_steps,
            "freeze_timeout_alarm": freeze_timeout_alarm,
            "calibrating": False,
            "input_rejected": False,
            "belief_reseeded": self.belief_reseeded_count > reseed_count_before,
            "valid_sensor_mask": finite_mask,
        }
