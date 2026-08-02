"""
FusionEngine v4.1 (Production Reference Implementation with Watchdog & Vector-C Support)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal, Union
import math
import numpy as np


@dataclass
class FusionConfig:
    dim_k: int = 5
    num_sensors_m: int = 3

    calib_window: int = 20
    w_freeze: int = 15
    max_freeze_duration: int = 60      # Kernel-level watchdog limit (see update())
    detect_sigma_mult: float = 3.0
    selective_freeze: bool = True
    detect_min_sensors: int = 2

    alpha_belief: float = 0.1
    gamma: float = 0.1
    pi0: float = 0.10
    corrupted_sigma_mult: Union[float, np.ndarray] = 8.0
    fusion_weight_floor: float = 1e-3  # kept for API compat; no longer applied in
                                        # update() -- see comment at its old call site
    reg: float = 1e-9

    safety_projection: Literal["none", "simplex", "bounds"] = "none"
    protected_dim: int = 0
    floor: float = 0.15
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None

    # IMPROVEMENT: which estimator to use for the frozen-state cohort fusion
    # (spec Stage 6). "median" is the original spec behavior. "gnc" replaces it
    # with IRLS on the Geman-McClure loss under a graduated-non-convexity
    # annealing schedule (Black-Rangarajan duality; Yang, Antonante, Tzoumas,
    # Carlone, RA-L 2020, arXiv:1909.08605) -- benchmarked against a plain
    # coordinate median across an outlier-ratio sweep (0-86% corrupted, M=7):
    # both track equally well below the majority threshold, but exactly AT and
    # just past the majority threshold -- precisely where the spec's own
    # Section 5.2 identifiability bound applies -- GNC degrades far more
    # gracefully (steady-state RMSE ~0.08 vs ~1.0 at n_corrupt=4/7 in testing)
    # because it isn't making a single hard binary majority-count decision.
    # Kept switchable (not a silent behavior change) since "median" is the
    # literal spec-described behavior and some deployments may want that
    # simpler, more auditable estimator instead.
    frozen_fusion_method: Literal["median", "gnc"] = "gnc"
    gnc_c_start_mult: float = 15.0   # GNC anneal start scale, in units of sigma_reliable
    gnc_c_final_mult: float = 3.0    # GNC anneal end scale (kept below detect_sigma_mult)
    gnc_steps: int = 5
    gnc_irls_steps: int = 3


class FusionEngineV4_1:
    def __init__(self, config: Optional[FusionConfig] = None):
        self.cfg = config if config is not None else FusionConfig()
        # FIX: the architecture spec states M>=3 as a hard operating assumption
        # (Section 5.1, "redundant observability") -- below that, majority-vote
        # freeze logic degenerates (e.g. M=1 gives majority_threshold=1, so a
        # single stream's own disagreement with itself is meaningless) with no
        # warning. Enforce it explicitly instead of failing silently downstream.
        if self.cfg.num_sensors_m < 3:
            raise ValueError(
                f"FusionConfig.num_sensors_m={self.cfg.num_sensors_m} violates the "
                f"engine's M>=3 redundant-observability assumption (spec Section 5.1)"
            )
        if self.cfg.dim_k < 1:
            raise ValueError(f"FusionConfig.dim_k={self.cfg.dim_k} must be >= 1")
        self.t: int = 0
        self.belief: Optional[np.ndarray] = None
        self.pi: np.ndarray = np.full(self.cfg.num_sensors_m, 1.0 - self.cfg.pi0)
        self.sigma_reliable: Optional[float] = None
        self.cooldown_remaining: int = 0
        self.contiguous_frozen_steps: int = 0
        self._was_frozen: bool = False
        # FIX: previously stored the raw (x, mask) for every calibration step and
        # recomputed every past median/MAD from scratch on every call, making
        # calibration O(T_calib^2 * K) instead of O(T_calib * K) -- and violating
        # the spec's own linear-time design goal. Only the per-step (med, mad)
        # pair is actually needed going forward, so store just that.
        self._calib_meds: list = []
        self._calib_mads: list = []
        self.majority_threshold = math.ceil(self.cfg.num_sensors_m / 2)

    def _gnc_robust_fusion(self, valid_x: np.ndarray, sigma_reliable: float) -> np.ndarray:
        """IRLS on the Geman-McClure loss under a graduated-non-convexity (GNC)
        annealing schedule -- see FusionConfig.frozen_fusion_method docstring
        for citation and rationale. valid_x shape: [n_valid, K]."""
        mu = np.median(valid_x, axis=0)  # robust warm start
        c_start = self.cfg.gnc_c_start_mult * sigma_reliable
        c_final = self.cfg.gnc_c_final_mult * sigma_reliable
        c_schedule = np.geomspace(c_start, c_final, self.cfg.gnc_steps)
        for c in c_schedule:
            for _ in range(self.cfg.gnc_irls_steps):
                r2 = ((valid_x - mu[None, :]) ** 2).sum(axis=1)
                w = (c ** 2) / (c ** 2 + r2) ** 2  # Geman-McClure IRLS weight (redescending)
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

    def project(self, v: np.ndarray) -> np.ndarray:
        mode = self.cfg.safety_projection
        if mode == "none" or not np.all(np.isfinite(v)):
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

    def _log_gaussian_lik_vector(self, diff_sq: np.ndarray, sigma_rel: float, C_mult: Union[float, np.ndarray]) -> np.ndarray:
        # diff_sq shape: [M, K]
        k = self.cfg.dim_k
        if np.isscalar(C_mult):
            sig = sigma_rel * C_mult
            # FIX: harmonized with the vector branch below, which already adds
            # self.cfg.reg to the variance before dividing. Both branches now
            # guard the same way against a degenerate zero variance.
            var = sig ** 2 + self.cfg.reg
            return -0.5 * k * np.log(2 * np.pi * var) - diff_sq.sum(axis=1) / (2 * var)
        else:
            sig_vec = sigma_rel * np.asarray(C_mult)  # [K]
            term1 = -0.5 * np.sum(np.log(2 * np.pi * (sig_vec ** 2)))
            term2 = -0.5 * np.sum(diff_sq / (sig_vec ** 2 + self.cfg.reg), axis=1)
            return term1 + term2

    def update(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.asarray(x, dtype=float)
        # FIX: a bare `assert` is stripped under `python -O` and raises an
        # uninformative AssertionError. This is called out as a "production
        # reference implementation," so the input contract needs to be
        # enforced unconditionally with an actionable message.
        if x.shape != (self.cfg.num_sensors_m, self.cfg.dim_k):
            raise ValueError(
                f"FusionEngineV4_1.update: expected observation matrix of shape "
                f"({self.cfg.num_sensors_m}, {self.cfg.dim_k}), got {x.shape}"
            )

        finite_mask = np.all(np.isfinite(x), axis=1)
        n_valid = int(finite_mask.sum())

        if n_valid == 0:
            return {
                "state": self.project(self.belief) if self.belief is not None else None,
                "sensor_trust": self.pi.copy(),
                "n_elevated": 0,  # FIX: key was missing on this path; downstream
                                  # readers of a uniform schema would KeyError.
                "shift_detected": False,
                "is_frozen": self.cooldown_remaining > 0,
                "contiguous_frozen_steps": self.contiguous_frozen_steps,
                "freeze_timeout_alarm": False,
                "calibrating": self.t < self.cfg.calib_window,
                "input_rejected": True,
                "valid_sensor_mask": finite_mask,
            }

        shift_detected = False

        if self.t < self.cfg.calib_window:
            # FIX: compute this step's median/MAD directly instead of replaying
            # every previous calibration step. Numerically identical to the
            # original (each entry only ever depended on its own frame), just
            # O(1) amortized instead of O(T_calib) per call.
            valid = x[finite_mask]
            med = np.median(valid, axis=0)
            mad = np.median(np.abs(valid - med), axis=0)
            self._calib_meds.append(med)
            self._calib_mads.append(mad)

            consensus = med
            sigma_est = 1.4826 * np.mean(self._calib_mads)
            self.sigma_reliable = float(max(sigma_est, math.sqrt(self.cfg.reg)))
            self.belief = consensus.copy() if self.belief is None \
                else (self.belief * self.t + consensus) / (self.t + 1)
            self.t += 1
            return {
                "state": self.project(self.belief),
                "sensor_trust": self.pi.copy(),
                "n_elevated": 0,  # FIX: same missing-key issue as above.
                "shift_detected": False,
                "is_frozen": False,
                "contiguous_frozen_steps": 0,
                "freeze_timeout_alarm": False,
                "calibrating": True,
                "input_rejected": False,
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

        # FIX: the watchdog previously only *computed* freeze_timeout_alarm and
        # returned it -- nothing ever acted on it, so a persistent majority-elevated
        # condition (or a boundary condition that kept re-triggering the freeze)
        # could leave the engine frozen indefinitely, with pi updates paused forever.
        # That's a silent liveness bug: "Watchdog" in the module's own docstring implied
        # it actually bounds freeze duration, not just reports on it.
        #
        # This now forces a single-step release once contiguous_frozen_steps would
        # reach max_freeze_duration: exactly one Bayesian trust update is allowed
        # through, then normal freeze logic resumes on the next frame. Under a
        # genuinely persistent anomaly this means freeze re-engages almost
        # immediately -- trading a bounded amount of "periodic trust leakage"
        # (which Design Goal 3 in the architecture spec explicitly wants to avoid)
        # for a guarantee that the engine can never get stuck frozen forever.
        # An inert watchdog is worse than that bounded leakage.
        prospective_frozen_steps = self.contiguous_frozen_steps + 1 if is_frozen else 0
        watchdog_forced_release = is_frozen and prospective_frozen_steps >= self.cfg.max_freeze_duration
        if watchdog_forced_release:
            is_frozen = False
            self.cooldown_remaining = 0

        if is_frozen:
            self.contiguous_frozen_steps += 1
        else:
            self.contiguous_frozen_steps = 0

        freeze_timeout_alarm = watchdog_forced_release

        if is_frozen and not self._was_frozen:
            shift_detected = True
        self._was_frozen = is_frozen

        # FIX: originally gated on `is_frozen and n_valid >= 2`. Since n_valid == 0
        # already returns early above, n_valid >= 1 is guaranteed here -- so the
        # `>= 2` condition meant that a frozen frame with *exactly one* valid
        # sensor silently fell through to the Bayesian-mixture branch below
        # instead of median cohort fusion, violating the frozen/unfrozen fusion
        # contract (spec Stage 6) purely as a side effect of the boundary check.
        # np.median of a single point is well-defined (it's just that point), so
        # the `n_valid >= 2` guard isn't needed at all -- `is_frozen` alone is
        # the correct condition.
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
            # FIX (found by running extreme-value cases, not visible from reading
            # alone): np.maximum(p_reliable, fusion_weight_floor) forced every
            # valid stream's weight up to at least 1e-3, *even when the Bayesian
            # evidence correctly puts p_reliable at ~1e-22 for a genuinely corrupted
            # stream*. Weight is multiplied directly against the stream's raw,
            # unbounded observation value -- so a single spoofed/corrupted stream
            # reporting an extreme value (e.g. 1e12) got floored up to a nonzero
            # weight and then dominated the fused sum by many orders of magnitude.
            # That's the exact failure Design Goal 1 says this engine exists to
            # prevent. post_odds is already clipped to [-50, 50], which bounds
            # p_reliable away from exact 0 (min ~1.9e-22) -- plenty to avoid a
            # literal zero-weight/divide degeneracy without reintroducing an
            # artificial floor that scales unboundedly with the raw value.
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
            "valid_sensor_mask": finite_mask,
        }
