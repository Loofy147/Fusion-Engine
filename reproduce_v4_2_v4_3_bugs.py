"""Standalone reproductions of the two bugs found by testing v4.2+v4.3 together."""
import numpy as np
import warnings
from fusion_engine_v4_2_verified import FusionConfig, FusionEngineV4_1
from fusion_engine_v4_3 import FusionConfigV4_3, FusionEngineV4_3

print("=" * 78)
print("BUG G1: corrupted_sigma_mult < 1.0 degenerates trust for ALL sensors uniformly")
print("=" * 78)
for c2 in [0.0, 1e-6, 1e-3]:
    cfg = FusionConfig(dim_k=3, num_sensors_m=3, corrupted_sigma_mult=np.array([8.0, 8.0, c2]))
    f = FusionEngineV4_1(cfg)
    rng = np.random.default_rng(0)
    for _ in range(20):
        f.update(rng.normal(0, 0.05, size=(3, 3)))
    sigma_r = f.sigma_reliable
    x = rng.normal(0, 0.05, size=(3, 3))
    x[0] += 20 * sigma_r  # sensor 0 is the ONLY actually-corrupted sensor
    diff_sq = (x - f.belief[None, :]) ** 2
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        log_l_rel = f._log_gaussian_lik_vector(diff_sq, sigma_r, 1.0)
        log_l_cor = f._log_gaussian_lik_vector(diff_sq, sigma_r, cfg.corrupted_sigma_mult)
    pi_safe = np.clip(f.pi, cfg.reg, 1 - cfg.reg)
    prior_odds = np.log(pi_safe / (1 - pi_safe))
    p = 1 / (1 + np.exp(-np.clip(prior_odds + (log_l_rel - log_l_cor), -50, 50)))
    print(f"  corrupted_sigma_mult[2]={c2:<8}  p_reliable = {np.round(p, 6)}"
          f"   (sensor 0 is the real fault; correct answer is ~[low, high, high])")
print("  Result: a 0 in one dimension collapses ALL sensors to p~0 (everyone distrusted).")
print("  A near-zero (but nonzero) value collapses ALL sensors to p~1 (everyone trusted,")
print("  INCLUDING the actually-corrupted one). Neither matches the correct per-sensor")
print("  discrimination the whole engine exists to compute.")

print()
print("=" * 78)
print("BUG G2: v4.3's NaN-belief 'recovery' only masks project()'s output;")
print("        self.belief itself stays NaN forever, so detection goes permanently blind")
print("=" * 78)
cfg3 = FusionConfigV4_3(dim_k=5, num_sensors_m=3, warn_on_disabled_safety_projection=False)
f3 = FusionEngineV4_3(cfg3)
rng = np.random.default_rng(1)
for _ in range(20):
    f3.update(rng.normal(0, 0.05, size=(3, 5)))
f3._inner.belief = np.full(5, np.nan)  # simulate poisoning via any route
out = f3.update(rng.normal(0, 0.05, size=(3, 5)))
print(f"  reported state right after poisoning: {out['state']}  <- looks fine")
print(f"  internal self.belief at the same moment: {f3._inner.belief}  <- still NaN")
n_elevated_seen = set()
for i in range(300):
    out = f3.update(rng.normal(0, 0.05, size=(3, 5)) + 50.0)  # obvious, huge anomaly
    n_elevated_seen.add(out["n_elevated"])
print(f"  n_elevated values seen over 300 steps of a +50 anomaly: {n_elevated_seen}")
print(f"  internal belief after 300 more steps: {f3._inner.belief}")
print("  Result: the engine never recovers on its own. It reports a plausible-looking")
print("  state forever while every fault-detection mechanism (elevated/majority/watchdog)")
print("  is silently inert, because every residual computed against a NaN belief is NaN,")
print("  and NaN comparisons are always False.")
