import numpy as np
import warnings
import time
from fusion_engine_v4_4 import FusionConfig, FusionEngineV4_4

results = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((status, name))
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))

def calib(f, M, K, steps, sigma=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(steps):
        f.update(rng.normal(0, sigma, size=(M, K)))

# ---- regression: everything that passed in v4.2/v4.3 still passes ----
try:
    FusionEngineV4_4(FusionConfig(num_sensors_m=2)); r1 = False
except ValueError: r1 = True
check("R1: num_sensors_m<3 still raises", r1)

f = FusionEngineV4_4(FusionConfig(dim_k=5, num_sensors_m=3, warn_on_disabled_safety_projection=False))
try:
    f.update(np.zeros((3, 4))); r2 = False
except ValueError: r2 = True
check("R2: bad shape still raises ValueError", r2)

cfg = FusionConfig(dim_k=5, num_sensors_m=3, warn_on_disabled_safety_projection=False)
f = FusionEngineV4_4(cfg)
rng = np.random.default_rng(0)
calib(f, 3, 5, 20, rng=rng)
x = rng.normal(0, 0.05, size=(3, 5)); x[0, 0] = 2e154
out = f.update(x)
check("R3: overflow-magnitude value still gated, output stays finite",
      out["state"] is not None and np.all(np.isfinite(out["state"])))

cfg = FusionConfig(dim_k=5, num_sensors_m=3, max_freeze_duration=60, frozen_fusion_method="gnc",
                    warn_on_disabled_safety_projection=False)
f = FusionEngineV4_4(cfg)
rng = np.random.default_rng(3)
calib(f, 3, 5, 20, rng=rng)
sigma_r = f.sigma_reliable
min_pi01 = 1.0
for i in range(5000):
    x = rng.normal(0, 0.05, size=(3, 5)); x[0] += 10*sigma_r; x[1] += 10*sigma_r
    f.update(x)
    min_pi01 = min(min_pi01, f.pi[0], f.pi[1])
check("R4: majority-ambiguity trust still resolves correctly, no collapse",
      min_pi01 > 0.5 and f.pi[2] < 0.3, f"min={min_pi01:.4f}, final={np.round(f.pi,4)}")

# ---- G1: corrupted_sigma_mult < 1.0 is now caught and clipped ----
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cfg = FusionConfig(dim_k=3, num_sensors_m=3, corrupted_sigma_mult=np.array([8.0, 8.0, 0.0]),
                        warn_on_disabled_safety_projection=False)
    f = FusionEngineV4_4(cfg)
    warned = any("corrupted_sigma_mult" in str(wi.message) for wi in w)
check("G1a: vector corrupted_sigma_mult with a 0 component warns and is clipped",
      warned and np.all(f.cfg.corrupted_sigma_mult >= 1.0),
      f"clipped to {f.cfg.corrupted_sigma_mult}")

rng = np.random.default_rng(0)
calib(f, 3, 3, 20, rng=rng)
sigma_r = f.sigma_reliable
p0s, p1s, p2s = [], [], []
for _ in range(100):
    x = rng.normal(0, 0.05, size=(3, 3)); x[0] += 20*sigma_r
    diff_sq = (x - f.belief[None,:])**2
    log_l_rel = f._log_gaussian_lik_vector(diff_sq, sigma_r, 1.0)
    log_l_cor = f._log_gaussian_lik_vector(diff_sq, sigma_r, f.cfg.corrupted_sigma_mult)
    pi_safe = np.clip(f.pi, cfg.reg, 1-cfg.reg)
    prior_odds = np.log(pi_safe/(1-pi_safe))
    p = 1/(1+np.exp(-np.clip(prior_odds+(log_l_rel-log_l_cor), -50, 50)))
    p0s.append(p[0]); p1s.append(p[1]); p2s.append(p[2])
check("G1b: after clipping, the actually-corrupted sensor is correctly distrusted "
      "and clean sensors are NOT dragged down with it (the original bug's core symptom) "
      "-- averaged over 100 trials, since any single noisy draw is inherently variable",
      np.mean(p0s) < 0.1 and np.mean(p1s) > 0.5 and np.mean(p2s) > 0.5,
      f"mean p_reliable = [{np.mean(p0s):.4f}, {np.mean(p1s):.4f}, {np.mean(p2s):.4f}] (sensor 0 is the real fault)")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cfg2 = FusionConfig(dim_k=3, num_sensors_m=3, corrupted_sigma_mult=0.5,
                         warn_on_disabled_safety_projection=False)
    f2 = FusionEngineV4_4(cfg2)
check("G1c: scalar corrupted_sigma_mult < 1.0 also caught (not just the vector form)",
      f2.cfg.corrupted_sigma_mult == 1.0)

# ---- G2: real recovery from a poisoned belief, not just a masked output ----
cfg = FusionConfig(dim_k=5, num_sensors_m=3, warn_on_disabled_safety_projection=False)
f = FusionEngineV4_4(cfg)
rng = np.random.default_rng(1)
calib(f, 3, 5, 20, rng=rng)
f.belief = np.full(5, np.nan)  # simulate poisoning via whatever route
out = f.update(rng.normal(0, 0.05, size=(3, 5)))
check("G2a: belief is actually reseeded (finite) on the very next valid frame, not just masked",
      np.all(np.isfinite(f.belief)) and out["belief_reseeded"] == True,
      f"belief after reseed: {np.round(f.belief,4)}")

# does detection actually work again afterward? (the real test -- not just "looks fine")
n_elevated_seen = set()
for i in range(20):
    out = f.update(rng.normal(0, 0.05, size=(3, 5)) + 50.0)  # obvious anomaly
    n_elevated_seen.add(out["n_elevated"])
check("G2b: fault detection is FUNCTIONALLY restored after reseed (a +50 anomaly is "
      "correctly flagged, unlike v4.3 where it stayed permanently blind)",
      n_elevated_seen != {0}, f"n_elevated values seen after reseed under +50 anomaly: {n_elevated_seen}")

# ---- optional min_sigma_reliable mitigation for the calib/operating mismatch ----
cfg_nomit = FusionConfig(dim_k=5, num_sensors_m=6, warn_on_disabled_safety_projection=False)
f_nomit = FusionEngineV4_4(cfg_nomit)
rng = np.random.default_rng(2)
calib(f_nomit, 6, 5, 20, sigma=0.001, rng=rng)
frozen_nomit = sum(f_nomit.update(rng.normal(0, 0.05, size=(6, 5)))["is_frozen"] for _ in range(500))

cfg_mit = FusionConfig(dim_k=5, num_sensors_m=6, min_sigma_reliable=0.04,
                        warn_on_disabled_safety_projection=False)
f_mit = FusionEngineV4_4(cfg_mit)
rng = np.random.default_rng(2)
calib(f_mit, 6, 5, 20, sigma=0.001, rng=rng)
frozen_mit = sum(f_mit.update(rng.normal(0, 0.05, size=(6, 5)))["is_frozen"] for _ in range(500))
check("min_sigma_reliable mitigates the quiet-calibration/louder-operation false-freeze scenario",
      frozen_mit < frozen_nomit / 5,
      f"frozen steps/500: no floor={frozen_nomit}, with floor={frozen_mit}")

n_pass = sum(1 for s, _ in results if s == "PASS")
print(f"\nSUMMARY: {n_pass}/{len(results)} PASSED")
