"""
Independent verification of fusion_engine_v4_1.py against the categories
listed in Section 8.1 of the architecture spec. None of this test code
previously existed -- it's being written now to actually check the claims,
not to reproduce a suite we don't have access to.
"""
import time
import numpy as np
from fusion_engine_v4_1 import FusionEngineV4_1, FusionConfig

results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("::" + detail if detail else ""))


# 1. Simplex projection with protected floor
cfg = FusionConfig(dim_k=4, num_sensors_m=3, safety_projection="simplex",
                    protected_dim=0, floor=0.3)
eng = FusionEngineV4_1(cfg)
v = np.array([0.01, 0.9, 0.05, 0.04])  # dim 0 would get squeezed out by raw simplex proj
w = eng.project(v)
check("1. simplex+floor: sums to 1", np.isclose(w.sum(), 1.0), f"sum={w.sum():.6f}")
check("1. simplex+floor: protected dim >= floor", w[0] >= cfg.floor - 1e-9, f"w[0]={w[0]:.4f}")
check("1. simplex+floor: all non-negative", np.all(w >= -1e-12), f"min={w.min():.2e}")

# randomized stress version of the same invariant (this also covers "12. safety-floor invariants")
rng = np.random.default_rng(7)
ok = True
for _ in range(2000):
    v = rng.normal(0, 5, 4)
    w = eng.project(v)
    if w[0] < cfg.floor - 1e-8 or w.sum() - 1.0 > 1e-6 or np.any(w < -1e-9):
        ok = False
        break
check("12. safety-floor invariant holds over 2000 random vectors", ok)

# 2. Bound projection
cfg_b = FusionConfig(dim_k=3, num_sensors_m=3, safety_projection="bounds",
                      lower_bounds=np.array([0.0, -1.0, 5.0]),
                      upper_bounds=np.array([10.0, 1.0, 6.0]))
eng_b = FusionEngineV4_1(cfg_b)
out = eng_b.project(np.array([-5.0, 3.0, 100.0]))
check("2. bound projection clips correctly",
      np.allclose(out, [0.0, 1.0, 6.0]), f"got {out}")

# 3. Extreme shift convergence
cfg_c = FusionConfig(dim_k=2, num_sensors_m=5, calib_window=15, w_freeze=8, alpha_belief=0.15)
eng_c = FusionEngineV4_1(cfg_c)
rng2 = np.random.default_rng(1)
for _ in range(cfg_c.calib_window):
    eng_c.update(rng2.normal(0, 0.02, (5, 2)))
target = np.array([30.0, 30.0])
last_state = None
converged_at = None
for step in range(400):
    x = target + rng2.normal(0, 0.02, (5, 2))
    out = eng_c.update(x)
    last_state = out["state"]
    if converged_at is None and np.max(np.abs(last_state - target)) < 0.5:
        converged_at = step
check("3. extreme shift eventually converges near new target",
      converged_at is not None, f"converged_at step={converged_at}, final_state={last_state}")

# 4. Long-horizon freeze stability (no crash / no NaN / watchdog respected over a long persistent shift)
cfg_d = FusionConfig(dim_k=2, num_sensors_m=4, calib_window=10, w_freeze=5, max_freeze_duration=25)
eng_d = FusionEngineV4_1(cfg_d)
rng3 = np.random.default_rng(2)
for _ in range(cfg_d.calib_window):
    eng_d.update(rng3.normal(0, 0.02, (4, 2)))
bad = False
max_contig = 0
for step in range(2000):
    x = np.array([[0.0, 0.0], [40.0, 40.0], [40.0, 40.0], [40.0, 40.0]]) + rng3.normal(0, 0.02, (4, 2))
    out = eng_d.update(x)
    max_contig = max(max_contig, out["contiguous_frozen_steps"])
    if out["state"] is None or not np.all(np.isfinite(out["state"])):
        bad = True
        break
check("4. long-horizon freeze: 2000 steps, no NaN/crash", not bad)
check("4. long-horizon freeze: watchdog cap always respected",
      max_contig <= cfg_d.max_freeze_duration, f"max_contig={max_contig}, cap={cfg_d.max_freeze_duration}")

# 5. Persistent minority fault isolation
cfg_e = FusionConfig(dim_k=2, num_sensors_m=5, calib_window=15, w_freeze=5, gamma=0.15)
eng_e = FusionEngineV4_1(cfg_e)
rng4 = np.random.default_rng(3)
for _ in range(cfg_e.calib_window):
    eng_e.update(rng4.normal(0, 0.02, (5, 2)))
froze_ever = False
for step in range(120):
    x = rng4.normal(0, 0.02, (5, 2))
    x[0] += 25.0  # stream 0 persistently faulty, minority (1 of 5)
    out = eng_e.update(x)
    if out["is_frozen"]:
        froze_ever = True
check("5. minority fault: system never freezes (1-of-5 stays minority)", not froze_ever)
check("5. minority fault: faulty stream's trust drops well below others",
      eng_e.pi[0] < 0.3 and np.mean(eng_e.pi[1:]) > 0.8,
      f"pi={np.round(eng_e.pi,3)}")
check("5. minority fault: fused state stays near true value, not pulled toward fault",
      np.max(np.abs(eng_e.belief)) < 2.0, f"belief={eng_e.belief}")

# 6. Exact 50% tie handling (M=4, majority_threshold = ceil(4/2) = 2)
cfg_f = FusionConfig(dim_k=2, num_sensors_m=4, calib_window=10, w_freeze=3)
eng_f = FusionEngineV4_1(cfg_f)
rng5 = np.random.default_rng(4)
for _ in range(cfg_f.calib_window):
    eng_f.update(rng5.normal(0, 0.02, (4, 2)))
check("6. majority_threshold computed as ceil(M/2)", eng_f.majority_threshold == 2)
x_tie = np.array([[0.0, 0.0], [0.0, 0.0], [30.0, 30.0], [30.0, 30.0]])
out_tie = eng_f.update(x_tie)
check("6. exact 50% elevated triggers freeze deterministically (no crash/exception)",
      out_tie["is_frozen"] is True, f"n_elevated={out_tie['n_elevated']}")

# 7. Robust calibration under startup outliers
cfg_g = FusionConfig(dim_k=2, num_sensors_m=5, calib_window=20)
eng_g = FusionEngineV4_1(cfg_g)
rng6 = np.random.default_rng(5)
for step in range(cfg_g.calib_window):
    x = rng6.normal(0, 0.05, (5, 2))
    if step == 3:
        x[2] += 500.0  # single wild startup outlier on one stream
    eng_g.update(x)
check("7. calibration robust to a single startup outlier (sigma stays small, not blown up)",
      eng_g.sigma_reliable < 1.0, f"sigma_reliable={eng_g.sigma_reliable:.4f}")

# 8. NaN handling (partial + total)
cfg_h = FusionConfig(dim_k=2, num_sensors_m=4, calib_window=10)
eng_h = FusionEngineV4_1(cfg_h)
rng7 = np.random.default_rng(6)
for _ in range(cfg_h.calib_window):
    eng_h.update(rng7.normal(0, 0.02, (4, 2)))
x_partial = rng7.normal(0, 0.02, (4, 2))
x_partial[1] = np.nan
out_partial = eng_h.update(x_partial)
check("8. NaN handling: partial NaN row masked, not rejected",
      not out_partial["input_rejected"] and not out_partial["valid_sensor_mask"][1])
out_total = eng_h.update(np.full((4, 2), np.nan))
check("8. NaN handling: all-NaN frame rejected cleanly, belief held",
      out_total["input_rejected"] and np.allclose(out_total["state"], eng_h.belief))

# 9. Extreme residual stability (overflow guard)
cfg_i = FusionConfig(dim_k=2, num_sensors_m=4, calib_window=10)
eng_i = FusionEngineV4_1(cfg_i)
rng8 = np.random.default_rng(8)
for _ in range(cfg_i.calib_window):
    eng_i.update(rng8.normal(0, 0.02, (4, 2)))
x_extreme = np.array([[0.0, 0.0], [1e12, 1e12], [-1e12, 1e12], [0.0, 0.0]])
out_extreme = eng_i.update(x_extreme)
check("9. extreme residual: no NaN/Inf leaks into state or trust",
      np.all(np.isfinite(out_extreme["state"])) and np.all(np.isfinite(eng_i.pi)),
      f"state={out_extreme['state']}")

# 10. Debounced shift detection (edge-triggered, not level)
cfg_j = FusionConfig(dim_k=2, num_sensors_m=4, calib_window=10, w_freeze=6)
eng_j = FusionEngineV4_1(cfg_j)
rng9 = np.random.default_rng(9)
for _ in range(cfg_j.calib_window):
    eng_j.update(rng9.normal(0, 0.02, (4, 2)))
edges = 0
for step in range(15):
    x = np.array([[0.0, 0.0], [30.0, 30.0], [30.0, 30.0], [30.0, 30.0]]) if step < 3 else rng9.normal(0, 0.02, (4, 2))
    out = eng_j.update(x)
    if out["shift_detected"]:
        edges += 1
check("10. shift_detected fires exactly once for one shift episode (edge-triggered)",
      edges == 1, f"edges fired={edges}")

# 11. Throughput
cfg_k = FusionConfig(dim_k=16, num_sensors_m=12, calib_window=20)
eng_k = FusionEngineV4_1(cfg_k)
rngk = np.random.default_rng(10)
for _ in range(cfg_k.calib_window):
    eng_k.update(rngk.normal(0, 0.02, (12, 16)))
N = 20000
t0 = time.perf_counter()
for _ in range(N):
    eng_k.update(rngk.normal(0, 0.02, (12, 16)))
dt = time.perf_counter() - t0
rate = N / dt
check("11. throughput measured (informational, no target given in spec)",
      True, f"{rate:,.0f} updates/sec at M=12,K=16 ({dt*1e6/N:.1f} us/update)")

# Extra: operating-assumption enforcement check (spec 5.1 claims M>=3, K>=1 -- is it enforced?)
try:
    bad_cfg = FusionConfig(dim_k=2, num_sensors_m=1)
    bad_eng = FusionEngineV4_1(bad_cfg)
    bad_eng.update(np.array([[0.0, 0.0]]))
    enforced = False
except Exception:
    enforced = True
check("EXTRA: spec's 'M>=3' operating assumption (5.1) is enforced in code", enforced,
      "if FAIL: M=1 silently runs with no warning, majority_threshold=ceil(1/2)=1")

print()
n_pass = sum(1 for _, ok, _ in results if ok)
print(f"=== {n_pass}/{len(results)} checks passed ===")
for name, ok, detail in results:
    if not ok:
        print("  FAILED:", name, "--", detail)
