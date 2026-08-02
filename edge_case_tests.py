import sys
sys.path.insert(0, "/home/claude/policy_benchmark"); sys.path.insert(0, "/home/claude/fusion_test")
from fusion_engine_v4_1 import FusionEngineV4_1, FusionConfig
import numpy as np

K = 4
cfg = FusionConfig(dim_k=K, num_sensors_m=3, calib_window=20, w_freeze=15,
                    max_freeze_duration=60, detect_sigma_mult=3.0, alpha_belief=0.1,
                    gamma=0.1, pi0=0.10)
eng = FusionEngineV4_1(cfg)
rng = np.random.default_rng(0)
mu = np.array([0.25] * K)

for t in range(20):
    eng.update(np.tile(mu, (3, 1)) + rng.normal(0, 0.02, size=(3, K)))

# Feed a PERMANENT, never-resolving disagreement across all 3 sensors and check
# the watchdog actually bounds how long the engine can stay frozen.
runs, current = [], 0
history = []
for t in range(200):
    x = np.array([
        mu + rng.normal(0, 0.02, K) + np.array([3.0] + [0] * (K - 1)),
        mu + rng.normal(0, 0.02, K) - np.array([3.0] + [0] * (K - 1)),
        mu + rng.normal(0, 0.02, K) + np.array([0, 3.0] + [0] * (K - 2)),
    ])
    out = eng.update(x)
    history.append(out["is_frozen"])
    if out["is_frozen"]:
        current += 1
    else:
        if current > 0:
            runs.append(current)
        current = 0
if current > 0:
    runs.append(current)

print("WATCHDOG LIVENESS TEST (persistent, never-resolving anomaly, 200 steps)")
print("longest contiguous frozen run:", max(runs) if runs else 0, " (configured max_freeze_duration:", cfg.max_freeze_duration, ")")
print("number of forced watchdog releases:", len(runs))
print("total steps frozen:", sum(history), "/ 200")
assert max(runs) < cfg.max_freeze_duration, "WATCHDOG FAILED: exceeded configured max_freeze_duration"
print("PASS: watchdog bounds freeze duration as claimed")
print()

# Extreme-value spoofing: one sensor reports 1e12, check the fused belief doesn't blow up.
eng2 = FusionEngineV4_1(cfg)
for t in range(50):
    x = np.tile(mu, (3, 1)) + rng.normal(0, 0.02, size=(3, K))
    out2 = eng2.update(x)
pre_spoof_state = out2["state"].copy()

x_spoof = np.array([mu + rng.normal(0, 0.02, K), mu + rng.normal(0, 0.02, K), np.array([1e12] * K)])
out2 = eng2.update(x_spoof)
print("EXTREME-VALUE SPOOFING TEST (1e12 on one of three sensors)")
print("pre-spoof state: ", pre_spoof_state)
print("post-spoof state:", out2["state"])
print("sensor_trust (pi) after spoof:", out2["sensor_trust"])
assert np.all(np.abs(out2["state"]) < 10), "SPOOFING TEST FAILED: fused belief blew up"
print("PASS: fused belief unaffected by the extreme spoofed reading")
