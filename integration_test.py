import sys
sys.path.insert(0, "/home/claude/policy_benchmark")
sys.path.insert(0, "/home/claude/fusion_test")

import json
import itertools
import numpy as np

from common import K, PROTECTED_DIM, FLOOR, project, sample_regime_pair
from v3_phase2_protected import sensor_readings, censored_recovery_stats, T, T_SHIFT, CALIB_WINDOW
from fusion_engine_v4_1 import FusionEngineV4_1, FusionConfig

N_SEEDS = 150
BASE_SEED = 55555
EPS = 0.05
STABLE_WINDOW = 10


def run_v4_1(seed, correlated_crisis, adversarial, frozen_fusion_method):
    rng = np.random.default_rng(seed)
    mu_a, mu_b = sample_regime_pair(rng, min_shift=0.3)
    bias_vecs = {1: rng.normal(0, 0.12, size=K), 2: rng.normal(0, 0.12, size=K)}

    cfg = FusionConfig(
        dim_k=K, num_sensors_m=3, calib_window=CALIB_WINDOW, w_freeze=15,
        detect_sigma_mult=3.0, selective_freeze=True, alpha_belief=0.1, gamma=0.1,
        pi0=0.10, corrupted_sigma_mult=8.0, safety_projection="simplex",
        protected_dim=PROTECTED_DIM, floor=FLOOR, frozen_fusion_method=frozen_fusion_method,
    )
    eng = FusionEngineV4_1(cfg)

    mu_true_log = np.zeros((T, K))
    state_log = np.zeros((T, K))
    for t in range(T):
        mu_true = mu_a if t < T_SHIFT else mu_b
        x = sensor_readings(rng, mu_true, bias_vecs, t, T_SHIFT, correlated_crisis, adversarial)
        out = eng.update(x)
        mu_true_log[t] = mu_true
        state_log[t] = out["state"]

    err = np.linalg.norm(state_log - mu_true_log, axis=1)
    return err


def main():
    results = {}
    for correlated_crisis, adversarial in itertools.product([False, True], [False, True]):
        key = f"correlated={correlated_crisis}_adversarial={adversarial}"
        stats = {name: {"rec": [], "flag": [], "cost": []} for name in ("v4_1_median", "v4_1_gnc")}
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            for name, method in (("v4_1_median", "median"), ("v4_1_gnc", "gnc")):
                err = run_v4_1(seed, correlated_crisis, adversarial, method)
                rf, tf = censored_recovery_stats(err, T_SHIFT, EPS, STABLE_WINDOW, T)
                stats[name]["flag"].append(rf)
                stats[name]["rec"].append(tf)
                stats[name]["cost"].append(float(np.sum(err)))
        results[key] = {}
        for name in ("v4_1_median", "v4_1_gnc"):
            flags = np.array(stats[name]["flag"]); recs = np.array(stats[name]["rec"])
            results[key][name] = {
                "recovered_fraction": float(flags.mean()),
                "restricted_mean_censored": float(recs.mean()),
                "combined_cost_mean": float(np.mean(stats[name]["cost"])),
            }

    with open("/home/claude/fusion_test/v4_1_vs_v3_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Pull in the already-computed v3 numbers (frozen / adapt / prot) for a side-by-side
    with open("/home/claude/policy_benchmark/results_v3/phase2_protected_summary.json") as f:
        v3 = json.load(f)["results"]

    print(f"{'crisis':>7} {'advers':>7} {'policy':>14} | {'recov%':>7} {'restr.mean':>11} {'cost':>8}")
    for key in results:
        for name in ("frozen", "adapt", "prot"):
            r = v3[key][name]
            print(f"{key.split('_')[0].split('=')[1]:>7} {key.split('_')[1].split('=')[1]:>7} {name:>14} | "
                  f"{r['recovered_fraction']*100:>6.1f}% {r['restricted_mean_censored']:>11.1f} {r['combined_cost_mean']:>8.2f}")
        for name in ("v4_1_median", "v4_1_gnc"):
            r = results[key][name]
            print(f"{key.split('_')[0].split('=')[1]:>7} {key.split('_')[1].split('=')[1]:>7} {name:>14} | "
                  f"{r['recovered_fraction']*100:>6.1f}% {r['restricted_mean_censored']:>11.1f} {r['combined_cost_mean']:>8.2f}")
        print()


if __name__ == "__main__":
    main()
