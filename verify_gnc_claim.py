import sys
sys.path.insert(0, "/home/claude/fusion_test")
import numpy as np
from fusion_engine_v4_1 import FusionEngineV4_1, FusionConfig

K = 3
M = 7
SIGMA_RELIABLE = 0.05
N_TRIALS = 500

cfg = FusionConfig(dim_k=K, num_sensors_m=M, gnc_c_start_mult=15.0, gnc_c_final_mult=3.0,
                    gnc_steps=5, gnc_irls_steps=3, reg=1e-9)
eng = FusionEngineV4_1(cfg)  # only used to call the bound _gnc_robust_fusion method

rng = np.random.default_rng(0)
mu_true = np.array([0.3, -0.2, 0.5])

print(f"{'n_corrupt':>10} {'frac':>6} | {'median RMSE':>12} {'GNC RMSE':>10}")
for n_corrupt in range(0, M):
    med_errs, gnc_errs = [], []
    for _ in range(N_TRIALS):
        x = mu_true[None, :] + rng.normal(0, SIGMA_RELIABLE, size=(M, K))
        corrupt_idx = rng.choice(M, size=n_corrupt, replace=False)
        for idx in corrupt_idx:
            # wild outlier, matching the "corrupted" model used elsewhere in this session:
            # far outside the reliable noise band, not a subtle bias
            x[idx] = mu_true + rng.normal(0, 1.0, size=K) * rng.choice([-1, 1], size=K) + rng.normal(0, 3.0, size=K)
        med = np.median(x, axis=0)
        gnc = eng._gnc_robust_fusion(x, SIGMA_RELIABLE)
        med_errs.append(np.linalg.norm(med - mu_true))
        gnc_errs.append(np.linalg.norm(gnc - mu_true))
    med_rmse = np.sqrt(np.mean(np.array(med_errs) ** 2))
    gnc_rmse = np.sqrt(np.mean(np.array(gnc_errs) ** 2))
    flag = "  <-- majority threshold (ceil(7/2)=4)" if n_corrupt == 4 else ""
    print(f"{n_corrupt:>10} {n_corrupt/M*100:>5.0f}% | {med_rmse:>12.4f} {gnc_rmse:>10.4f}{flag}")
