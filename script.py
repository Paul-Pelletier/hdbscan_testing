# =========================================================
# Geometry-Driven Regime Discovery on Synthetic IV Surfaces
# HDBSCAN + 2D Chen–Lyons Signatures + Temporal Stickiness
# Time is used ONLY for diagnostics and visualization
# =========================================================

import numpy as np
import torch
import signatory
import matplotlib.pyplot as plt
import hdbscan
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa


# =========================================================
# 1. Synthetic IV Surface Generation (Market-consistent SVI)
# =========================================================

def generate_iv_surfaces(T, maturities, log_moneyness, regime_blocks, noise_std=0.002):

    def regime_at_time(t):
        for start, end, r in regime_blocks:
            if start <= t <= end:
                return r
        return regime_blocks[-1][2]

    def svi_total_var(k, a, b, rho, m, sigma):
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def svi_params(regime, tau):
        # ATM total variance anchor (~20% vol @ 1Y)
        a = 0.035 + 0.012 * (1 - np.exp(-tau / 1.5))

        if regime == "carry":
            b = 0.10 + 0.03 * np.exp(-tau)
            rho = -0.10
            m = 0.0
            sigma = 0.22
            return a, b, rho, m, sigma

        if regime == "crash":
            b = 0.16 + 0.04 * (1 - np.exp(-tau))
            rho = -0.75
            m = -0.02
            sigma = 0.18
            return a, b, rho, m, sigma

        if regime == "event":
            b = 0.13
            rho = 0.35 * np.tanh(2 * (tau - 1.0))
            m = 0.0
            sigma = 0.32 - 0.10 * np.tanh(2 * (tau - 1.0))
            return a, b, rho, m, sigma

        return a, 0.10, -0.10, 0.0, 0.20

    n_k = len(log_moneyness)
    n_tau = len(maturities)

    iv_surfaces = np.zeros((T, n_k, n_tau))
    true_regimes = []

    for t in range(T):
        r = regime_at_time(t)
        true_regimes.append(r)

        for j, tau in enumerate(maturities):
            a, b, rho, m, sigma = svi_params(r, tau)
            rho = np.clip(rho, -0.999, 0.999)

            for i, k in enumerate(log_moneyness):
                w = svi_total_var(k, a, b, rho, m, sigma)
                iv = np.sqrt(max(w / tau, 1e-10))
                iv_surfaces[t, i, j] = iv + np.random.normal(0.0, noise_std)

    return iv_surfaces, true_regimes


# =========================================================
# 2. 2D Signature Paths
# =========================================================

def extract_2d_paths(iv_surfaces, moneyness_idx, maturities):
    log_tau = np.log(maturities)
    paths = []

    for surface in iv_surfaces:
        path_set = []
        for idx in moneyness_idx:
            ivs = surface[idx, :]
            path_set.append(np.stack([log_tau, ivs], axis=1))
        paths.append(path_set)

    return paths


# =========================================================
# 3. Chen–Lyons Signatures
# =========================================================

def signature_2d(path, depth):
    t = torch.tensor(path, dtype=torch.float32).unsqueeze(0)
    return signatory.signature(t, depth=depth).squeeze(0).numpy()


def build_embeddings(paths, depth):
    X = []
    for pset in paths:
        sigs = [signature_2d(p, depth) for p in pset]
        X.append(np.concatenate(sigs))
    return np.asarray(X)


# =========================================================
# 4. Clustering
# =========================================================

def run_hdbscan(X, min_cluster_size=15, min_samples=5):
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True
    )
    labels = hdb.fit_predict(X)
    return labels, hdb.probabilities_, hdb


# =========================================================
# 5. Diagnostic Mapping (synthetic only)
# =========================================================

def map_clusters_to_regimes(labels, true_int):
    pred = np.full_like(true_int, -2)
    for cl in np.unique(labels):
        if cl == -1:
            continue
        idx = np.where(labels == cl)[0]
        counts = np.bincount(true_int[idx])
        if np.sum(counts == counts.max()) > 1:
            continue
        pred[idx] = np.argmax(counts)
    pred[labels == -1] = -1
    return pred


# =========================================================
# 6. Temporal Stickiness
# =========================================================

def sticky_filter(pred, prob, window=5, prob_thresh=0.8):
    out = np.full_like(pred, -2)
    prev = -2

    for t in range(len(pred)):
        w0 = max(0, t - window + 1)
        w_labels = pred[w0:t+1]
        w_prob = prob[w0:t+1]

        valid = w_labels >= 0
        if not np.any(valid):
            out[t] = prev
            continue

        vals, cnts = np.unique(w_labels[valid], return_counts=True)
        maj = vals[np.argmax(cnts)]

        mask = w_labels == maj
        if cnts.max() >= window // 2 + 1 and np.mean(w_prob[mask]) >= prob_thresh:
            out[t] = maj
            prev = maj
        else:
            out[t] = prev

    return out


# =========================================================
# 7. Visualization
# =========================================================

def visualize(X, labels, true_int, pred_raw, pred_smooth, prob, names):
    X2 = umap.UMAP(random_state=42).fit_transform(X)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].scatter(X2[:,0], X2[:,1], c=true_int, cmap="Set1", s=10)
    axs[0].set_title("UMAP – True Regime (Geometry Only)")

    axs[1].scatter(X2[:,0], X2[:,1], c=labels, cmap="tab10", s=10)
    axs[1].set_title("UMAP – HDBSCAN Clusters")

    sc = axs[2].scatter(X2[:,0], X2[:,1], c=prob, cmap="viridis", s=10)
    axs[2].set_title("UMAP – Membership Probability")
    plt.colorbar(sc, ax=axs[2])

    plt.tight_layout()
    plt.show()

    t = np.arange(len(true_int))
    fig, axs = plt.subplots(2, 1, figsize=(18, 7), sharex=True)

    axs[0].step(t, true_int, where="post", lw=2, color="black", label="True")
    axs[0].step(t, pred_smooth, where="post", lw=2.5, color="tab:blue", label="Sticky")
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels(names)
    axs[0].legend()

    axs[1].plot(t, prob, color="tab:green")
    axs[1].set_ylim(0, 1.05)
    axs[1].set_title("HDBSCAN Membership Probability")

    plt.tight_layout()
    plt.show()

    mask = pred_smooth >= 0
    print("ARI:", adjusted_rand_score(true_int[mask], pred_smooth[mask]))
    print("NMI:", normalized_mutual_info_score(true_int[mask], pred_smooth[mask]))


# =========================================================
# 8. 3D Animation
# =========================================================

def animate_iv_and_regimes(iv, log_moneyness, maturities, true_int, pred_smooth, prob, names,
                           fps=30, stride=2):

    Xg, Yg = np.meshgrid(log_moneyness, maturities, indexing="ij")
    T = iv.shape[0]

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1, 0.4])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax2d = fig.add_subplot(gs[0, 1])
    axp = fig.add_subplot(gs[1, 1], sharex=ax2d)

    t = np.arange(T)
    ax2d.step(t, true_int, where="post", color="black", lw=2, label="True Regime")
    ax2d.step(t, pred_smooth, where="post", color="tab:blue", lw=2, alpha=0.8, label="Sticky Regime")
    ax2d.set_yticks(range(len(names)))
    ax2d.set_yticklabels(names)
    ax2d.set_ylabel("Regime Index")
    ax2d.set_title("Regime Timeline (Step)")
    ax2d.legend()
    ax2d.set_xlim(0, T-1)

    axp.plot(t, prob, color="tab:green", lw=1.5, label="Membership Probability")
    axp.set_ylabel("HDBSCAN Probability")
    axp.set_xlabel("Time Index")
    axp.set_ylim(0, 1.05)
    axp.set_title("Membership Probability")
    axp.legend()

    cursor2d = ax2d.axvline(0, color="red", lw=2, alpha=0.7, label="Current t")
    cursorprob = axp.axvline(0, color="red", lw=2, alpha=0.7)
    prob_marker, = axp.plot([0], [prob[0]], 'o', color='red', markersize=8)

    surf = [ax3d.plot_surface(Xg, Yg, iv[0], cmap='viridis', edgecolor='k', alpha=0.95)]
    ax3d.set_xlabel('Log-Moneyness')
    ax3d.set_ylabel('Maturity')
    ax3d.set_zlabel('Implied Volatility')
    ax3d.set_title(f'IV Surface at t=0 | True: {names[true_int[0]]} | Sticky: {names[pred_smooth[0]]} | Prob: {prob[0]:.2f}')
    ax3d.view_init(elev=30, azim=135)

    def update(frame):
        i = min(frame * stride, T - 1)
        # Update 3D surface
        surf[0].remove()
        surf[0] = ax3d.plot_surface(Xg, Yg, iv[i], cmap='viridis', edgecolor='k', alpha=0.95)
        ax3d.set_title(f'IV Surface at t={i} | True: {names[true_int[i]]} | Sticky: {names[pred_smooth[i]]} | Prob: {prob[i]:.2f}')
        # Update vertical cursors
        cursor2d.set_xdata([i, i])
        cursorprob.set_xdata([i, i])
        # Update probability marker
        prob_marker.set_data([i], [prob[i]])
        return surf[0], cursor2d, cursorprob, prob_marker

    n_frames = (T + stride - 1) // stride
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False, repeat=True)
    plt.show()
    return anim


# =========================================================
# 9. Main
# =========================================================

if __name__ == "__main__":
    np.random.seed(42)

    T = 900
    maturities = np.array([1/12, 0.25, 0.5, 1.0, 3.0])
    log_moneyness = np.linspace(-0.4, 0.4, 17)

    regime_names = ["carry", "crash", "event"]
    pattern = [("carry",100), ("crash",100), ("event",100),
               ("crash",50), ("carry",50), ("event",50)]

    blocks = []
    t0 = 0
    while t0 < T:
        for r, L in pattern:
            blocks.append((t0, min(t0+L-1, T-1), r))
            t0 += L
            if t0 >= T:
                break

    iv, true = generate_iv_surfaces(T, maturities, log_moneyness, blocks)
    paths = extract_2d_paths(iv, range(len(log_moneyness)), maturities)

    X = build_embeddings(paths, depth=3)
    X = StandardScaler().fit_transform(X)

    labels, prob, _ = run_hdbscan(X)
    true_int = np.array([regime_names.index(r) for r in true])
    pred_raw = map_clusters_to_regimes(labels, true_int)
    pred_smooth = sticky_filter(pred_raw, prob)

    visualize(X, labels, true_int, pred_raw, pred_smooth, prob, regime_names)
    _anim_ref = animate_iv_and_regimes(iv, log_moneyness, maturities, true_int, pred_smooth, prob, regime_names)
