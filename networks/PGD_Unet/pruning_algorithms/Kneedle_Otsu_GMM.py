import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class PruneResult:
    layer_name: str
    method: str
    threshold: float
    num_channels: int
    num_keep: int
    num_pruned: int
    prune_ratio: float
    keep_indices: np.ndarray
    keep_mask: np.ndarray
    scores: np.ndarray


def _safe_numpy_1d(scores) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    if len(scores) == 0:
        raise ValueError("scores must not be empty")
    return scores


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


def _topk_keep_mask(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, len(scores)))
    idx = np.argsort(scores)[::-1][:k]
    mask = np.zeros(len(scores), dtype=bool)
    mask[idx] = True
    return mask


def static_prune_by_ratio(scores: np.ndarray, prune_ratio: float) -> np.ndarray:
    """
    Keep the top-k channels by score for a fixed pruning ratio.

    If prune_ratio = r, the number of kept channels is:
        ceil((1 - r) * num_channels)
    """
    scores = _safe_numpy_1d(scores)
    if not 0.0 <= float(prune_ratio) < 1.0:
        raise ValueError("prune_ratio must be in [0, 1).")

    num_channels = len(scores)
    num_keep = max(1, int(np.ceil((1.0 - float(prune_ratio)) * num_channels)))
    return _topk_keep_mask(scores, num_keep)


def apply_min_channel_constraint(
    scores: np.ndarray,
    mask: np.ndarray,
    min_keep_ratio: float = 0.4,
    min_keep_channels: Optional[int] = None,
) -> np.ndarray:
    total = len(scores)

    if min_keep_channels is None:
        min_keep_channels = int(np.ceil(total * min_keep_ratio))

    min_keep_channels = max(1, min(total, min_keep_channels))

    if mask.sum() < min_keep_channels:
        return _topk_keep_mask(scores, min_keep_channels)

    return mask


def kneedle_threshold(scores: np.ndarray) -> float:
    """
    Manual Kneedle-style threshold:
    1. sort ascending
    2. normalize x and y to [0,1]
    3. find point maximizing (y - x)
    """
    scores = _safe_numpy_1d(scores)
    sorted_scores = np.sort(scores)

    if len(sorted_scores) <= 2:
        return float(np.median(sorted_scores))

    x = np.linspace(0.0, 1.0, len(sorted_scores))
    y = _normalize_minmax(sorted_scores)

    diff = y - x
    knee_idx = int(np.argmax(diff))
    tau = float(sorted_scores[knee_idx])
    return tau


def otsu_threshold(scores: np.ndarray, num_bins: int = 256) -> float:
    """
    Otsu threshold on 1D scores.
    """
    scores = _safe_numpy_1d(scores)

    if np.allclose(scores, scores[0]):
        return float(scores[0])

    hist, bin_edges = np.histogram(scores, bins=num_bins)
    hist = hist.astype(np.float64)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = hist.sum()

    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    numerator = (mu_t * omega - mu) ** 2
    denominator = omega * (1.0 - omega)
    denominator[denominator == 0] = 1e-12

    sigma_b2 = numerator / denominator
    idx = int(np.argmax(sigma_b2))
    tau = float(bin_centers[idx])
    return tau


def gmm_threshold(scores: np.ndarray, random_state: int = 42) -> float:
    """
    Fit 2-component GMM and approximate decision threshold.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for GMM. Install with: pip install scikit-learn")

    scores = _safe_numpy_1d(scores)

    if np.allclose(scores, scores[0]):
        return float(scores[0])

    x = scores.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=random_state)
    gmm.fit(x)

    means = gmm.means_.reshape(-1)
    covs = gmm.covariances_.reshape(-1)
    weights = gmm.weights_.reshape(-1)

    order = np.argsort(means)
    mu1, mu2 = means[order[0]], means[order[1]]
    var1, var2 = covs[order[0]], covs[order[1]]
    w1, w2 = weights[order[0]], weights[order[1]]

    sigma1 = np.sqrt(max(var1, 1e-12))
    sigma2 = np.sqrt(max(var2, 1e-12))

    # Solve intersection of two weighted Gaussians:
    # w1*N(x|mu1,s1) = w2*N(x|mu2,s2)
    a = 1.0 / (2 * var2) - 1.0 / (2 * var1)
    b = mu1 / var1 - mu2 / var2
    c = (mu2 ** 2) / (2 * var2) - (mu1 ** 2) / (2 * var1) + np.log((w2 * sigma1) / (w1 * sigma2))

    if abs(a) < 1e-12:
        # fallback to linear case
        if abs(b) < 1e-12:
            tau = 0.5 * (mu1 + mu2)
        else:
            tau = -c / b
        return float(tau)

    roots = np.roots([a, b, c])
    roots = np.real(roots[np.isreal(roots)])

    # Choose root lying between the two means if possible
    valid = roots[(roots >= mu1) & (roots <= mu2)]
    if len(valid) > 0:
        return float(valid[0])

    # fallback
    return float((mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2))


def build_prune_result(
    layer_name: str,
    method: str,
    scores: np.ndarray,
    threshold: float,
    min_keep_ratio: float = 0.4,
    min_keep_channels: Optional[int] = None,
) -> PruneResult:
    scores = _safe_numpy_1d(scores)

    keep_mask = scores >= threshold
    keep_mask = apply_min_channel_constraint(
        scores=scores,
        mask=keep_mask,
        min_keep_ratio=min_keep_ratio,
        min_keep_channels=min_keep_channels,
    )
    return build_prune_result_from_mask(
        layer_name=layer_name,
        method=method,
        scores=scores,
        keep_mask=keep_mask,
        threshold=threshold,
    )


def build_prune_result_from_mask(
    layer_name: str,
    method: str,
    scores: np.ndarray,
    keep_mask: np.ndarray,
    threshold: Optional[float] = None,
) -> PruneResult:
    scores = _safe_numpy_1d(scores)
    keep_mask = np.asarray(keep_mask, dtype=bool).reshape(-1)
    if len(keep_mask) != len(scores):
        raise ValueError("keep_mask must have the same length as scores")
    if not keep_mask.any():
        raise ValueError("keep_mask must keep at least one channel")

    if threshold is None:
        threshold = float(scores[keep_mask].min())

    keep_indices = np.where(keep_mask)[0]
    num_channels = len(scores)
    num_keep = int(keep_mask.sum())
    num_pruned = num_channels - num_keep
    prune_ratio = num_pruned / num_channels

    return PruneResult(
        layer_name=layer_name,
        method=method,
        threshold=float(threshold),
        num_channels=num_channels,
        num_keep=num_keep,
        num_pruned=num_pruned,
        prune_ratio=float(prune_ratio),
        keep_indices=keep_indices,
        keep_mask=keep_mask,
        scores=scores,
    )


def prune_one_layer(
    scores: np.ndarray,
    layer_name: str,
    method: str = "kneedle",
    min_keep_ratio: float = 0.4,
    min_keep_channels: Optional[int] = None,
    static_prune_ratio: Optional[float] = None,
    otsu_bins: int = 256,
    gmm_random_state: int = 42,
) -> PruneResult:
    scores = _safe_numpy_1d(scores)
    method = method.lower()

    if method in {"static", "middle_static"}:
        if static_prune_ratio is None:
            raise ValueError(f"static_prune_ratio is required when method='{method}'")
        keep_mask = static_prune_by_ratio(scores, static_prune_ratio)
        tau = float(scores[keep_mask].min())
        return build_prune_result_from_mask(
            layer_name=layer_name,
            method=method,
            scores=scores,
            keep_mask=keep_mask,
            threshold=tau,
        )
    elif method == "kneedle":
        tau = kneedle_threshold(scores)
    elif method == "otsu":
        tau = otsu_threshold(scores, num_bins=otsu_bins)
    elif method == "gmm":
        tau = gmm_threshold(scores, random_state=gmm_random_state)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return build_prune_result(
        layer_name=layer_name,
        method=method,
        scores=scores,
        threshold=tau,
        min_keep_ratio=min_keep_ratio,
        min_keep_channels=min_keep_channels,
    )


def prune_all_layers(
    layer_scores: Dict[str, np.ndarray],
    method: str = "kneedle",
    default_min_keep_ratio: float = 0.4,
    layer_min_keep_ratio: Optional[Dict[str, float]] = None,
    layer_min_keep_channels: Optional[Dict[str, int]] = None,
    static_prune_ratio: Optional[float] = None,
    otsu_bins: int = 256,
    gmm_random_state: int = 42,
) -> Dict[str, PruneResult]:
    results = {}

    for layer_name, scores in layer_scores.items():
        min_ratio = default_min_keep_ratio
        if layer_min_keep_ratio is not None and layer_name in layer_min_keep_ratio:
            min_ratio = layer_min_keep_ratio[layer_name]

        min_ch = None
        if layer_min_keep_channels is not None and layer_name in layer_min_keep_channels:
            min_ch = layer_min_keep_channels[layer_name]

        result = prune_one_layer(
            scores=scores,
            layer_name=layer_name,
            method=method,
            min_keep_ratio=min_ratio,
            min_keep_channels=min_ch,
            static_prune_ratio=static_prune_ratio,
            otsu_bins=otsu_bins,
            gmm_random_state=gmm_random_state,
        )
        results[layer_name] = result

    return results


def print_prune_summary(results: Dict[str, PruneResult]) -> None:
    print("=" * 100)
    print(f"{'Layer':30s} {'Method':10s} {'#Ch':>6s} {'Keep':>6s} {'Prune':>6s} {'Ratio':>8s} {'Tau':>12s}")
    print("=" * 100)
    for layer_name, r in results.items():
        print(
            f"{layer_name:30s} {r.method:10s} "
            f"{r.num_channels:6d} {r.num_keep:6d} {r.num_pruned:6d} "
            f"{r.prune_ratio:8.3f} {r.threshold:12.6f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    # Ví dụ giả lập importance scores cho 3 layer
    np.random.seed(123)

    layer_scores = {
        "encoder.block1.conv": np.abs(np.random.normal(0.05, 0.02, 32)),
        "encoder.block2.conv": np.abs(np.concatenate([
            np.random.normal(0.03, 0.01, 40),
            np.random.normal(0.20, 0.05, 24)
        ])),
        "decoder.block3.conv": np.abs(np.concatenate([
            np.random.normal(0.02, 0.01, 50),
            np.random.normal(0.15, 0.04, 14)
        ])),
    }

    layer_min_keep_ratio = {
        "encoder.block1.conv": 0.7,   # early layer giữ nhiều hơn
        "encoder.block2.conv": 0.4,
        "decoder.block3.conv": 0.6,   # deep/decoder giữ cẩn thận hơn
    }

    for method in ["static", "middle_static", "kneedle", "otsu", "gmm"]:
        print(f"\nMETHOD = {method.upper()}")
        results = prune_all_layers(
            layer_scores=layer_scores,
            method=method,
            static_prune_ratio=0.5 if method in {"static", "middle_static"} else None,
            default_min_keep_ratio=0.4,
            layer_min_keep_ratio=layer_min_keep_ratio,
        )
        print_prune_summary(results)

        # lấy mask của 1 layer
        sample_layer = "encoder.block2.conv"
        print(f"\nKeep indices for {sample_layer}:")
        print(results[sample_layer].keep_indices)
