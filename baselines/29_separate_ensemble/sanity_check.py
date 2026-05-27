"""Small sanity checks for folder 29 losses, medoids, and ensemble rules."""
import numpy as np
import torch

from ensemble import (
    DEFAULT_TAUS,
    ENSEMBLE_STRATEGIES,
    class_prior,
    ensemble_distributions,
    medoid_distribution,
    median_decode_probs_np,
)
from model import SupervisedContrastiveLoss, emd_loss


def test_emd_loss_finite():
    """EMD^2 loss should return a finite scalar for classifier logits."""
    logits = torch.randn(12, 5)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 4])
    loss = emd_loss(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_supcon_variants_finite():
    """Both SupCon variants should return finite losses on paired labels."""
    embeddings = torch.randn(10, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    for variant in ("normal", "distance_weighted"):
        loss = SupervisedContrastiveLoss(temperature=0.07, variant=variant)(embeddings, labels)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_medoid_distributions_sum_to_one():
    """Medoid probabilities should be valid distributions for every tau."""
    z_train = torch.randn(20, 16)
    y_train = torch.tensor([0, 1, 2, 3, 4] * 4)
    z_query = torch.randn(7, 16)
    for tau in DEFAULT_TAUS:
        probs = medoid_distribution(z_train, y_train, z_query, tau=tau, device=torch.device("cpu"))
        assert probs.shape == (7, 5)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_ensemble_strategies_valid():
    """All six ensemble strategies should decode to labels in 0..4."""
    class_probs = np.array(
        [
            [0.70, 0.20, 0.05, 0.03, 0.02],
            [0.05, 0.10, 0.20, 0.55, 0.10],
            [0.02, 0.08, 0.15, 0.25, 0.50],
        ],
        dtype=np.float32,
    )
    retrieval_probs = np.array(
        [
            [0.40, 0.35, 0.15, 0.05, 0.05],
            [0.05, 0.20, 0.35, 0.30, 0.10],
            [0.05, 0.05, 0.10, 0.30, 0.50],
        ],
        dtype=np.float32,
    )
    prior = np.array([0.40, 0.25, 0.20, 0.10, 0.05], dtype=np.float32)
    outputs = ensemble_distributions(class_probs, retrieval_probs, prior)
    assert set(outputs) == set(ENSEMBLE_STRATEGIES)
    for probs in outputs.values():
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        preds = median_decode_probs_np(probs)
        assert np.all((0 <= preds) & (preds <= 4))


def test_prior_corrected_handles_nonuniform_prior():
    """Prior-corrected POE should remain normalized with skewed class priors."""
    labels = [0] * 50 + [1] * 20 + [2] * 15 + [3] * 10 + [4] * 5
    prior = class_prior(labels)
    class_probs = np.full((4, 5), 0.2, dtype=np.float32)
    retrieval_probs = np.full((4, 5), 0.2, dtype=np.float32)
    probs = ensemble_distributions(class_probs, retrieval_probs, prior)["poe_prior_corrected"]
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(np.isfinite(probs))


def main():
    """Run all sanity checks from a plain Python entry point."""
    test_emd_loss_finite()
    test_supcon_variants_finite()
    test_medoid_distributions_sum_to_one()
    test_ensemble_strategies_valid()
    test_prior_corrected_handles_nonuniform_prior()
    print("Folder 29 sanity checks passed.")


if __name__ == "__main__":
    main()
