"""Small sanity checks for mean pooling and SupCon losses."""
import torch

from model import SupervisedContrastiveLoss, mean_pool


def test_supcon_finite():
    embeddings = torch.randn(10, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    loss_fn = SupervisedContrastiveLoss(temperature=0.07, variant="normal")
    loss = loss_fn(embeddings, labels)
    assert torch.isfinite(loss), "Normal SupCon returned non-finite loss"


def test_weighted_supcon_finite():
    embeddings = torch.randn(10, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    loss_fn = SupervisedContrastiveLoss(temperature=0.07, variant="distance_weighted")
    loss = loss_fn(embeddings, labels)
    assert torch.isfinite(loss), "Weighted SupCon returned non-finite loss"


def test_mean_pool_mask():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    expected = torch.tensor([[2.0, 2.0]])
    assert torch.allclose(pooled, expected), f"Expected {expected}, got {pooled}"


def main():
    test_supcon_finite()
    test_weighted_supcon_finite()
    test_mean_pool_mask()
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
