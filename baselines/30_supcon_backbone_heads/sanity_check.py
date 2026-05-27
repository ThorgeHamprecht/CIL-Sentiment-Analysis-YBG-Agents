"""Small checks for folder 30 losses and decoders."""
import torch

from model import (
    CORALHead,
    SupervisedContrastiveLoss,
    coral_decode,
    coral_loss,
    regression_decode,
    regression_loss,
)


def test_supcon_finite():
    """SupCon should return a finite scalar on paired class labels."""
    embeddings = torch.randn(10, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    loss = SupervisedContrastiveLoss(temperature=0.07, variant="normal")(embeddings, labels)
    assert torch.isfinite(loss)


def test_coral_head_loss_and_decode():
    """The frozen ordinal CORAL head should train/decode on pooled features."""
    head = CORALHead(hidden_size=8)
    features = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 4, 2])
    logits = head(features)
    loss = coral_loss(logits, labels)
    preds = coral_decode(logits)
    assert logits.shape == (6, 4)
    assert torch.isfinite(loss)
    assert torch.all((0 <= preds) & (preds <= 4))


def test_regression_loss_and_decode():
    """Scalar regression predictions should round/clamp to valid labels."""
    preds = torch.tensor([-1.2, 0.49, 1.51, 3.6, 9.0])
    labels = torch.tensor([0, 0, 2, 4, 4])
    loss = regression_loss(preds, labels)
    decoded = regression_decode(preds)
    assert torch.isfinite(loss)
    assert decoded.tolist() == [0, 0, 2, 4, 4]


def main():
    """Run all folder 30 sanity checks."""
    test_supcon_finite()
    test_coral_head_loss_and_decode()
    test_regression_loss_and_decode()
    print("Folder 30 sanity checks passed.")


if __name__ == "__main__":
    main()
