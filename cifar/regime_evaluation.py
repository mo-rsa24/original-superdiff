import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from cifar.models.experts import SmallMNISTClassifier


def train_mnist_classifier(mnist_train, device, epochs=2, bs=256, lr=1e-3):
    clf = SmallMNISTClassifier().to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    dl = DataLoader(mnist_train, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)

    clf.train()
    for ep in range(epochs):
        for i, (x,y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"classifier epoch {ep+1}/{epochs} done")
    return clf

@torch.no_grad()
def existence_scores_48(x48, clf, digit: int, device):
    if x48.min() < 0:
        x48 = (x48 + 1) / 2
    x48 = x48.clamp(0, 1)

    probs = []
    # Slide 28x28 window over 48x48 image
    for top in range(0, 48-28+1):
        for left in range(0, 48-28+1):
            patch = x48[:, :, top:top+28, left:left+28].to(device)
            p = torch.softmax(clf(patch), dim=1)[:, digit]
            probs.append(p)
    probs = torch.stack(probs, dim=1)  # [B, 21*21]
    return probs.max(dim=1).values

@torch.no_grad()
def eval_existential(x, clf, device):
    s4 = existence_scores_48(x, clf, 4, device).detach().cpu().numpy()
    s7 = existence_scores_48(x, clf, 7, device).detach().cpu().numpy()
    other_digits = [d for d in range(10) if d not in (4, 7)]
    other_scores = []
    for d in other_digits:
        other_scores.append(existence_scores_48(x, clf, d, device).detach().cpu().numpy())
    max_other = np.max(np.stack(other_scores, axis=0), axis=0)
    return {
        "mean_exists4": float(s4.mean()),
        "mean_exists7": float(s7.mean()),
        "mean_exists_both_proxy": float(np.minimum(s4, s7).mean()),
        "mean_exists_other_max": float(max_other.mean()),
        "mean_exclusivity": float((1.0 - max_other).mean()),
    }

@torch.no_grad()
def eval_only_digits(x, clf, device, target_digits=(4, 7)):
    target_digits = set(int(d) for d in target_digits)
    scores = {d: existence_scores_48(x, clf, d, device) for d in target_digits}
    s_target = torch.stack([scores[d] for d in sorted(target_digits)], dim=1)
    s4 = scores.get(4, torch.zeros_like(s_target[:, 0]))
    s7 = scores.get(7, torch.zeros_like(s_target[:, 0]))
    other_digits = [d for d in range(10) if d not in target_digits]
    other_scores = torch.stack([existence_scores_48(x, clf, d, device) for d in other_digits], dim=1)
    max_other = other_scores.max(dim=1).values
    return {
        "mean_exists4": float(s4.mean().item()),
        "mean_exists7": float(s7.mean().item()),
        "mean_exists_both_proxy": float(torch.minimum(s4, s7).mean().item()),
        "mean_max_other_digit": float(max_other.mean().item()),
        "mean_only_47_proxy": float((1.0 - max_other).mean().item()),
    }