"""Models with Wing loss - minimal version."""
import torch
import torch.nn as nn
import torchvision.models as models
import math


class WingLoss(nn.Module):
    """Wing loss - better for small errors in landmark detection."""

    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = omega - omega * math.log(1 + omega / epsilon)

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )
        return loss.mean()


def get_loss_function(name):
    losses = {
        'l1': nn.L1Loss(),
        'smoothl1': nn.SmoothL1Loss(),
        'wing': WingLoss(),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}")
    return losses[name]


class PartLocalizer(nn.Module):
    def __init__(self, backbone='resnet50', num_parts=3, config=None):
        super().__init__()
        self.num_parts = num_parts

        hidden_dim = getattr(config, 'HIDDEN_DIM', 512)
        dropout = getattr(config, 'DROPOUT', 0.4)

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'densenet':
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.coord_head = nn.Linear(hidden_dim // 2, num_parts * 2)
        self.vis_head = nn.Linear(hidden_dim // 2, num_parts)

        nn.init.xavier_uniform_(self.coord_head.weight, gain=0.01)
        nn.init.constant_(self.coord_head.bias, 0.5)

    def forward(self, x):
        features = self.head(self.backbone(x))
        coords = torch.sigmoid(self.coord_head(features)).view(-1, self.num_parts, 2)
        vis = torch.sigmoid(self.vis_head(features))
        return coords, vis