"""EchoProof Research — Neural Model Architecture for Deepfake Detection.

Features:
- Wav2Vec2 self-supervised backbone with configurable freeze depth.
- Temporal pooling (Attentive Statistical Pooling / Mean-Std Pooling).
- Binary classification head with dropout and LayerNorm.
- Focal loss and class-weighted cross entropy for handling spoof/bonafide imbalance.
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config


class AttentiveStatisticalPooling(nn.Module):
    """
    Attentive Statistical Pooling (ASP) calculates attention-weighted mean and standard deviation
    over time frames, capturing subtle temporal artifacts left by neural voice vocoders.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, time_steps, in_dim)
        weights = self.attention(x)  # (batch_size, time_steps, 1)
        mean = torch.sum(weights * x, dim=1)  # (batch_size, in_dim)
        residuals = x - mean.unsqueeze(1)
        variance = torch.sum(weights * (residuals ** 2), dim=1)
        std = torch.sqrt(torch.clamp(variance, min=1e-8))
        return torch.cat([mean, std], dim=1)  # (batch_size, 2 * in_dim)


class EchoProofClassifier(nn.Module):
    """
    EchoProof deepfake detection network combining self-supervised audio representations
    with attentive temporal pooling and classification projections.
    """

    def __init__(
        self,
        backbone_id: str = "facebook/wav2vec2-base",
        freeze_backbone_layers: int = 8,
        pooling: str = "asp",
        dropout: float = 0.3,
        num_classes: int = 2,
        backbone_config: Optional[dict] = None,
    ):
        super().__init__()
        self.backbone_id = backbone_id
        if backbone_config is None:
            self.backbone = Wav2Vec2Model.from_pretrained(backbone_id)
        else:
            self.backbone = Wav2Vec2Model(Wav2Vec2Config.from_dict(backbone_config))

        # Freeze lower feature extractor / CNN layers
        self.backbone.feature_extractor._freeze_parameters()

        # Optionally freeze lower transformer encoder layers to prevent overfitting
        if freeze_backbone_layers > 0 and hasattr(self.backbone.encoder, "layers"):
            for layer in self.backbone.encoder.layers[:freeze_backbone_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        if pooling == "asp":
            self.pool = AttentiveStatisticalPooling(hidden_size, hidden_dim=128)
            pooled_dim = hidden_size * 2
        else:
            self.pool = None
            pooled_dim = hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        waveforms: (batch_size, num_samples) at 16kHz
        returns: (batch_size, num_classes) logits [bonafide_logit, spoof_logit]
        """
        outputs = self.backbone(waveforms)
        hidden_states = outputs.last_hidden_state  # (B, T, D)

        if self.pool is not None:
            pooled = self.pool(hidden_states)  # (B, 2*D)
        else:
            pooled = torch.mean(hidden_states, dim=1)  # (B, D)

        logits = self.classifier(pooled)
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss to focus learning on hard, ambiguous speech deepfake examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
            focal_loss = alpha_weight * focal_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
