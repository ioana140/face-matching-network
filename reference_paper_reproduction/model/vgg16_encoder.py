
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights

from config import (
    FC_NUM_LAYERS,
    FC_UNITS,
    VGG16_FREEZE_UP_TO,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
)

class VGG16Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # Drop the original classifier; 128x128 input -> features output (B, 512, 4, 4)
        self.features = vgg.features

        for i, layer in enumerate(self.features):
            for param in layer.parameters():
                param.requires_grad = (i >= VGG16_FREEZE_UP_TO)

        self._log_freeze_status()

        in_features = 512 * 4 * 4   # flattened block-5 output

        layers = []
        for _ in range(FC_NUM_LAYERS):
            layers += [nn.Linear(in_features, FC_UNITS), nn.ReLU(inplace=True)]
            in_features = FC_UNITS

        self.fc_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (B, 512, 4, 4)
        x = torch.flatten(x, 1)    # (B, 8192)
        x = self.fc_head(x)        # (B, 512)
        return x

    def _log_freeze_status(self) -> None:
        n_frozen    = sum(1 for p in self.features.parameters() if not p.requires_grad)
        n_trainable = sum(1 for p in self.features.parameters() if p.requires_grad)
        print(f"[vgg16_encoder] Frozen params    : {n_frozen}")
        print(f"[vgg16_encoder] Trainable params : {n_trainable}")
        print(f"[vgg16_encoder] Freeze boundary  : "
              f"features[0..{VGG16_FREEZE_UP_TO-1}] frozen, "
              f"features[{VGG16_FREEZE_UP_TO}..30] trainable (block 5)")


def build_encoder() -> VGG16Encoder:
    return VGG16Encoder()


if __name__ == "__main__":
    enc = build_encoder()
    x   = torch.zeros(2, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    out = enc(x)
    print(f"Encoder output shape: {out.shape}")
