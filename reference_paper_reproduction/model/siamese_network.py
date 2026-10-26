

import torch
import torch.nn as nn

from model.distance_layer import EuclideanDistance
from model.vgg16_encoder  import VGG16Encoder, build_encoder


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = build_encoder()
        self.distance = EuclideanDistance()

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        f_a = self.encoder(img_a)   # (B, 512)
        f_b = self.encoder(img_b)   # (B, 512)
        d   = self.distance(f_a, f_b)  # (B,)
        return d

    def get_embedding(self, img: torch.Tensor) -> torch.Tensor:
        return self.encoder(img)


if __name__ == "__main__":
    from config import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH

    model = SiameseNetwork()
    a = torch.zeros(4, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    b = torch.zeros(4, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    d = model(a, b)
    print(f"Output distances shape : {d.shape}")
    print(f"Sample distances       : {d.detach().numpy()}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters       : {total_params:,}")
