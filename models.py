import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class TeacherModel(nn.Module):
    def __init__(self, num_sectors=12):
        super().__init__()
        self.slit_encoder = Encoder()
        self.axial_encoder = Encoder()

        self.head = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_sectors)
        )

    def forward(self, slit, axial):
        s = self.slit_encoder(slit)
        a = self.axial_encoder(axial)
        fused = torch.cat([s, a], dim=1)
        return self.head(fused)


class StudentModel(nn.Module):
    def __init__(self, num_sectors=12):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_sectors)
        )

    def forward(self, slit):
        feat = self.encoder(slit)
        return self.head(feat)
