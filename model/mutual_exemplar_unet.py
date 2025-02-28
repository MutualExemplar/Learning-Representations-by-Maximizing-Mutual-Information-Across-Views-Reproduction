import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res2Net_v1b import res2net50_v1b_26w_4s  # Import Res2Net
from model.projector import Projector  # Import the updated Projector

class MutualExemplarUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(MutualExemplarUNet, self).__init__()

        # **🔹 Encoder (Pretrained Res2Net, Removing Fully Connected Layer)**
        res2net = res2net50_v1b_26w_4s(pretrained=True)
        self.encoder_first_layer = nn.Sequential(
            res2net.conv1,
            res2net.bn1,
            res2net.relu,
        )
        
        self.encoder = nn.Sequential(
            res2net.maxpool,  # ✅ Ensure maxpool is used before Res2Net layers
            res2net.layer1,  # 256 channels
            res2net.layer2,  # 512 channels
            res2net.layer3,  # 1024 channels
            res2net.layer4,  # 2048 channels
        )

        # **🔹 Decoder with Upsampling**
        decoder_out_channels = 64
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),

            nn.Conv2d(128, decoder_out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        # **🔹 Projection head for contrastive learning**
        self.projector = Projector(in_dim=2048, out_dim=decoder_out_channels//2)

        # **🔹 Segmentation Classifier**
        self.classifier = nn.Sequential(
            nn.Conv2d(decoder_out_channels, decoder_out_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_out_channels//2, num_classes, kernel_size=1),  # Output Segmentation Mask
        )

        # **🔹 Final Upsampling to Match Ground Truth**
        self.upsample_final = nn.Upsample(size=(512, 288), mode="bilinear", align_corners=True)

    def forward(self, x):
        # **🔹 Apply Encoder First Layer**
        x = self.encoder_first_layer(x)

        # **🔹 Pass through Encoder**
        x = self.encoder(x)  # ✅ Now maintains spatial dimensions

        # **🔹 Contrastive Features**
        features = self.projector(x)  # 🔹 Now correctly applied after encoder
        features = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-6)
         # Normalize embeddings

        # **🔹 Decode**
        x = self.decoder(x)

        # **🔹 Segmentation Output**
        segmentation = self.classifier(x)

        # ✅ **Final Upsampling for Output**
        segmentation = self.upsample_final(segmentation)
        
        if torch.isnan(segmentation).any() or torch.isnan(features).any():
            print("⚠ NaN detected in output! Skipping batch.")
            return torch.zeros_like(segmentation), torch.zeros_like(features)

        return segmentation, features  # Return segmentation mask + feature embeddings
