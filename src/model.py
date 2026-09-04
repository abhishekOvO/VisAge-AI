import torch
import torch.nn as nn
import torchvision.models as models

class AgeGenderModel(nn.Module):
    """
    Multi-task Deep Convolutional Neural Network for simultaneous Age Regression
    and Gender Binary Classification based on EfficientNet-B0 architecture.
    """
    def __init__(self, pretrained=True, backbone_name="efficientnet_b0"):
        super(AgeGenderModel, self).__init__()
        
        self.backbone_name = backbone_name
        
        if backbone_name == "efficientnet_b0":
            if hasattr(models, 'EfficientNet_B0_Weights'):
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                backbone = models.efficientnet_b0(weights=weights)
            else:
                backbone = models.efficientnet_b0(pretrained=pretrained)
            self.features = backbone.features
            in_features = 1280
        else:
            if hasattr(models, 'MobileNet_V2_Weights'):
                weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
                backbone = models.mobilenet_v2(weights=weights)
            else:
                backbone = models.mobilenet_v2(pretrained=pretrained)
            self.features = backbone.features
            in_features = backbone.last_channel

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Shared feature projection layer
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3)
        )
        
        # Gender Classification Head (0 = Male, 1 = Female)
        self.gender_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Age Regression Head
        self.age_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        shared = self.shared_fc(x)
        
        gender_logit = self.gender_head(shared)
        age = self.age_head(shared)
        
        return gender_logit, age

def get_model(pretrained=True, backbone_name="efficientnet_b0"):
    return AgeGenderModel(pretrained=pretrained, backbone_name=backbone_name)

if __name__ == "__main__":
    model = get_model(pretrained=False)
    dummy_input = torch.randn(4, 3, 200, 200)
    gender_out, age_out = model(dummy_input)
    print("EfficientNet-B0 Model loaded successfully!")
    print("Gender output shape:", gender_out.shape)
    print("Age output shape:", age_out.shape)
