import torch
import torch.nn as nn
import torchvision.models as models

class AgeGenderModel(nn.Module):
    """
    Multi-task Deep Convolutional Neural Network for simultaneous Age Regression
    and Gender Binary Classification based on MobileNetV2 architecture.
    """
    def __init__(self, pretrained=True):
        super(AgeGenderModel, self).__init__()
        
        # Load MobileNetV2 backbone
        if hasattr(models, 'MobileNet_V2_Weights'):
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            mobilenet = models.mobilenet_v2(weights=weights)
        else:
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            
        self.features = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        in_features = mobilenet.last_channel # 1280
        
        # Shared feature projection layer
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.35)
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

def get_model(pretrained=True):
    return AgeGenderModel(pretrained=pretrained)

if __name__ == "__main__":
    model = get_model(pretrained=False)
    dummy_input = torch.randn(4, 3, 200, 200)
    gender_out, age_out = model(dummy_input)
    print("Model loaded successfully!")
    print("Gender output shape:", gender_out.shape)
    print("Age output shape:", age_out.shape)
