import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Default ImageNet normalization standards
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size=200):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    return train_transform, val_transform

class UTKFaceDataset(Dataset):
    """
    Custom PyTorch Dataset for UTKFace dataset.
    Filename format: [age]_[gender]_[race]_[date].jpg.chip.jpg
    """
    def __init__(self, file_paths, ages, genders, transform=None):
        self.file_paths = file_paths
        self.ages = ages
        self.genders = genders
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.float32)
        
        return image, age, gender

def parse_utkface_dir(data_dir):
    file_paths, ages, genders = [], [], []
    valid_exts = ('.jpg', '.png', '.jpeg')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
        
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(valid_exts):
            continue
        parts = fname.split('_')
        if len(parts) >= 3:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                if 0 <= age <= 116 and gender in (0, 1):
                    file_paths.append(os.path.join(data_dir, fname))
                    ages.append(age)
                    genders.append(gender)
            except ValueError:
                continue
                
    return file_paths, ages, genders

def get_dataloaders(data_dir=r"C:\Users\ashid\Documents\all_utkface", batch_size=64, img_size=128, num_workers=2):
    file_paths, ages, genders = parse_utkface_dir(data_dir)
    print(f"[DATASET] Total valid images found: {len(file_paths)}")
    
    # Train 80%, Val 10%, Test 10%
    X_train, X_temp, y_age_train, y_age_temp, y_gen_train, y_gen_temp = train_test_split(
        file_paths, ages, genders, test_size=0.2, random_state=42
    )
    
    X_val, X_test, y_age_val, y_age_test, y_gen_val, y_gen_test = train_test_split(
        X_temp, y_age_temp, y_gen_temp, test_size=0.5, random_state=42
    )
    
    train_tf, val_tf = get_transforms(img_size=img_size)
    
    train_dataset = UTKFaceDataset(X_train, y_age_train, y_gen_train, transform=train_tf)
    val_dataset = UTKFaceDataset(X_val, y_age_val, y_gen_val, transform=val_tf)
    test_dataset = UTKFaceDataset(X_test, y_age_test, y_gen_test, transform=val_tf)
    
    # Set persistent workers if num_workers > 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    for images, ages, genders in train_loader:
        print("Batch images shape:", images.shape)
        print("Batch ages shape:", ages.shape)
        print("Batch genders shape:", genders.shape)
        break
