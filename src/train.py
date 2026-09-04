import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.model import get_model
from src.dataset import parse_utkface_dir, get_transforms, UTKFaceDataset
from sklearn.model_selection import train_test_split

SAVED_MODELS_DIR = "saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pt")

def train_one_epoch_chunk(model, dataloader, criterion_gender, criterion_age, optimizer, device, gender_weight=1.0, age_weight=1.0):
    model.train()
    running_loss = 0.0
    running_gender_correct = 0
    running_age_mae = 0.0
    total_samples = 0
    
    for images, ages, genders in dataloader:
        images = images.to(device)
        ages = ages.unsqueeze(1).to(device)
        genders = genders.unsqueeze(1).to(device)
        
        ages_scaled = ages / 100.0
        
        optimizer.zero_grad()
        
        gender_logits, age_preds = model(images)
        loss_gender = criterion_gender(gender_logits, genders)
        loss_age = criterion_age(age_preds, ages_scaled)
        loss = gender_weight * loss_gender + age_weight * loss_age
        
        loss.backward()
        optimizer.step()
            
        batch_size = images.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size
        
        preds_gender = (torch.sigmoid(gender_logits) > 0.5).float()
        running_gender_correct += (preds_gender == genders).sum().item()
        
        pred_ages_unscaled = age_preds * 100.0
        running_age_mae += torch.abs(pred_ages_unscaled - ages).sum().item()
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_gender_acc = (running_gender_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    epoch_age_mae = running_age_mae / total_samples if total_samples > 0 else 0.0
    
    return epoch_loss, epoch_gender_acc, epoch_age_mae

def evaluate(model, dataloader, criterion_gender, criterion_age, device, gender_weight=1.0, age_weight=1.0):
    model.eval()
    running_loss = 0.0
    running_gender_correct = 0
    running_age_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, ages, genders in dataloader:
            images = images.to(device)
            ages = ages.unsqueeze(1).to(device)
            genders = genders.unsqueeze(1).to(device)
            
            ages_scaled = ages / 100.0
            
            gender_logits, age_preds = model(images)
            loss_gender = criterion_gender(gender_logits, genders)
            loss_age = criterion_age(age_preds, ages_scaled)
            loss = gender_weight * loss_gender + age_weight * loss_age
            
            batch_size = images.size(0)
            total_samples += batch_size
            running_loss += loss.item() * batch_size
            
            preds_gender = (torch.sigmoid(gender_logits) > 0.5).float()
            running_gender_correct += (preds_gender == genders).sum().item()
            
            pred_ages_unscaled = age_preds * 100.0
            running_age_mae += torch.abs(pred_ages_unscaled - ages).sum().item()
            
    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    val_gender_acc = (running_gender_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    val_age_mae = running_age_mae / total_samples if total_samples > 0 else 0.0
    
    return val_loss, val_gender_acc, val_age_mae

def train_additional_epochs(data_dir=r"C:\Users\ashid\Documents\all_utkface", additional_epochs=15, chunk_size=4000, batch_size=128, img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("EFFICIENTNET-B0 HIGH-ACCURACY FINE-TUNING ENGINE")
    print("=" * 65)
    print(f"Device:           {device.type.upper()}")
    print(f"Total Dataset:    23,708 Images")
    print(f"Chunk Size/Epoch: {chunk_size} Images (~{chunk_size // batch_size} Batches/Epoch)")
    print(f"Additional Epochs:{additional_epochs}")
    print("=" * 65 + "\n")
    
    file_paths, ages, genders = parse_utkface_dir(data_dir)
    
    X_train, X_temp, y_age_train, y_age_temp, y_gen_train, y_gen_temp = train_test_split(
        file_paths, ages, genders, test_size=0.2, random_state=42
    )
    X_val, X_test, y_age_val, y_age_test, y_gen_val, y_gen_test = train_test_split(
        X_temp, y_age_temp, y_gen_temp, test_size=0.5, random_state=42
    )
    
    train_tf, val_tf = get_transforms(img_size=img_size)
    full_train_dataset = UTKFaceDataset(X_train, y_age_train, y_gen_train, transform=train_tf)
    val_dataset = UTKFaceDataset(X_val, y_age_val, y_gen_val, transform=val_tf)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = get_model(pretrained=True, backbone_name="efficientnet_b0").to(device)
    
    criterion_gender = nn.BCEWithLogitsLoss()
    criterion_age = nn.L1Loss()
    
    # Unfreeze deep EfficientNet features for maximum accuracy
    print("--- Unfreezing EfficientNet-B0 Feature Layers for High-Accuracy Fine-Tuning ---")
    for param in model.features[-6:].parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=additional_epochs, eta_min=1e-5)
    
    best_val_loss = float('inf')
    best_gender_acc = 0.0
    best_age_mae = 99.0
    
    start_time = time.time()
    num_train_samples = len(full_train_dataset)
    
    for epoch in range(1, additional_epochs + 1):
        t0 = time.time()
        chunk_indices = random.sample(range(num_train_samples), min(chunk_size, num_train_samples))
        chunk_subset = Subset(full_train_dataset, chunk_indices)
        train_loader = DataLoader(chunk_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        train_loss, train_gen_acc, train_age_mae = train_one_epoch_chunk(
            model, train_loader, criterion_gender, criterion_age, optimizer, device
        )
        val_loss, val_gen_acc, val_age_mae = evaluate(
            model, val_loader, criterion_gender, criterion_age, device
        )
        
        scheduler.step()
        elapsed = time.strftime("%M:%S", time.gmtime(time.time() - t0))
        
        print(f"Epoch {epoch:02d}/{additional_epochs:02d} [{elapsed}] | "
              f"Train Loss: {train_loss:.4f} (Gen: {train_gen_acc:.2f}%, Age MAE: {train_age_mae:.2f}y) | "
              f"Val Loss: {val_loss:.4f} (Gen: {val_gen_acc:.2f}%, Age MAE: {val_age_mae:.2f}y)")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_gender_acc = val_gen_acc
            best_age_mae = val_age_mae
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   --> New Best Checkpoint saved to {MODEL_SAVE_PATH} (Val Loss: {val_loss:.4f})")
            
    total_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print("\n" + "=" * 65)
    print(f"EFFICIENTNET-B0 FINE-TUNING COMPLETE IN {total_time}")
    print(f"   * Best Gender Accuracy: {best_gender_acc:.2f}%")
    print(f"   * Best Age MAE:         {best_age_mae:.2f} years")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    train_additional_epochs(additional_epochs=15, chunk_size=4000)
