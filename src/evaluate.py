import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from src.model import get_model
from src.dataset import get_dataloaders

MODEL_PATH = os.path.join("saved_models", "best_model.pt")

def evaluate_model(data_dir=r"C:\Users\ashid\Documents\all_utkface", model_path=MODEL_PATH, batch_size=32, img_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVALUATE] Using device: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
        
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size, img_size=img_size)
    
    model = get_model(pretrained=False, backbone_name="efficientnet_b0").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_gender_true = []
    all_gender_pred_prob = []
    all_age_true = []
    all_age_pred = []
    
    with torch.no_grad():
        for images, ages, genders in test_loader:
            images = images.to(device)
            gender_logits, age_preds = model(images)
            
            probs = torch.sigmoid(gender_logits).cpu().numpy().flatten()
            age_predictions = age_preds.cpu().numpy().flatten() * 100.0
            
            all_gender_true.extend(genders.numpy())
            all_gender_pred_prob.extend(probs)
            all_age_true.extend(ages.numpy())
            all_age_pred.extend(age_predictions)
            
    all_gender_true = np.array(all_gender_true)
    all_gender_pred_prob = np.array(all_gender_pred_prob)
    all_gender_pred_bin = (all_gender_pred_prob > 0.5).astype(int)
    
    all_age_true = np.array(all_age_true)
    all_age_pred = np.array(all_age_pred)
    
    # Calculate Gender Metrics
    gender_acc = accuracy_score(all_gender_true, all_gender_pred_bin) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_gender_true, all_gender_pred_bin, average='binary')
    auc = roc_auc_score(all_gender_true, all_gender_pred_prob)
    
    # Calculate Age Metrics
    age_mae = mean_absolute_error(all_age_true, all_age_pred)
    age_rmse = np.sqrt(mean_squared_error(all_age_true, all_age_pred))
    age_r2 = r2_score(all_age_true, all_age_pred)
    
    print("\n" + "="*50)
    print(" MODEL PERFORMANCE EVALUATION REPORT")
    print("="*50)
    print(f"Total Test Samples Evaluated: {len(all_gender_true)}")
    print("-" * 50)
    print(" GENDER CLASSIFICATION METRICS:")
    print(f"  * Accuracy:        {gender_acc:.2f}%")
    print(f"  * Precision:       {precision:.4f}")
    print(f"  * Recall:          {recall:.4f}")
    print(f"  * F1 Score:        {f1:.4f}")
    print(f"  * ROC AUC:         {auc:.4f}")
    print("-" * 50)
    print(" AGE REGRESSION METRICS:")
    print(f"  * Mean Abs Error:  {age_mae:.2f} years")
    print(f"  * Root MSE (RMSE): {age_rmse:.2f} years")
    print(f"  * R2 Score:        {age_r2:.4f}")
    print("="*50 + "\n")
    
    return {
        "gender_accuracy": gender_acc,
        "gender_f1": f1,
        "age_mae": age_mae,
        "age_rmse": age_rmse,
        "age_r2": age_r2
    }

if __name__ == "__main__":
    evaluate_model()
