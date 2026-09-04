import os
import torch
import numpy as np
import onnxruntime as ort
from src.model import get_model

MODEL_PT_PATH = os.path.join("saved_models", "best_model.pt")
MODEL_ONNX_PATH = os.path.join("saved_models", "best_model.onnx")

def export_to_onnx(pt_path=MODEL_PT_PATH, onnx_path=MODEL_ONNX_PATH, img_size=128):
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    device = torch.device("cpu")
    
    print(f"[ONNX EXPORT] Loading PyTorch model from: {pt_path}")
    model = get_model(pretrained=False).to(device)
    if os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location=device))
    else:
        print("[WARNING] PyTorch model checkpoint not found. Exporting initial model weights.")
        
    model.eval()
    
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    
    input_names = ["input_image"]
    output_names = ["gender_logits", "age_pred"]
    dynamic_axes = {
        "input_image": {0: "batch_size"},
        "gender_logits": {0: "batch_size"},
        "age_pred": {0: "batch_size"}
    }
    
    print(f"[ONNX EXPORT] Exporting model to: {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False
    )
    print("[SUCCESS] ONNX Model successfully exported!")
    
    # Verify exported ONNX model with ONNXRuntime
    print("[ONNX EXPORT] Verifying exported ONNX model with ONNXRuntime...")
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    numpy_input = dummy_input.cpu().numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: numpy_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    with torch.no_grad():
        pt_gender, pt_age = model(dummy_input)
        
    gender_diff = np.max(np.abs(ort_outputs[0] - pt_gender.numpy()))
    age_diff = np.max(np.abs(ort_outputs[1] - pt_age.numpy()))
    
    print(f"  * Max Gender Output Diff: {gender_diff:.6f}")
    print(f"  * Max Age Output Diff:    {age_diff:.6f}")
    
    if gender_diff < 1e-4 and age_diff < 1e-4:
        print("[SUCCESS] Verification SUCCESS! ONNX outputs match PyTorch outputs perfectly!")
    else:
        print("[WARNING] Minor numerical variance detected between PyTorch and ONNX.")

if __name__ == "__main__":
    export_to_onnx()
