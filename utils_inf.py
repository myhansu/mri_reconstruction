import os
import torch
import numpy as np
from tqdm import tqdm
import h5py

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from utils_train import initialize_metrics


def load_model(path, model):
    checkpoint=torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def run_inference(model, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    psnr, ssim = initialize_metrics(device)

    ssim_scores = []
    psnr_scores = []

    ssim.reset()
    psnr.reset()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Processing test data")):
            inputs = inputs.unsqueeze(1).to(device)
            targets = targets.unsqueeze(1).to(device)
            outputs = model(inputs)

            ssim_score = ssim(outputs, targets).cpu().item()
            psnr_score = psnr(outputs, targets).cpu().item()

            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)

            for j,recon in enumerate(outputs):
                output_path = os.path.join(output_dir, f"slice_{i}_{j}.h5")
                with h5py.File(output_path, "w") as hf:
                    hf.create_dataset("reconstruction", data=recon.cpu())

    mean_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)
    mean_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    
    metrics_df = pd.DataFrame({
        'ssim': ssim_scores,
        'psnr': psnr_scores
    })
    
    # Save summary statistics
    summary_df = pd.DataFrame({
        'metric': ['SSIM', 'PSNR'],
        'mean': [mean_ssim, mean_psnr],
        'std': [std_ssim, std_psnr]
    })

    metrics_df.to_csv(os.path.join(output_dir, "metrics_single.csv"))
    summary_df.to_csv(os.path.join(output_dir, "metrics_mean.csv"))
    
    test_id = output_dir.split("/")[1]
    print(f"{test_id}")
    print(f"SSIM: Mean = {mean_ssim:.4f}, Std = {std_ssim:.4f}")
    print(f"PSNR: Mean = {mean_psnr:.4f}, Std = {std_psnr:.4f}")