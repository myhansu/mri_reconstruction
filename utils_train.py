import os
import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random

from fastmri.data.subsample import create_mask_for_mask_type 
from fastmri.data.transforms import to_tensor, center_crop, apply_mask, normalize, normalize_instance
from fastmri import ifft2c, complex_abs, rss
from tqdm import tqdm
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class fastmri_dataset(Dataset):
    def __init__(self, data_dir, data_list, mask_type, center_fractions, accelerations, crop_size=(320,320), transform=None):
        self.data_dir = data_dir
        self.file_paths = [os.path.join(data_dir, fname) for fname in data_list if fname.endswith('.h5')]
        self.transform = transform
        self.mask_type = mask_type
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.crop_size = crop_size
        self.slice_info = []

        # Pre-compute mask function to avoid recreation in __getitem__
        self.mask_func = create_mask_for_mask_type(
        mask_type_str=self.mask_type,
        center_fractions=self.center_fractions,
        accelerations=self.accelerations
)

        # Precompute slice indexing to avoid iterating over files at every __getitem__
        for file_idx, file_path in enumerate(self.file_paths):
            with h5py.File(file_path, 'r') as f:
                num_slices = f['kspace'].shape[0]
                self.slice_info.extend([(file_idx, slice_idx) for slice_idx in range(num_slices)])


    def __len__(self):
        return len(self.slice_info)


    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_info[idx]
        file_path = self.file_paths[file_idx]
        
        with h5py.File(file_path, 'r') as f:
            kspace = to_tensor(f['kspace'][slice_idx])
            target = to_tensor(f['reconstruction_rss'][slice_idx])

            masked_kspace, mask, _ = apply_mask(kspace.unsqueeze(0), self.mask_func)
            masked_kspace = masked_kspace.squeeze(0)

            undersampled_image = complex_abs(ifft2c(masked_kspace))
            undersampled_image_cropped = center_crop(undersampled_image, self.crop_size)
            undersampled_image_rss = rss(undersampled_image_cropped, dim=0)

            # normalize input
            undersampled_image_rss, mean, std = normalize_instance(undersampled_image_rss, eps=1e-11)
            undersampled_image_rss = undersampled_image_rss.clamp(-6, 6)

            # normalize target
            if target is not None:
                target = center_crop(target, self.crop_size)
                target = normalize(target, mean, std, eps=1e-11)
                target = target.clamp(-6, 6)


        if self.transform:
            undersampled_image_rss, target = self.transform(undersampled_image_rss, target)

        return undersampled_image_rss.float(), target.float()


class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, save_path="checkpoints"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float("inf") #instead of None
        self.counter = 0
        self.early_stop = False

        os.makedirs(save_path, exist_ok=True)

    def __call__(self, val_loss, epoch, model, exp_name):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            os.makedirs((os.path.join(self.save_path,exp_name)), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict()
            }, os.path.join(os.path.join(self.save_path,exp_name),f"epoch={epoch}_valloss={self.best_loss}.pt"))
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def initialize_metrics(device):
    psnr = PeakSignalNoiseRatio().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    return psnr, ssim


def train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, epochs, device,exp_name):

    scaler = GradScaler()
    writer = SummaryWriter(log_dir="./logs")
    early_stopping = EarlyStopping()
    best_metrics = {"psnr":0, "ssim":0}


    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0

        torch.cuda.empty_cache()

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            #ensuring inputs and targets are properly distributed across both gpus
            inputs, targets = inputs.unsqueeze(1).to(device, non_blocking=True), targets.unsqueeze(1).to(device, non_blocking=True)

            with autocast():  # Mixed precision training
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # delete unnecessary tensors
            del outputs
            del loss

        avg_train_loss = train_loss / len(train_loader)

        # Validation Phase
        model.eval()

        psnr, ssim = initialize_metrics(device)
        val_metrics = {'loss': 0, 'psnr': 0, 'ssim': 0}
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"\t - Validation"):
                inputs, targets = inputs.unsqueeze(1).to(device), targets.unsqueeze(1).to(device)
                outputs = model(inputs)

                val_metrics['loss'] += loss_fn(outputs, targets).item()
                val_metrics['psnr'] += psnr(outputs, targets).item()
                val_metrics['ssim'] += ssim(outputs, targets).item()

                # delete unnecessary tensors
                del outputs

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        scheduler.step(val_metrics['loss'])

        # Log losses
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_metrics['loss'], epoch)
        writer.add_scalar("Metrics/PSNR", val_metrics['psnr'], epoch)
        writer.add_scalar("Metrics/SSIM", val_metrics['ssim'], epoch)

        # Update best metrics
        if val_metrics['psnr'] > best_metrics['psnr']:
            best_metrics['psnr'] = val_metrics['psnr']
        if val_metrics['ssim'] > best_metrics['ssim']:
            best_metrics['ssim'] = val_metrics['ssim']

        print(f"\t  - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, Val PSNR: {val_metrics['psnr']:.4f}, Val SSIM: {val_metrics['ssim']:.4f}")
        
        # Early stopping
        early_stopping(val_metrics['loss'], epoch+1, model,exp_name)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        torch.cuda.empty_cache()

    writer.close()
    return best_metrics
