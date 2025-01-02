import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from fastmri.models import Unet
from utils_train import fastmri_dataset, train_model, set_seed


def main():
    config = {
    'exp_name': "fold=8_chans=64",
    'data_dir': "/home/yusuf/Desktop/projects/data/fastmri/multicoil_train",
    'filenames_csv': "train_file_lists/train_files_AXT1POST.csv",
    'mask_type': "equispaced",
    'center_fractions': [0.08],
    'accelerations': [8],
    'batch_size' : 4,      #2
    'epochs' : 30,
    'lr' : 1e-3,
    'num_workers': 4
    }

    set_seed(42)    # implementation should be improved

    train_csv = pd.read_csv(config["filenames_csv"])
    data_list = train_csv.file_name.to_list()[:200]
    train_files, val_files = train_test_split(data_list, test_size=0.1, random_state=42)

    train_dataset = fastmri_dataset(config['data_dir'], train_files, config['mask_type'], config['center_fractions'], config['accelerations'])
    val_dataset = fastmri_dataset(config['data_dir'], val_files, config['mask_type'], config['center_fractions'], config['accelerations'])

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'],
        pin_memory=True
        )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],
        pin_memory=True
        )
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(in_chans=1, out_chans=1, chans=64, num_pool_layers=4).to(device)
    model = nn.DataParallel(model).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_metrics = train_model(model, train_loader, val_loader, optimizer, loss_fn, scheduler, config['epochs'], device, config['exp_name'])
    print("Best Metrics:", best_metrics)


if __name__=="__main__":
    main()
