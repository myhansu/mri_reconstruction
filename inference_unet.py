import pandas as pd
import os
import torch
from torch.utils.data import DataLoader
from torch import nn
from fastmri.models import Unet
from utils_train import fastmri_dataset
from utils_inf import load_model, run_inference


def main():
    
    config = {
    'model_path': "checkpoints/fold4/epoch=28_valloss=0.037321543939930354.pt",
    'filenames_csv': "test_file_lists/test_filesAXFLAIR.csv",
    'data_dir': "/home/yusuf/Desktop/projects/data/fastmri/multicoil_val",
    'mask_type': "equispaced",
    'center_fractions': [0.08],
    'accelerations': [4],
    'batch_size' : 1,      #2
}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4).to(device)
    model = nn.DataParallel(model).to(device)

    model = load_model(config["model_path"], model)

    test_csv = pd.read_csv(config["filenames_csv"])
    test_files = test_csv.file_name.to_list()

    test_files = os.listdir(config['data_dir'])[:10]
    test_dataset = fastmri_dataset(config['data_dir'], test_files, config['mask_type'], config['center_fractions'], config['accelerations'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    output_dir = os.path.join("reconstructions",config["model_path"].split("/")[1])
    run_inference(model, test_loader, device, output_dir)


if __name__ == "__main__":
    main()