"""
Unconditioned prior over tool designs. Sampling from this prior will give you random tools.
Right now it is independent factors of l1, l2, theta sampled from uniform distributions.
We will extend towards more expressive and complex priors in the future.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tool_design_prior import ToolDesignPrior


class DummyToolDataset(Dataset):
    def __init__(self,tool_csv_file, transform=None, target_transform=None) :
        self.tools = pd.read_csv(tool_csv_file)
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.tools)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        pass
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = decode_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label


def main():
    pass

    
if __name__ == "__main__":
    main()