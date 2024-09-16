import os
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

class NeRFDataset(Dataset):
    def __init__(self, file_path="../data/tiny_nerf_data.npz", use_cuda=True) -> None:
        super().__init__()
        data = np.load(os.path.join(os.path.dirname(__file__),file_path))
        if use_cuda:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.images = torch.from_numpy(data['images']).float()
        self.poses = torch.from_numpy(data['poses']).float()
        self.focal = torch.tensor(data['focal'], dtype=torch.float32)

    def __getitem__(self, index, view=False):
        img = self.images[index].to(self.device)
        pose = self.poses[index].to(self.device)
        if view :
            print(f"ID : {index} \npose : {pose}\nfocal: {self.focal}")
            plt.imshow(img.cpu().numpy())
            plt.axis('off')
            plt.show()

        return img, pose, self.focal
    
    def __len__(self):
        return len(self.images)
    
    def get_data(self):
        return self.images, self.poses, self.focal
    

if __name__ == "__main__":
    dataset = NeRFDataset()
    img, pose, focal = dataset.__getitem__(5, view=False)
    # compute_rays(img, pose, focal, 0, 5,10,torch.device('cpu'))


