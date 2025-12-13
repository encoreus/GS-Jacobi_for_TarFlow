import argparse
import builtins
import pathlib

import os
import time
import torch
import torchvision as tv
import GS_Jacobi_sampling
import utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
utils.set_random_seed(19)
import numpy as np
import random


## Calculate IGM and CRM

# IGM and CRM are calculated in the forward process of TarFlow, which need to pass a small batch of training set through the models

def main(args):
    dataset = args.dataset
    dataset1 = {'imagenet': 'imagenet', 'imagenet64_uncond': 'imagenet64','imagenet64_cond': 'imagenet', 'afhq': 'afhq','imagenet64_patch4':'imagenet'}[dataset]
    num_classes = {'imagenet': 1000, 'imagenet64_uncond': 0,'imagenet64_cond': 1000, 'afhq': 3, 'imagenet64_patch4': 1000}[dataset]
    img_size = {'imagenet': 128, 'imagenet64_uncond': 64, 'imagenet64_cond': 64, 'afhq': 256, 'imagenet64_patch4': 64}[dataset]
    patch_size = {'imagenet': 4, 'imagenet64_uncond': 2,'imagenet64_cond': 2, 'afhq': 8, 'imagenet64_patch4': 4}[dataset]
    channels = {'imagenet': 1024, 'imagenet64_uncond': 768, 'imagenet64_cond': 768,'afhq': 768, 'imagenet64_patch4': 1024}[dataset]
    noise_std = {'imagenet': 0.15, 'imagenet64_uncond': 0.05,'imagenet64_cond': 0.05, 'afhq': 0.07, 'imagenet64_patch4': 0.05}[dataset]
    channel_size = 3
    blocks = 8
    layers_per_block = 8

    # The paths of data and model parameters
    data_path = './data'
    if dataset == 'afhq':
        ckpt_file = './models/afhq_model_8_768_8_8_0.07.pth'
    elif dataset == 'imagenet':
        ckpt_file = './models/imagenet_model_4_1024_8_8_0.15.pth'
    elif dataset == 'imagenet64_uncond':
        ckpt_file = './models/imagenet64_model_2_768_8_8_0.05.pth'
    elif dataset == 'imagenet64_cond':
        ckpt_file = './models/imagenet_model_2_768_8_8_0.05.pth'
    elif dataset == 'imagenet64_patch4':
        ckpt_file ='./models/imagenet_model_4_1024_8_8_0.05.pth'

    device = 'cuda:0'

    batch_size = args.num_mc
    # set detect_mode = True, set the norm (1,2,'fro')
    model = GS_Jacobi_sampling.Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size, 
                channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
                num_classes=num_classes,detect_mode=True,norm=args.norm).to(device)
    model.load_state_dict(torch.load(ckpt_file))
    print('checkpoint loaded!')

    num_mc = args.num_mc
    # Random choose a small batch of images from the training set
    class RandomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None, max_samples=args.num_mc):
            self.root_dir = root_dir
            self.transform = transform
            self.max_samples = max_samples
            
            self.all_files = self._find_image_files(root_dir)
            self.sampled_files = random.sample(self.all_files, min(max_samples, len(self.all_files)))

        def _find_image_files(self, directory):
            image_extensions = ('.jpg', '.png', '.jpeg', '.JPEG')
            image_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_files.append(os.path.join(root, file))
            return image_files

        def __len__(self):
            return len(self.sampled_files)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.sampled_files[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

    transform = tv.transforms.Compose([
        tv.transforms.Resize(img_size),
        tv.transforms.CenterCrop(img_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5,), (0.5,)),
    ])

    start_time = time.time()
    dataset = RandomImageDataset(
        root_dir=os.path.join(data_path, dataset1),
        transform=transform,
        max_samples=num_mc
    )
    print(f"Total images in subset: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,     
        num_workers=8,
        pin_memory=True,
    )
    print(f"Data loading cost: {time.time() - start_time:.2f} seconds")

    images = next(iter(data_loader))
    images = images.to(device)
    y = None
    with torch.no_grad():
        z, outputs, logdets, IGN_list, CRN_list = model(images,y) 

    print('The Initial Guessing Metric:')
    print(IGN_list)
    print('The Convergence Ranking Metric:')
    print(CRN_list)
    CRN_percent = [crn[3] for crn in CRN_list]
    CRN_percent = np.array(CRN_percent)/sum(CRN_percent)*100
    print('The CRM percent:')
    print(CRN_percent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imagenet', type=str, choices=['imagenet', 'imagenet64_patch4', 'afhq', 'imagenet64_uncond'], help='Name of dataset')
    parser.add_argument('--norm',default=2,choices=[1,2,'fro'], help='The norm used to calculate IGM and CRM')
    parser.add_argument('--num_mc',default=1000,type=int,help='The number of images used to calculate')
    args = parser.parse_args()
    main(args)




