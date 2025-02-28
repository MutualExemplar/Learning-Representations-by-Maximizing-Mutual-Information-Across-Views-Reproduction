import os
import sys
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import transforms
import numpy as np
import torch
import random
from gridmask import GridMask  # Ensure GridMask is imported correctly


class ObjDataset(data.Dataset):
    def __init__(self, images, gts, trainsize):
        self.trainsize = trainsize
        self.images = sorted(images)
        self.gts = sorted(gts)
        self.filter_files()
        self.size = len(self.images)
       # print(f"ObjDataset {self.size = }")
        self.gridmask = GridMask()

        # **Define Augmentations for Three Networks (F1, F2, F3)**
        # crop images instead of resizing
        self.img_transform_F1 = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.img_transform_F2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(3),
            transforms.Resize((self.trainsize)),                       
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.img_transform_F3 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.GaussianBlur(5),
            transforms.Resize((self.trainsize)),            
            transforms.ToTensor(),
            self.gridmask, 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        #print(f"Before transforms: image type = {type(image)}")  # Debugging

        # Ensure image is a PIL image before applying transforms
        if isinstance(image, tuple):
            image = image[0]
            
        image_F1 = self.img_transform_F1(image)
        image_F2 = self.img_transform_F2(image)
        image_F3 = self.img_transform_F3(image)

        gt = self.gt_transform(gt)


        return image_F1, image_F2, image_F3, gt


    def filter_files(self):
        """Ensure image-mask pairs are valid."""
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


class ValObjDataset(data.Dataset):
    """Validation Dataset (No augmentations)."""
    def __init__(self, images, gts, trainsize):
        self.trainsize = trainsize
        self.images = sorted(images)
        self.gts = sorted(gts)
        self.filter_files()
        self.size = len(self.images)
        #print(f"ValDataset {self.size = }")

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.ToTensor()
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def __len__(self):
        return self.size


def image_loader(image_root, gt_root, val_img_root, val_gt_root, batch_size, image_size, split=0.8, labeled_ratio=0.05):
    """
    Loads datasets and applies Tri-view augmentations.
    """
    images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')])
    gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')])
    
    val_images = sorted([os.path.join(val_img_root, f) for f in os.listdir(val_img_root) if f.endswith('.png') or f.endswith('.jpg')])
    val_labels = sorted([os.path.join(val_gt_root, f) for f in os.listdir(val_gt_root) if f.endswith('.png') or f.endswith('.jpg')])
    
    if len(val_images) != len(images):
        print("val_images does not have the same size as test images", file=sys.stderr)
    
    train_images, val_images = images[:int(len(images) * split)], val_images[int(len(val_images) * split):]
    train_gts, val_gts = gts[:int(len(gts) * split)], val_labels[int(len(val_labels) * split):]

    labeled_train_images = train_images[:int(len(train_images) * labeled_ratio)]
    unlabeled_train_images = train_images[int(len(train_images) * labeled_ratio):]

    labeled_train_gts = train_gts[:int(len(train_gts) * labeled_ratio)]
    unlabeled_train_gts = train_gts[int(len(train_gts) * labeled_ratio):]
    
    # print(f"{len(labeled_train_images) = }")
    # print(f"{len(unlabeled_train_images) = }")
    # print(f"{len(labeled_train_gts) = }")
    # print(f"{len(unlabeled_train_gts) = }")
    
    labeled_train_dataset = ObjDataset(labeled_train_images, labeled_train_gts, image_size)
    unlabeled_train_dataset = ObjDataset(unlabeled_train_images, unlabeled_train_gts, image_size)
    val_dataset = ValObjDataset(val_images, val_gts, image_size)

    labeled_data_loader = data.DataLoader(dataset=labeled_train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    unlabeled_data_loader = data.DataLoader(dataset=unlabeled_train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)

    return labeled_data_loader, unlabeled_data_loader, val_loader