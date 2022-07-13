import os
from glob import glob
from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd, 
    Resized,
    Orientationd)


def image_transform(data_dir):
    """
    We have to do three things:
        1-load images 
        2-do the transforms
        3-convert them to tensors
    """
    train_images = sorted(glob(os.path.join(data_dir, 'M1', '*.nii.gz')))
    train_labels = sorted(glob(os.path.join(data_dir, 'L1', '*.nii.gz')))
    
    val_images = sorted(glob(os.path.join(data_dir, 'M2', '*.nii.gz')))
    val_labels = sorted(glob(os.path.join(data_dir, 'L2', '*.nii.gz')))
    
    test_images = sorted(glob(os.path.join(data_dir, 'M3', '*.nii.gz')))
    test_labels = sorted(glob(os.path.join(data_dir, 'L3', '*.nii.gz')))
    
    train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
    val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]
    test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]

    
    orig_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),    
            ToTensord(keys=['image', 'label'])
        ]
    )
    
    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            #add channels to the images that monai can interpret in this way
            AddChanneld(keys=['image', 'label']),
            #rescale the voxels pixdim=(height,width,depth) 
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
            Orientationd(keys=['image', 'label'], axcodes="RAS"),
            ScaleIntensityRanged(keys='image', a_min=-100, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 128]),
            #to tensor should be the last transform
            ToTensord(keys=['image', 'label'])
        ]
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
            Orientationd(keys=['image', 'label'], axcodes="RAS"),
            ScaleIntensityRanged(keys='image', a_min=-100, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 128]),
            ToTensord(keys=['image', 'label'])
        ]
    )
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    
    return train_ds, val_ds, test_ds


