import os
from glob import glob

from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    AddChanneld,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    Orientationd

)

from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.utils import first
import matplotlib.pyplot as plt



def preprocess(data_dir):
    
    train_images = sorted(glob(os.path.join(data_dir, 'M1', '*.nii.gz')))
    train_labels = sorted(glob(os.path.join(data_dir, 'L1', '*.nii.gz')))
    
    val_images = sorted(glob(os.path.join(data_dir, 'M2', '*.nii.gz')))
    val_labels = sorted(glob(os.path.join(data_dir, 'L2', '*.nii.gz')))
    
    test_images = sorted(glob(os.path.join(data_dir, 'M3', '*.nii.gz')))
    test_labels = sorted(glob(os.path.join(data_dir, 'L3', '*.nii.gz')))
    
    #make a dataframe which contains images and labels
    
    train_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(train_images, train_labels)]
    
    val_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(val_images, val_labels)]
    
    test_files = [{"image": image_name, 'label': label_name} for image_name, label_name in zip(test_images, test_labels)]
    
    """
    We have to do three things:
        1-load images 
        2-do the transforms
        3-convert them to tensors
    """
    
    #the images without changes
    orig_transforms = Compose(
    
        [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),    
            ToTensord(keys=['image', 'label'])
        ]
    )
    
    # =============================================================================
    # transforms:
    #     
    # Spacingd: This function will assist us in changing the voxel dimensions because
    # we don’t know if a dataset of medical images was acquired with the same scan or
    # with different scans, so they may have different voxel dimensions (width, height, depth),
    # so we need to generalize all of them to the same dimensions.
    # 
    # ScalIntensityRanged: This function will assist us in performing two tasks at the
    # same time: the first will be to change the contrast from that dense vision into
    # something more visible, and the second will be to normalize the voxel values and
    # place them between 0 and 1 so that the training will be faster.
    # 
    # CropForegroundd: This function will assist us in cropping out the empty regions
    # of the image that we do not require, leaving only the region of interest.
    # 
    # Resized: Finally, this function is optional, but in my opinion, it is required 
    # if you used the cropforeground function, because the function that will do the 
    # crop will have the output with random dimensions depending on each patient, so 
    # if we do not add an operation to give the same dimensions to all patients, our 
    # model will not work.
    # =============================================================================
    
    # =============================================================================
    # in the ScaleIntensityRange we only applied it to the image because we don’t need
    # to change the intensity or normalize the labels’ values.
    # =============================================================================
    
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
            Resized(keys=['image', 'label'], spatial_size=[128,128,128]),
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
            ToTensord(keys=['image', 'label'])
        ]
    )
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, collate_fn=pad_list_data_collate)
    
    orig_ds = Dataset(data=train_files, transform=orig_transforms)
    orig_loader = DataLoader(orig_ds, batch_size=1, collate_fn=pad_list_data_collate)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=pad_list_data_collate)
    
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, collate_fn=pad_list_data_collate)
    
    return train_ds,train_loader,orig_ds,orig_loader,val_ds,val_loader,test_ds,test_loader
    
    '''
    drawing the plot of first item of train loader and orig loader
    '''
def drawing_plot(train_loader,orig_loader,val_loader):    
    test_patient = first(train_loader)
    orig_patient = first(orig_loader)
    
    plt.figure('test', (12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title('Orig patient')
    #[number of batches, number of channels, height, width, Slices] these are the 
    #argomans of image
    plt.imshow(orig_patient['image'][0, 0, : ,: ,82], cmap= "gray")
    
    plt.subplot(1, 3, 2)
    plt.title('Slice of a patient')
    #[number of batches, number of channels, height, width, Slices] these are the 
    #argomans of image
    plt.imshow(test_patient['image'][0, 0, : ,: ,82], cmap= "gray")
    
    plt.subplot(1, 3, 3)
    plt.title('Slice of a label')
    #[number of batches, number of channels, height, width, Slices] these are the 
    #argomans of image
    plt.imshow(test_patient['label'][0, 0, : ,: ,82], cmap= "gray")


    '''
    drawing different batches slices instead of first item
    '''
    
    # train_loader_show = DataLoader(train_ds, batch_size=40, collate_fn=pad_list_data_collate)
    # train_batches= next(iter(train_loader_show))
    
    # orig_loader_show = DataLoader(orig_ds, batch_size=40, collate_fn=pad_list_data_collate)
    # orig_batches= next(iter(orig_loader_show))
    
    # plt.figure('test', (12, 6))
    
    # plt.subplot(1, 3, 1)
    # plt.title('Orig patient')
    # #[number of batches, number of channels, height, width, Slices] these are the 
    # #argomans of image
    # plt.imshow(orig_batches['image'][38, 0, : ,: ,120], cmap= "gray")
    
    # plt.subplot(1, 3, 2)
    # plt.title('Slice of a patient')
    # #[number of batches, number of channels, height, width, Slices] these are the 
    # #argomans of image
    # plt.imshow(train_batches['image'][38, 0, : ,: ,120], cmap= "gray")
    
    # plt.subplot(1, 3, 3)
    # plt.title('Slice of a label')
    # #[number of batches, number of channels, height, width, Slices] these are the 
    # #argomans of image
    # plt.imshow(train_batches['label'][38, 0, : ,: ,120], cmap= "gray")