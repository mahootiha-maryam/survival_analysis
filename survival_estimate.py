from preprocess_monai import preprocess,drawing_plot
from making_dataset import surv_times, mixed_dataset
from train_torch import train_survmodel
import pandas as pd
import torchtuples as tt
from torch.utils.data import DataLoader


# =============================================================================
# loading images folders that contains training, valdating and testing folders
# =============================================================================
data_dir = '/home/mary/Documents/Anonimized_Images/cropped'

# =============================================================================
# preprocess the images to train the model by an appropriate and normalized dataset
# plot the images after preprocessing to see the images before training
# =============================================================================
train_ds,train_loader,orig_ds,orig_loader,val_ds,val_loader,test_ds,test_loader = preprocess(data_dir)
drawing_plot(train_loader, orig_loader, val_loader)

# =============================================================================
# loading the files that contain survival times and events. Change duration times
# to multiple specified time intervals (classes)
# =============================================================================
df = pd.read_csv ('/home/mary/Downloads/survival_Comet.csv')
target_train,target_val, target_test,labtrans = surv_times(df)

# =============================================================================
# make a mixed dataset that has images as x and duration+events as y
# =============================================================================
dataset_train,dataset_val,dataset_test= mixed_dataset(train_ds, target_train, val_ds, target_val, test_ds, target_test)

# =============================================================================
# making dataloaders from the mixed dataset
# =============================================================================
batch_size =10
def collate_fn(batch):
    """Stacks the entries of a nested tuple"""
    return tt.tuplefy(batch).stack()
dl_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=collate_fn)
dl_val = DataLoader(dataset_val, batch_size, shuffle=True, collate_fn=collate_fn)
dl_test = DataLoader(dataset_test, batch_size, shuffle=False, collate_fn=collate_fn)


train_survmodel(labtrans,dl_train,dl_val)