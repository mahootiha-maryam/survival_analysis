import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import Encoder_FC 
from load_data import DatasetBatch, surv_times
from my_transforms import image_transform
from Eval import plot_survival, evaluate_survmodel
import torch
from pycox.models import LogisticHazard
import torchtuples as tt
from pycox.evaluation import EvalSurv

# for reproducability
#np.random.seed(1234)
#_ = torch.manual_seed(1234)
     

Survival_df = pd.read_csv ('/home/mary/Downloads/survival_Comet(1).csv')

target_train, target_val, durations_test, events_test, labtrans, durations_train, events_train, target_test = surv_times(Survival_df)

data_dir = '/home/mary/Documents/Anonimized_Images/cropped'
train_ds, val_ds, test_ds = image_transform(data_dir)

dataset_train = DatasetBatch(train_ds, *target_train)
dataset_val = DatasetBatch(val_ds, *target_val)
dataset_test = DatasetBatch(test_ds, *target_test)

batch_size = 1

dl_train = tt.data.DataLoaderBatch(dataset_train, batch_size, shuffle=True)
dl_val = tt.data.DataLoaderBatch(dataset_val, batch_size, shuffle=False)
dl_test = tt.data.DataLoaderBatch(dataset_test, batch_size, shuffle=False)

mynet = Encoder_FC(num_channels=1, out_features=labtrans.out_features)

mymodel = LogisticHazard(mynet, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

#callbacks = [tt.cb.EarlyStopping(patience=5)]
epochs = 100

log = mymodel.my_fit_dataloader(dl_train, epochs, save_epoch=10, path="run", callbacks=None, verbose=True, val_dataloader=dl_val)

loaded_test = tt.data.dataloader_input_only(dl_test)
loaded_train = tt.data.dataloader_input_only(dl_train)

surv_image = mymodel.predict_surv_df(loaded_test)
surv_image_interp = mymodel.interpolate(20).predict_surv_df(loaded_test)

surv_train_image = mymodel.predict_surv_df(loaded_train)
surv_train_image_interp = mymodel.interpolate(20).predict_surv_df(loaded_train)

plot_survival(surv_image, surv_train_image, surv_image_interp, surv_train_image_interp)

evaluate_survmodel(surv_image_interp, surv_train_image_interp, durations_test, events_test, durations_train, events_train)
