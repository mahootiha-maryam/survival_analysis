import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import Encoder_FC 
from load_data import DatasetBatch_Both, surv_times
from my_transforms import image_transform
from Eval import plot_survival, evaluate_survmodel
import torch
from pycox.models import LogisticHazard
import torchtuples as tt


Survival_df = pd.read_csv ('/home/mary/Downloads/survival_Comet(1).csv')

target_train, target_val, durations_test, events_test, labtrans, durations_train, events_train, target_test = surv_times(Survival_df)

data_dir = '/home/mary/Documents/Anonimized_Images/cropped'
train_ds, val_ds, test_ds = image_transform(data_dir)

dataset_train = DatasetBatch_Both(train_ds, *target_train)
dataset_val = DatasetBatch_Both(val_ds, *target_val)
dataset_test = DatasetBatch_Both(test_ds, *target_test)

batch_size = 1

dl_train1 = tt.data.DataLoaderBatch(dataset_train, batch_size, shuffle=True)
dl_val1 = tt.data.DataLoaderBatch(dataset_val, batch_size, shuffle=False)
dl_test1 = tt.data.DataLoaderBatch(dataset_test, batch_size, shuffle=False)

mynet1 = Encoder_FC(num_channels=2, out_features=labtrans.out_features)

mymodel1 = LogisticHazard(mynet1, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

#callbacks = [tt.cb.EarlyStopping(patience=5)]
epochs = 100

log = mymodel1.my_fit_dataloader(dl_train1, epochs, save_epoch=10, path="run_both", callbacks=None, verbose=True, val_dataloader=dl_val1)

loaded_test1 = tt.data.dataloader_input_only(dl_test1)
loaded_train1 = tt.data.dataloader_input_only(dl_train1)


surv_image1 = mymodel1.predict_surv_df(loaded_test1)
surv_image_interp1 = mymodel1.interpolate(20).predict_surv_df(loaded_test1)

surv_train_image1 = mymodel1.predict_surv_df(loaded_train1)
surv_train_image_interp1 = mymodel1.interpolate(20).predict_surv_df(loaded_train1)

plot_survival(surv_image1, surv_train_image1, surv_image_interp1, surv_train_image_interp1)

evaluate_survmodel(surv_image_interp1, surv_train_image_interp1, durations_test, events_test, durations_train, events_train)
