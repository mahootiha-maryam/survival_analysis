import os

import pandas as pd
import matplotlib.pyplot as plt
from models import Encoder_FC  
from load_data import DatasetBatch, surv_times, DatasetBatch_Both
from my_transforms import image_transform
import torch
from pycox.models import LogisticHazard
import torchtuples as tt


Survival_df = pd.read_csv ('/home/mary/Downloads/survival_Comet(1).csv')

target_train, target_val, durations_test, events_test, labtrans, durations_train, events_train, target_test = surv_times(Survival_df)

data_dir = '/home/mary/Documents/Anonimized_Images/cropped'
train_ds, val_ds, test_ds = image_transform(data_dir)

dataset_train = DatasetBatch(train_ds, *target_train)
dataset_val = DatasetBatch(val_ds, *target_val)
dataset_test = DatasetBatch(test_ds, *target_test)

batch_size = 1

dl_train2 = tt.data.DataLoaderBatch(dataset_train, batch_size, shuffle=True)
dl_val2 = tt.data.DataLoaderBatch(dataset_val, batch_size, shuffle=False)
dl_test2 = tt.data.DataLoaderBatch(dataset_test, batch_size, shuffle=False)

net_test = Encoder_FC(num_channels=1,out_features=labtrans.out_features)

model_test = LogisticHazard(net_test, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

model_test.load_model_weights("model_weights_10.pt")

loaded_test2 = tt.data.dataloader_input_only(dl_test2)

surv_image2 = model_test.predict_surv_df(loaded_test2)

print(surv_image2)

# surv.iloc[:, 0:5].plot(drawstyle='steps-post')
# plt.title('discrete survival times for test dataset')
# plt.ylabel('S(t | x)')
# _ = plt.xlabel('Time')

# plt.show()
