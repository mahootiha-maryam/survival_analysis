import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard
from pycox.evaluation import EvalSurv

df = pd.read_csv ('/home/mary/Documents/clinical_data.csv')

df_train=df[0:40]
df_val=df[40:50]
df_test=df[50:]


cols_std = ['MaxDoftumor (mm)', 'Age', 'Rec_Time', 'bmi' ] # numeric variables
cols_bin = ['Previousliverresection0No1yes', 'Chemowithin6monthspriorsurgery', 'Chemowithin4monthsaftersurgery', 'Multiplelesions1', 'Age67', 'Kjønn0mann1kvinne'] # binary variables


standardize = [([col], StandardScaler()) for col in cols_std]
leave = [(col, None) for col in cols_bin]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

num_durations = 10
labtrans = LogisticHazard.label_transform(num_durations)

get_target = lambda df: (df['Survivalday'].values.astype(int), df['Status(dead=1, alive=0)'].values.astype(int))
target_train = labtrans.fit_transform(*get_target(df_train))
target_val = labtrans.transform(*get_target(df_val))


train = (x_train, target_train)
val = (x_val, target_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)


in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
batch_size = 4

# lrfind = model.lr_finder(x_train, target_train, batch_size, tolerance=50)
# print(lrfind.get_best_lr())

epochs = 500

callbacks = [tt.cb.EarlyStopping(patience=10)]

log = model.fit(x_train, target_train, batch_size, epochs,callbacks, val_data=val)

_ = log.plot()

surv = model.predict_surv_df(x_test)

surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

surv = model.interpolate(10).predict_surv_df(x_test)

surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

print(ev.concordance_td('antolini'))

time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
# ev.brier_score(time_grid).plot()
# plt.ylabel('Brier score')
# _ = plt.xlabel('Time')

# ev.nbll(time_grid).plot()
# plt.ylabel('NBLL')
# _ = plt.xlabel('Time')

print(ev.integrated_brier_score(time_grid) )

print(ev.integrated_nbll(time_grid) )