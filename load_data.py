import torch
import torchtuples as tt
from torch.utils.data import Dataset
from pycox.models import LogisticHazard

class DatasetBatch(Dataset):  
    def __init__(self, image_dataset, time, event):
        self.image_dataset = image_dataset
        self.time, self.event = tt.tuplefy(time, event).to_tensor()
        
    def __len__(self):
        return len(self.time)

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
   
        img = [self.image_dataset[i]['image']  for i in index]
        img = torch.stack(img)
        
        return tt.tuplefy(img, (self.time[index], self.event[index]))
    
    
class DatasetBatch_Both(Dataset):  
    def __init__(self, image_dataset, time, event):
        self.image_dataset = image_dataset
        self.time, self.event = tt.tuplefy(time, event).to_tensor()
        
    def __len__(self):
        return len(self.time)

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        
        img = [torch.cat((self.image_dataset[i]['image'], self.image_dataset[i]['label']), 0)  for i in index]
        img = torch.stack(img)
        
        return tt.tuplefy(img, (self.time[index], self.event[index]))
    

def surv_times(Survival_df):
        
    df_train=Survival_df[0:40]
    df_val=Survival_df[40:50]
    df_test=Survival_df[50:]
    
    labtrans = LogisticHazard.label_transform(20)
    get_target = lambda Survival_df: (Survival_df['Survival_month'].values.astype(int), Survival_df['Status'].values.astype(int))
    target_train = labtrans.fit_transform(*get_target(df_train))
    target_val = labtrans.transform(*get_target(df_val))
    target_test = labtrans.transform(*get_target(df_test))
    
    # We don't need to transform the test labels
    durations_test, events_test = get_target(df_test)
    durations_train, events_train = get_target(df_train)
    
    return target_train, target_val, durations_test, events_test ,labtrans, durations_train, events_train, target_test
