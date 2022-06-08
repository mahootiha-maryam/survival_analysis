
from pycox.models import LogisticHazard
import torchtuples as tt
from torch.utils.data import Dataset


def surv_times(df):
        
    df_train=df[0:40]
    df_val=df[40:50]
    df_test=df[50:]
    
    labtrans = LogisticHazard.label_transform(20)
    get_target = lambda df: (df['Survival'].values.astype(int), df['Event'].values.astype(int))
    target_train = labtrans.fit_transform(*get_target(df_train))
    target_val = labtrans.transform(*get_target(df_val))
    target_test = labtrans.transform(*get_target(df_test))
    
    return target_train,target_val, target_test,labtrans
    

def mixed_dataset(train_ds,target_train,val_ds,target_val,test_ds,target_test):

    '''
    Creating a Custom Dataset for your files
    _____________________________________________________________________________
    The __init__ function is run once when instantiating the Dataset object
    The __len__ function returns the number of samples in our dataset
    The __getitem__ function loads and returns a sample from the dataset at the 
    given index. Returns the tensor image and corresponding label in a tuple
    _____________________________________________________________________________
    This class takes the images as x, time duration and events as y(label)
    Then return all of them as a dataset with inheritance from Dataset class
    '''
    
    class DatasetSingle(Dataset):
    
        def __init__(self, image_dataset, time, event):
            self.image_dataset = image_dataset
            self.time, self.event = tt.tuplefy(time, event).to_tensor()
    
        def __len__(self):
            return len(self.image_dataset)
    
        def __getitem__(self, index):
            if type(index) is not int:
                raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")
            #extract the image pixels from the dictionary of dataset
            img = self.image_dataset[index]['image']
            return img, (self.time[index], self.event[index])
        
        
    dataset_train = DatasetSingle(train_ds, *target_train)
    dataset_val = DatasetSingle(val_ds, *target_val)
    dataset_test = DatasetSingle(test_ds, *target_test)
    
    return dataset_train,dataset_val,dataset_test
