import matplotlib.pyplot as plt
import numpy as np
from pycox.evaluation import EvalSurv
import sys
 


def plot_survival(surv_image, surv_train_image, surv_image_interp, surv_train_image_interp):

    surv_image.iloc[:, 0:5].plot(drawstyle='steps-post')
    plt.title('discrete survival times for test dataset')
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')
    plt.savefig('plot_D_Test.png', dpi=300, bbox_inches='tight')
    
    surv_train_image.iloc[:, 0:5].plot(drawstyle='steps-post')
    plt.title('discrete survival times for train dataset')
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')
    plt.savefig('plot_D_Train.png', dpi=300, bbox_inches='tight')
    
    
    surv_image_interp.iloc[:, 0:5].plot(drawstyle='steps-post')
    plt.title('continuous survival times for test dataset')
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')
    plt.savefig('plot_C_Test.png', dpi=300, bbox_inches='tight')
    
    surv_train_image_interp.iloc[:, 0:5].plot(drawstyle='steps-post')
    plt.title('continuous survival times for train dataset')
    plt.ylabel('S(t | x)')
    _ = plt.xlabel('Time')
    plt.savefig('plot_C_Train.png', dpi=300, bbox_inches='tight')



def evaluate_survmodel(surv_image_interp, surv_train_image_interp, durations_test, events_test, durations_train, events_train):
    print('#####################################################################')
    
    ev = EvalSurv(surv_image_interp, durations_test, events_test, censor_surv='km')
    Ctd=ev.concordance_td('antolini')
    with open("Eval_indexes.txt", "w") as external_file:
        add_text = f'The concordance score for test dataset is:{Ctd}'
        print(add_text, file=external_file)
        external_file.close()

    
    ev_train = EvalSurv(surv_train_image_interp, durations_train, events_train, censor_surv='km')
    Ctd_train=ev_train.concordance_td('antolini')
    with open("Eval_indexes.txt", "a") as external_file:
        add_text = f'The concordance score for train  dataset is:{Ctd_train}'
        print(add_text, file=external_file)
        external_file.close()
    
    print('#####################################################################')
    
    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    IBS=ev.integrated_brier_score(time_grid)
    IBN=ev.integrated_nbll(time_grid)
    with open("Eval_indexes.txt", "a") as external_file:
        add_text = f'The integrated brier score for test dataset is:{IBS}'
        print(add_text, file=external_file)
        external_file.close()
    with open("Eval_indexes.txt", "a") as external_file:
        add_text = f'The integrated nbll for test dataset is:{IBN}'
        print(add_text, file=external_file)
        external_file.close()
    
    
    time_grid_train = np.linspace(durations_train.min(), durations_train.max(), 100)
    IBS_train=ev_train.integrated_brier_score(time_grid_train)
    IBN_train=ev_train.integrated_nbll(time_grid_train)
    with open("Eval_indexes.txt", "a") as external_file:
        add_text = f'The integrated brier score for train dataset is:{IBS_train}'
        print(add_text, file=external_file)
        external_file.close()
    
    with open("Eval_indexes.txt", "a") as external_file:
        add_text = f'The integrated nbll for train dataset is:{IBN_train}'
        print(add_text, file=external_file)
        external_file.close()