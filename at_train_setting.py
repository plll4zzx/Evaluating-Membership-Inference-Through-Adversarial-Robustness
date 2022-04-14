
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import numpy as np

pgd_at_config={
    'cifar10':{
        'epsilon':8/255, 
        'steps':7, 
        'lp':np.inf, 
        'step_size':2/255,
        'lr_schedule':[
            [0,0.1], [80, 0.01], 
            [160, 0.001], [200, 0.0005], 
            [240, 0.0001]
        ],
        'opt':RMSprop,
    },
    'fashion':{
        'epsilon':0.1, 
        'steps':8, 
        'lp':np.inf, 
        'step_size':0.02,
        'lr_schedule':[
            [0,0.1], [30, 0.01], 
            [50, 0.001], [70, 1e-4], 
            [90, 1e-5], [110, 1e-6], 
            [200, 0.0001]
        ],
        'opt':Adam,
    },
    'yale':{
        'epsilon':8/255, 
        'steps':7, 
        'lp':np.inf, 
        'step_size':2/255,
        'lr_schedule':[
            [0,1e-4], [20,1e-4], 
            [100, 5e-5], [200, 1e-5], 
            [300, 0.0001]
        ],
        'opt':Adam,
    },
}
