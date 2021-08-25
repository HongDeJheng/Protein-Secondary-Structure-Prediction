import pandas as pd
import numpy as np

def data_prepro():    
    data = pd.read_csv('data/ss.cleaned.csv')
    
    data = data[data.has_nonstd_aa == False]

    label = data['sst3'].values
    data = data['seq'].values
    
    idx = np.random.permutation(np.arange(len(label)))
    label = label[idx]
    data = data[idx]

    data = data[:50000]
    label = label[:50000]

    return data, label