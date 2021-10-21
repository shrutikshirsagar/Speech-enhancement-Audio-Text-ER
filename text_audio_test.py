import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys

sys.path.insert(1, '../')

import json
import pandas as pd
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm
from STFE import Models, DataPreparer
from tensorflow.keras import optimizers
from pickle import load

emotion_key = {
    'anger' : 0,
    'sad' : 1
}

def get_df(gmap_dir, msf_dir, txt_dir, class_name):
    print('Processing', gmap_dir, msf_dir, txt_dir)
    gmap_list = [x for x in os.listdir(gmap_dir)]
    msf_list = [x for x in os.listdir(msf_dir)]
    common_list = list(set(gmap_list).intersection(msf_list))
    txt_list = [x for x in os.listdir(txt_dir)]

    common_list = list(set(common_list).intersection(txt_list))

    gmap = [gmap_dir + x for x in common_list]
    msf = [msf_dir + x for x in common_list]
    txt = [txt_dir + x for x in common_list]

    # print(gmap[0])
    gmap_len = len(pd.read_csv(gmap[0], sep = ';', header = None, skiprows = [0]).columns)
    text_len = 768
    # print(pd.read_csv(gmap[0], sep = ';', header = None, skiprows = [0]).columns)
    # print(gmap_len)
    # print('#', gmap_len, text_len)
    full = pd.DataFrame(columns = list(range(223 + gmap_len + text_len - 2)) + ['class'])
    # print(len(full.columns))
    i = 0

    for f in tqdm(common_list):
        gmap_curr = gmap_dir + f
        msf_curr = msf_dir + f
        txt_curr = txt_dir + f

        gmap_df = pd.read_csv(gmap_curr, sep = ';', header = None, skiprows = [0])
        gmap_df.drop([0, 1], axis = 1, inplace = True)
        # print('gmap', list(gmap_df.loc[0]))
        msf_df = pd.read_csv(msf_curr, sep = ',', header = None).mean(axis = 0)
        # print('msf', msf_df)
        txt_df = pd.read_csv(txt_curr)
        # print(txt_df)
        # print('txt', list(txt_df['0']))

        # print(msf_df.mean(axis = 0), msf_df.shape)
        # print(len(list(gmap_df.loc[0])), len(list(msf_df)), len(list(txt_df['0'])))
        full.loc[i] = list(gmap_df.loc[0]) + list(msf_df) + list(txt_df['0']) + [class_name]
        # break
        i += 1
    return full


def wrapper(gmap_grand, msf_grand, txt_grand, set_name, aud_noise, txt_noise):

    gmap_anger = gmap_grand + set_name + '_anger_' + aud_noise + '/'
    msf_anger = msf_grand + set_name + '_anger_' + aud_noise + '/msf/'
    txt_anger = txt_grand + set_name + '_anger_' + txt_noise + '/'
    
    gmap_sad = gmap_grand + set_name + '_sad_' + aud_noise + '/'
    msf_sad = msf_grand + set_name + '_sad_' + aud_noise + '/msf/'
    txt_sad = txt_grand + set_name + '_sad_' + txt_noise + '/'
    

    df_anger = get_df(gmap_anger, msf_anger, txt_anger, 'anger')
    df_sad = get_df(gmap_sad, msf_sad, txt_sad, 'sad')
    
    df_csv = pd.concat([df_anger, df_sad], ignore_index = True)
    
    df_csv = df_csv.sample(frac = 1)

    return df_csv

def splitXY(df):
    X = df.drop('class', axis = 1)
    y = [emotion_key[x] for x in df['class']]

    def f(x):
        return np.float(x)
    f2 = np.vectorize(f)
    
    X = np.array(X)
    # print(X)
    y = np.array(y)
    # print(X.shape, y.shape)
    # print(X.dtype, y.dtype)
    return f2(X), y


gmap_grand = '../../../mitacs/MELD_noise_eGEMAPS_feat/'
msf_grand = '../../../mitacs/MELD_dataset_MSF/'
txt_grand = '../../../mitacs/MELD_text/'

#TODO keep switching these for required test set
txt_noise = 'CAFETERIA_15dB'
aud_noise = txt_noise
# aud_noise = 'clean'

scaler_name = '../../models/c_ab_scaler.pkl'
model_name = '../../models/clean_ab.h5'

test_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'test', aud_noise, txt_noise)
X_test, y_test = splitXY(test_csv)

scaler = load(open(scaler_name, 'rb'))
X_test = scaler.transform(X_test)


print(np.max(X_test), np.min(X_test))
print("Done scaling data")

class_names = ['anger', 'sad']

nnn = Models.NormalNeuralNetwork(0.3, class_names, (X_test.shape[1], ))
sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.95, nesterov=False)

nnn.l_model(model_name, sgd)

nnn_metrics = nnn.get_metrics(X_test, y_test)
print(nnn_metrics['Bal_Acc'])
print(nnn_metrics['Classification Report'])

