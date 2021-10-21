
# Training on non Augmented data only

# Works only on combination of eGEMAPS and MSF features. Can be altered if necessary

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
from pickle import dump


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
    # print('# 223', gmap_len, text_len)
    full = pd.DataFrame(columns = list(range(223 + gmap_len + text_len - 2)) + ['class'])
    # print(len(full.columns))
    i = 0

    for f in tqdm(common_list):
        gmap_curr = gmap_dir + f
        msf_curr = msf_dir + f
        txt_curr = txt_dir + f

        gmap_df = pd.read_csv(gmap_curr, sep = ';', header = None, skiprows = [0], index_col = False)
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

noise = 'clean'
noise2 = ['_0dB', '_10dB', '_20dB']

#noise_types = [] for using only clean data
noise_types = ['airport', 'babble']

aud_noise = noise
txt_noise = noise


name = 'text_' + txt_noise + '_aud_' + aud_noise + '_training'

# collecting clean data
train_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'train', noise, noise)
test_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'test', noise, noise)
dev_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'dev', noise, noise)

X_train, y_train = splitXY(train_csv)
X_test, y_test = splitXY(test_csv)
X_dev, y_dev = splitXY(dev_csv)

print(X_train.shape)

# print("Saved files")

# loading noisy data for training
for n in noise2:
    for noise_type in noise_types:
        n1 = noise_type + n
        print('\n', n1, '\n')
        dset_csv = wrapper(gmap_grand, msf_grand, txt_grand, 'train', n1, n1)
        dx_train, dy_train = splitXY(dset_csv)
        # print(dx_train.shape)
        X_train = np.concatenate([X_train, dx_train], axis = 0)
        y_train = np.concatenate([y_train, dy_train], axis = 0) 
        # print(X_train.shape)


scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_dev = scaler.transform(X_dev)

print("X_train shape", X_train.shape)

# TODO: Change scaler name
dump(scaler, open('../../models/c_ab_scaler.pkl', 'wb'))

print("Done scaling data")

class_names = ['anger', 'sad']
cws = [1, 1.8]
class_weights = {}
for i in range(len(class_names)):
    class_weights[i] = cws[i]

sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.95, nesterov=False)

nnn = Models.NormalNeuralNetwork(0.5, class_names, (X_train.shape[1], ))
nnn.model_compile(sgd)

nnn.model_fit(class_weights, 850, X_train, y_train, X_dev, y_dev, fig_name = name)

nnn_metrics = nnn.get_metrics(X_test, y_test)
print(nnn_metrics)

model = nnn.get_model()

# TODO change model name
model.save('../../models/clean_ab.h5', include_optimizer = False)