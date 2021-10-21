# Saves BERT features as individual CSV files in the tree where

import sys

sys.path.insert(1, '../')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from STFE.SpeechTextFeatures import *
from tqdm import tqdm
import re
import os
mname_txt = 'bert-base-uncased'
TFE = BERT_Text_Feature_Extracter(mname_txt)

transcript_path = '/home/amrgaballah/Desktop/exp_1/Machine_enh_tran/'

# transcript_path = '../noise_csv/'
set = 'dev'
noise = 'clean'
target = '/home/amrgaballah/Desktop/exp_1/Machine_enh_tran/MELD_machine_enh_BERTtext_feat/'




# for s in ['train', 'test', 'dev']:
for s in ['dev','train','test']:
      
    for n in ['airport_0dB']:
        data_frame = pd.read_csv(transcript_path + s + '_' + n + '.csv')

        req = np.array(data_frame[['Utterance', 'Class', 'ID']])
        print('\nPreparing data for ' + s + '_' + n)

        for u, e, id in tqdm(req):
                
            if pd.isna(u) == False:
                feats = pd.DataFrame(TFE.features_fromtext(u))
                id = id[:-3] + 'csv'
                out_dir = target + s + '_' + e + '_' + n + '/' 
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                feats.to_csv(out_dir+ id, index = False)
                    
                # print(id, target + s + '_' + e + '_' + n + '/' + id)
                # break






