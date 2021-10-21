import librosa    
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf
from pathlib import Path
import glob
import os
folds = ['dev_anger_airport_0dB', 'dev_sad_airport_0dB','train_anger_airport_0dB','train_sad_airport_0dB','test_anger_airport_0dB','test_sad_airport_0dB']
in_path1 = '/home/amrgaballah/Desktop/exp_1/Human_enh/'
out_path1 = '/home/amrgaballah/Desktop/exp_1/Human_enh_16bit/'

for fold in folds:
    in_path = os.path.join(in_path1, fold)
    print(in_path)
    out_path = os.path.join(out_path1, fold)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for filename in os.listdir(in_path):
        print(filename)
        spec = os.path.join(in_path,filename)
        print(spec)
        data, samplerate = sf.read(spec)
        
#         y, s = librosa.load(spec, sr=16000)
#         print(y.shape, s)
        final_fold = os.path.join(out_path,filename)
#         print(final_fold)
#         sf.write(final_fold, y, s)
        sf.write(final_fold, data, samplerate, subtype='PCM_16')
