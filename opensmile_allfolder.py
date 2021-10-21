
"""
Created on Fri Apr  3 13:15:25 2020

@author: shrutikshirsagar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pathlib import Path
import glob
import os
import os.path
pathin = "/home/amrgaballah/Desktop/exp_1/Human_enh_16bit"
pathout = "/home/amrgaballah/Desktop/exp_1/Human_enh_eGEMAPS"



def filematch(pathin, pathout):
    dirs = os.listdir("%s/%s"%(pathin, dirname))
    print('directory name', dirs)
    dirFiles1 = dirs #list of directory files
    dirFiles1.sort()
    print(dirFiles1)
    if not os.path.exists("%s/%s" % (pathout, dirname)):
    
        os.makedirs("%s/%s" % (pathout, dirname))
    new_path = os.path.join(pathin, dirname)
    print('new_path', new_path )
    out_path = os.path.join(pathout, dirname)
    print(out_path)
    for f in dirs:
        print('file', f)

        #cmd = "inst/bin/SMILExtract -C config/gemaps/eGeMAPSv01a.conf -I """ + new_path + "/" + f  +  "  -csvoutput """ + out_path + "/" + f + "features.csv" 
        cmd = "inst/bin/SMILExtract -C config/gemaps/eGeMAPSv01a.conf + '-appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'+'-lldcsvoutput' -I """ + new_path + "/" + f  +  "  -csvoutput """ +out_path + "/" + f.split('.')[0] + ".csv"

        print(cmd)
        os.system(cmd)
                
for dirname in os.listdir(pathin):
    print('directname', dirname)
    print('path in', os.listdir(pathin))
    print("Processing: %s/%s" % (pathin, dirname))
    #if os.path.isdir("%s/%s" % (pathin, dirname)):
    #pathout = "%s/%s" % (pathout, dirname)
    
    filematch(pathin, pathout)
        #if not os.path.exists(pathout):
