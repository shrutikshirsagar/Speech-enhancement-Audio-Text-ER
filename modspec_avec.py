import os, sys
from frontend import MFStats, srmr_audio
import numpy as np
import h5py


def listfolders():
   
    
        pathin = "//home/amrgaballah/Desktop/exp_1/Machine_enh/"
        pathout = '//home/amrgaballah/Desktop/exp_1/Machine_enh_MSF/'
    
        ftype = 3
    

        for dirname in os.listdir(pathin):
            print("Processing: %s/%s" % (pathin, dirname))
            
            mrs_dir = "%s/%s/%s" % (pathout, dirname, "mrs")
            msf_dir = "%s/%s/%s" % (pathout, dirname, "msf")
            
            print("ftype has been defined as : %s", ftype)
            pool_msr(pathin, dirname, mrs_dir, msf_dir,  ftype)


def write_file(path, file, features, ftype):
    if int(ftype) == 1:
        write_arff(path, file, features)
    elif int(ftype) == 2:
        write_hdf5(path, file, features)
    else:
        write_csv(path, file, features)

def write_hdf5(path, file, features):
    if not os.path.exists(path):
        os.makedirs(path)

    f = h5py.File("%s/%s"%(path, "%s%s" % (file[:-4], ".h5")), "w")
    f.create_dataset("mod", data=features)
    f.close()

def write_arff(path, file, features):
    if not os.path.exists(path):
        os.makedirs(path)
    f = open("%s/%s"%(path, "%s%s" % (file[:-4], ".arff")), "w")
    f.write("@RELATION %s\n"%(file[:-5]))
    f.write("\n")

    for col in range(0, len(features[0])):
        f.write("@attribute att%s numeric\n" % col)

    f.write("\n")
    f.write("@data\n")
    f.write("\n")

    for row in range(0, len(features)):
        str = ""
        for col in range(0, len(features[0])):
            str += "%f,"%(features[row][col])
        f.write("%s\n" % str[:-2])

    f.close()

def write_csv(path, file, features):
    if not os.path.exists(path):
        os.makedirs(path)

    np.savetxt('%s/%s' % (path, "%s%s" % (file[:-4], ".csv")), features, delimiter=",")

def get_no_examples(mf, no_examples):
    if len(mf) < no_examples:
        tmp = np.reshape(mf[len(mf) - 1], (1, mf.shape[1]))
        tmp = np.repeat(tmp, no_examples - len(mf), axis=0)
        mf = np.vstack((mf, tmp))
    else:
        mf = mf[0:no_examples]
    return mf

def pool_msr(path, dirname, mrs_dir, msf_dir,  ftype = 3):
    dirs = os.listdir("%s/%s"%(path, dirname))
    for f in dirs:
        if f.endswith('.wav'):
            print('%s/%s/%s' % (path, dirname, f))

            try:
                if dirname == "":
                    mf = srmr_audio(path, f)
                else:
                    mf = srmr_audio("%s/%s" % (path, dirname), f)
            except:
                continue
            mf = np.einsum('ijk->kij', mf)

            stats = MFStats(mf)

            mrs = np.reshape(mf, (mf.shape[0], mf.shape[1] * mf.shape[2]))
            nFrame = mrs.shape[0]
            write_file(mrs_dir, f, mrs, ftype)

            msf = stats.get_stats()
            write_file(msf_dir, f, msf, ftype)

            #mrs_pooling = stats.moving_stats(mrs, poolingsize)
            #mrs_pooling = get_no_examples(mrs_pooling,nFrame)

            #mrs_pooling = np.concatenate((mrs_pooling, mrs), axis=1)
            #write_file(pooling_1_dir, file, mrs_pooling, ftype)

            #msf_pooling = stats.moving_stats(msf, poolingsize)
            #msf_pooling = get_no_examples(msf_pooling,nFrame)

            #msf_pooling = np.concatenate((msf_pooling, mrs), axis=1)
            #write_file(pooling_2_dir, file, msf_pooling, ftype)

listfolders()



