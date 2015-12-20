__author__ = 'QiYE'
import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy
def load_data_multi(path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')

    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    uvd = f['joint_label_uvd'][...][:,jnt_idx,:]

    f.close()


    if is_shuffle:
        r0,r1,r2,uvd = shuffle(r0,r1,r2,uvd,random_state=0)
    # for i in numpy.random.randint(0,r0.shape[0],10):
    #     fig = plt.figure()
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(uvd[i,:,0]*72+12,uvd[i,:,1]*72+12)
    #     plt.show()
    r0.shape = (r0.shape[0], 1, r0.shape[1],r0.shape[2])
    r1.shape = (r1.shape[0], 1, r1.shape[1],r1.shape[2])
    r2.shape = (r2.shape[0], 1, r2.shape[1],r2.shape[2])
    return r0, r1,r2,uvd.reshape((uvd.shape[0], uvd.shape[1]*uvd.shape[2]))