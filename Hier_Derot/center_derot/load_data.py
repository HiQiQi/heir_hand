import numpy
import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def load_data_multi_center_continuous(path, is_shuffle):
    print 'is_shuffle',is_shuffle

    f = h5py.File(path,'r')

    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    r0.shape = (r0.shape[0], 1, r0.shape[1],r0.shape[2])
    r1.shape = (r1.shape[0], 1, r1.shape[1],r1.shape[2])
    r2.shape = (r2.shape[0], 1, r2.shape[1],r2.shape[2])


    idx = [0,9 ]

    print idx
    center_uvd = numpy.mean(joint_label_uvd[:,idx,:],axis=1)
    if is_shuffle:
        r0,r1,r2,center_uvd ,joint_label_uvd= shuffle(r0,r1,r2,center_uvd,joint_label_uvd,random_state=0)
    # for i in xrange(10):
    #
    #     plt.imshow(r0[i,0],'gray')
    #     u = center_uvd[i,0]*72+12
    #     v = center_uvd[i,1]*72+12
    #     plt.scatter(u,v,c='r')
    #     plt.scatter(joint_label_uvd[i,:,0]*96,joint_label_uvd[i,:,1]*96)
    #     plt.show()
    return r0, r1,r2, center_uvd