__author__ = 'QiYE'
import h5py
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy
from src.utils.show_statistics import show_hist
from sklearn.preprocessing import normalize

def norm_01(x):
    num=x.shape[0]
    chan = x.shape[1]
    img_size = x.shape[2]

    x.shape=(num*chan,img_size*img_size)
    min_v = numpy.min(x,axis=-1)
    max_v = numpy.max(x,axis=-1)
    range_v = max_v - min_v
    loc = numpy.where(range_v == 0)
    range_v[loc]=1.0
    min_v[loc]=0
    x=(x-min_v[:,numpy.newaxis])/range_v[:,numpy.newaxis]

    x.shape=(num,chan,img_size,img_size)
    return x



def load_data_multi(path,jnt_idx,is_shuffle):
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')

    r0 = numpy.squeeze(f['patch00'][...][jnt_idx])
    r1 =  numpy.squeeze(f['patch10'][...][jnt_idx])
    r2=  numpy.squeeze(f['patch20'][...][jnt_idx])
    pred_uvd_derot = f['pred_uvd_derot'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]

    f.close()

    offset=numpy.squeeze((gr_uvd_derot.reshape(gr_uvd_derot.shape[0],6,3)-pred_uvd_derot.reshape(gr_uvd_derot.shape[0],6,3))[:,jnt_idx,:])*10
    # show_hist(offset[:,0])
    # show_hist(offset[:,1])
    # show_hist(offset[:,2])
    # print offset[numpy.where(offset>1)]
    # print offset[numpy.where(offset<-1)]
    # offset[numpy.where(offset>1)]=0
    # offset[numpy.where(offset<-1)]=0
    if is_shuffle:
        r0,r1,r2,offset = shuffle(r0,r1,r2,offset,random_state=0)
    r0=norm_01(r0)
    r1=norm_01(r1)
    r2=norm_01(r2)
    # for i in numpy.random.randint(0,r0.shape[0],2):
    #     for j in xrange(6):
    #         print 'min mx', numpy.max(r0[i,j]),numpy.min(r0[i,j])
    #         fig = plt.figure()
    #         plt.imshow(r0[i,j],'gray')
    #         plt.show()
    # r0.shape = (r0.shape[0], 1, r0.shape[1],r0.shape[2])
    # r1.shape = (r1.shape[0], 1, r1.shape[1],r1.shape[2])
    # r2.shape = (r2.shape[0], 1, r2.shape[1],r2.shape[2])
    return r0, r1,r2,offset

def load_result_iter1(path,pred_uvd):
    print pred_uvd.shape
    pred_uvd.shape=(pred_uvd.shape[0],6,3)
    iter1_absuvd_name=['_iter1_bw0_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep610',
                       '_iter1_bw1_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep655',
                       '_iter1_bw2_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep450',
                       '_iter1_bw3_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep405',
                       '_iter1_bw4_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep450',
                       '_iter1_bw5_r012_egoff_c0064_h11_h22_gm0_lm6000_yt0_ep440']
    for i,name in enumerate(iter1_absuvd_name):
        pred_uvd[:,i,:]=numpy.load('%s_absuvd%s.npy'%(path,name))
    pred_uvd.shape=(pred_uvd.shape[0],18)
    return

def load_data_multi_initial(path,jnt_idx,is_shuffle):
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


