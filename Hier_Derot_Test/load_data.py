__author__ = 'QiYE'

import h5py
import numpy
import matplotlib.pyplot as plt
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd,crop_patch_enlarge
from sklearn.utils import shuffle
from src.utils import read_save_format
def get_thresh_bw_jnts(joint_label_uvd,ratio):
    thresh=[]

    for i in xrange(1,5):
        offset = numpy.sum((joint_label_uvd[:,i,:] - joint_label_uvd[:,i+1,:])**2,axis=-1)
        print 'jnt', i, i+1
        print 'offset min, max',numpy.min(offset),numpy.max(offset)
        offset_mean = numpy.mean(offset)
        print 'offsetmean', offset_mean
        print 'num offset<offset_mean/ratio',numpy.where(offset<offset_mean/ratio)[0].shape[0]

        thresh.append([i,i+1,offset_mean/ratio])
        print 'thresh for jnt pair',thresh[-1]
    for i in xrange(1,6):
        offset = numpy.sum((joint_label_uvd[:,0,:] - joint_label_uvd[:,i,:])**2,axis=-1)
        print 'jnt', i, i+1
        print 'offset min, max',numpy.min(offset),numpy.max(offset)
        offset_mean = numpy.mean(offset)
        print 'offsetmean', offset_mean
        print 'num offset<offset_mean/ratio',numpy.where(offset<offset_mean/ratio)[0].shape[0]
        thresh.append([0,i,offset_mean/ratio])
        print 'thresh for jnt pair', thresh[-1]
    return thresh

def load_norm_offset_uvd(path,pre_jnt_path,jnt_idx,patch_size,offset_depth_range,hand_width):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    depth = joint_label_uvd[:,:,2]
    print 'depth max',numpy.max(depth)
    print 'depth min',numpy.min(depth)

    prev_jnt_uvd = numpy.load(pre_jnt_path)
    cur_uvd =joint_label_uvd[:,jnt_idx,:]

    offset = norm_offset_uvd(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd,offset_depth_range=offset_depth_range,
                                                                patch_size=patch_size,hand_width=hand_width)

    return offset
def load_data_multi_base_uvd_normalized(path,prev_jnt_uvd,is_shuffle,jnt_idx,patch_size=44,patch_pad_width=4,offset_depth_range=1.0,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    depth = joint_label_uvd[:,:,2]
    print 'depth max',numpy.max(depth)
    print 'depth min',numpy.min(depth)

    cur_uvd =joint_label_uvd[:,jnt_idx,:]
    # offset = cur_uvd - prev_jnt_uvd.reshape((prev_jnt_uvd.shape[0],1,prev_jnt_uvd.shape[1]))

    # for i in xrange(00,10,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(joint_label_uvd[i,jnt_idx,0]*96,joint_label_uvd[i,jnt_idx,1]*96,c='g')
    #     plt.scatter(prev_jnt_uvd[i,0]*96,prev_jnt_uvd[i,1]*96,c='r')
    #     plt.show()
    p0,p1,p2 = crop_patch(prev_jnt_uvd,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset = norm_offset_uvd(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd,offset_depth_range=offset_depth_range,
                                                                patch_size=patch_size,hand_width=hand_width)
    if is_shuffle == True:
        p0, p1, p2 ,offset= shuffle(p0, p1, p2,offset)

    print 'u beyond [0,1]',numpy.where(offset[:,:,0]<0)[0].shape[0]+numpy.where(offset[:,:,0]>1)[0].shape[0]
    print 'v beyond [0,1]',numpy.where(offset[:,:,1]<0)[0].shape[0]+numpy.where(offset[:,:,1]>1)[0].shape[0]
    print 'd beyond [0,1]',numpy.where(offset[:,:,2]<0)[0].shape[0]+numpy.where(offset[:,:,2]>1)[0].shape[0]
    # show_patch_offset(p0,p1,p2,offset,r0,cur_uvd,patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
    return p0, p1, p2, offset.reshape((offset.shape[0],offset.shape[1]*offset.shape[2]))

def load_data_multi_mid_uvd_normalized(path,prev_jnt_uvd_pred,jnt_idx,patch_size=24,patch_pad_width=4,hand_width=96,hand_pad_width=0,offset_depth_range=0.8):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    # pre_idx_pred = (jnt_idx[0]-2)/4+1
    # print (jnt_idx[0]-2)/4+1
    cur_uvd=numpy.squeeze(joint_label_uvd[:,jnt_idx,:])

    # for i in xrange(00,10,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(joint_label_uvd[i,jnt_idx,0]*96,joint_label_uvd[i,jnt_idx,1]*96,c='g')
    #     plt.show()


    p0,p1,p2 = crop_patch(prev_jnt_uvd_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset = norm_offset_uvd(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd_pred,offset_depth_range=offset_depth_range,
                                                                patch_size=patch_size,hand_width=hand_width)
    # print 'u beyond [0,1]',numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
    # print 'v beyond [0,1]',numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
    # print 'd beyond [0,1]',numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]
    # show_patch_offset_jnt(p0,p1,p2,offset,r0,joint_label_uvd[:,jnt_idx,:],patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
    return p0, p1, p2, offset

def load_data_multi_top_uvd_normalized(path,prev_jnt_uvd_derot,jnt_idx,patch_size=24, patch_pad_width=4,offset_depth_range=0.8,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    print jnt_idx
    cur_uvd=numpy.squeeze(joint_label_uvd[:,jnt_idx,:])

    # for i in xrange(00,10,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(joint_label_uvd[i,jnt_idx,0]*96,joint_label_uvd[i,jnt_idx,1]*96,c='g')
    #     plt.show()
    #
    p0,p1,p2 = crop_patch(prev_jnt_uvd_derot,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset = norm_offset_uvd(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd_derot,offset_depth_range=offset_depth_range,
                                                                patch_size=patch_size,hand_width=hand_width)

    # show_patch_offset_jnt(p0,p1,p2,offset,r0,cur_uvd,patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    #
    # print numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
    # print numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
    # print numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]

    # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
    return p0, p1, p2, offset

