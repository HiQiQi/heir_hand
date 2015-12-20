__author__ = 'QiYE'
import h5py
import numpy
import matplotlib.pyplot as plt
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def show_patch(r0_patch,r1_patch,r2_patch,r0,gr_uvd_derot,pred_uvd_derot):
    num = 10
    index = numpy.random.randint(0,r0_patch.shape[0],num)
    for k in xrange(num):
        i = index[k]
        fig = plt.figure()
        ax= fig.add_subplot(221)
        ax.imshow(r0_patch[i],'gray')
        ax= fig.add_subplot(223)
        ax.imshow(r1_patch[i],'gray')
        ax= fig.add_subplot(224)
        ax.imshow(r2_patch[i],'gray')
        ax= fig.add_subplot(222)
        ax.imshow(r0[i],'gray')
        plt.scatter(gr_uvd_derot[i,:,0]*96,gr_uvd_derot[i,:,1]*96,c='r')
        plt.scatter(pred_uvd_derot[i,:,0]*96,pred_uvd_derot[i,:,1]*96,c='g')
        plt.title('%d'%i)
        plt.show()

def show_patch_offset(r0_patch,r1_patch,r2_patch,offset,patch_size,patch_pad_width):
    num=5
    index = numpy.arange(1,1+num,1)
    # index = numpy.random.randint(0,r0_patch.shape[0],10)
    for k in xrange(num):
        i = index[k]
        fig = plt.figure()
        ax= fig.add_subplot(221)
        ax.imshow(r0_patch[i],'gray')
        plt.scatter(patch_size/2+patch_pad_width,patch_size/2+patch_pad_width,c='r')
        plt.scatter(offset[i,0]*patch_size+patch_pad_width,offset[i,1]*patch_size+patch_pad_width,c='g')
        ax= fig.add_subplot(223)
        ax.imshow(r1_patch[i],'gray')
        ax= fig.add_subplot(224)
        ax.imshow(r2_patch[i],'gray')
        # ax= fig.add_subplot(222)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(numpy.mean(uvd[i,:,0])*96,numpy.mean(uvd[i,:,1])*96,c='r')
        # plt.scatter(uvd[i,:,0]*96,uvd[i,:,1]*96,c='g')
        plt.title('%d'%i)
        plt.show()



def load_data_r012_ego_offset(path,is_shuffle,jnt_idx,offset_depth_range,patch_size=44,patch_pad_width=4,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    pred_uvd_derot = f['pred_uvd_derot'][...]
    f.close()

    crop_center_uvd = numpy.squeeze(pred_uvd_derot[:,jnt_idx,:])
    gr_uvd_jnt = numpy.squeeze(gr_uvd_derot[:,jnt_idx,:])

    # print offset.shape
    # print 'u min,max ',numpy.min(offset[:,0]),numpy.max(offset[:,0])
    # show_hist(offset[:,0])
    # print 'v min,max ',numpy.min(offset[:,1]),numpy.max(offset[:,1])
    # show_hist(offset[:,1])
    # print 'd min,max ',numpy.min(offset[:,2]),numpy.max(offset[:,2])
    # show_hist(offset[:,2])
    # print "\n"

    p0,p1,p2 = crop_patch(crop_center_uvd,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset =gr_uvd_jnt-crop_center_uvd

    # offset = norm_offset_uvd(cur_uvd=crop_center_uvd,prev_uvd=gr_uvd_jnt,offset_depth_range=offset_depth_range,patch_size=patch_size,hand_width=hand_width)


    print 'u beyond [0,1]',numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
    print 'v beyond [0,1]',numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
    print 'd beyond [0,1]',numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]
    if is_shuffle == True:
        p0, p1, p2 ,offset= shuffle(p0, p1, p2,offset)

    # show_patch_offset(p0,p1,p2,offset,patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])

    return p0, p1, p2, offset


def load_data_r012_bw_offset(path,is_shuffle,jnt_idx,patch_size=44,patch_pad_width=4,hand_width=96,hand_pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    pred_uvd_derot = f['pred_uvd_derot'][...]
    f.close()
    idx = [0,9]
    crop_center_uvd = numpy.mean(pred_uvd_derot[:,idx,:],axis=1)
    offset = (gr_uvd_derot - pred_uvd_derot)[:,jnt_idx,:]
    for i in xrange(6):
        print 'u min,max ',numpy.min(offset[:,i,0]),numpy.max(offset[:,i,0])
        print 'v min,max ',numpy.min(offset[:,i,1]),numpy.max(offset[:,i,1])
        print 'd min,max ',numpy.min(offset[:,i,2]),numpy.max(offset[:,i,2])
        print "\n"

    p0,p1,p2 = crop_patch(crop_center_uvd,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)

    if is_shuffle == True:
        p0, p1, p2 ,offset= shuffle(p0, p1, p2,offset)

    # show_patch_offset(p0,p1,p2,r0,gr_uvd_derot[:,jnt_idx,:],pred_uvd_derot[:,jnt_idx,:])
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])

    return p0, p1, p2, offset.reshape((offset.shape[0],offset.shape[1]*offset.shape[2]))

# def load_data_multi_base_uvd_normalized(path,is_shuffle,jnt_idx,patch_size=44,patch_pad_width=4,offset_depth_range=1.0,hand_width=96,hand_pad_width=0):
#     '''creat pathes based on ground truth
#     htmap is a qunatized location for each joint
#     '''
#
#     f = h5py.File(path,'r')
#     r0 = f['r0'][...]
#     r1 = f['r1'][...]
#     r2 = f['r2'][...]
#     gr_uvd_derot = f['gr_uvd_derot'][...]
#     pred_uvd_derot = f['pred_uvd_derot'][...]
#     f.close()
#     idx = [0,9]
#     crop_center_uvd = numpy.mean(pred_uvd_derot[:,idx,:],axis=1)
#
#
#
#     offset = (gr_uvd_derot - pred_uvd_derot)[:,jnt_idx,:]
#     for i in xrange(6):
#         print 'u min,max ',numpy.min(offset[:,i,0]),numpy.max(offset[:,i,0])
#         print 'v min,max ',numpy.min(offset[:,i,1]),numpy.max(offset[:,i,1])
#         print 'd min,max ',numpy.min(offset[:,i,2]),numpy.max(offset[:,i,2])
#         print "\n"
#
#     # for i in xrange(00,10,1):
#     #     plt.imshow(r0[i],'gray')
#     #     plt.scatter(joint_label_uvd[i,jnt_idx,0]*96,joint_label_uvd[i,jnt_idx,1]*96,c='g')
#     #     plt.scatter(prev_jnt_uvd[i,0]*96,prev_jnt_uvd[i,1]*96,c='r')
#     #     plt.show()
#     p0,p1,p2 = crop_patch(crop_center_uvd,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
#
#     print 'u beyond [0,1]',numpy.where(offset[:,:,0]<0)[0].shape[0]+numpy.where(offset[:,:,0]>1)[0].shape[0]
#     print 'v beyond [0,1]',numpy.where(offset[:,:,1]<0)[0].shape[0]+numpy.where(offset[:,:,1]>1)[0].shape[0]
#     print 'd beyond [0,1]',numpy.where(offset[:,:,2]<0)[0].shape[0]+numpy.where(offset[:,:,2]>1)[0].shape[0]
#     if is_shuffle == True:
#         p0, p1, p2 ,offset= shuffle(p0, p1, p2,offset)
#
#     # show_patch_offset(p0,p1,p2,offset,r0,cur_uvd,patch_size=patch_size,patch_pad_width=4)
#     p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
#     p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
#     p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
#     # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
#
#     return p0, p1, p2, offset.reshape((offset.shape[0],offset.shape[1]*offset.shape[2]))

