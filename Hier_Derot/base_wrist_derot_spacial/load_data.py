__author__ = 'QiYE'
import h5py
import numpy
import matplotlib.pyplot as plt
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd,crop_patch_enlarge
from sklearn.utils import shuffle

def show_patch_offset(r0_patch,r1_patch,r2_patch,offset,r0,uvd,patch_size,patch_pad_width):
    # index = numpy.arange(0,10,1)
    index = numpy.random.randint(0,r0_patch.shape[0],10)
    for k in xrange(10):
        i = index[k]
        fig = plt.figure()
        ax= fig.add_subplot(221)
        ax.imshow(r0_patch[i],'gray')
        plt.scatter(patch_size/2+patch_pad_width,patch_size/2+patch_pad_width,c='r')
        plt.scatter(offset[i,:,0]*patch_size+patch_pad_width,offset[i,:,1]*patch_size+patch_pad_width,c='g')
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

def load_data_multi_base_uvd_normalized(path,pre_jnt_path,is_shuffle,jnt_idx,patch_size=44,patch_pad_width=4,offset_depth_range=1.0,hand_width=96,hand_pad_width=0):
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

    prev_jnt_uvd = numpy.load(pre_jnt_path)
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

def load_data_multi_base_uvd_normalized_enlarge(path,pre_jnt_path,jnt_idx,num_enlarge,dataset=None,patch_size=44, patch_pad_width=4,offset_depth_range=2.0,batch_size=100):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    prev_jnt_uvd_pred = numpy.load(pre_jnt_path)
    pre_idx = [0,9]
    prev_jnt_uvd_gr =numpy.mean(joint_label_uvd[:,pre_idx,:],axis=1)
    cur_uvd = joint_label_uvd[:,jnt_idx,:]
    # for i in xrange(00,10,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(joint_label_uvd[i,jnt_idx,0]*96,joint_label_uvd[i,jnt_idx,1]*96,c='g')
    #     plt.scatter(prev_jnt_uvd[i,0]*96,prev_jnt_uvd[i,1]*96,c='r')
    #     plt.show()
    #

    if dataset == None:
        p0,p1,p2 = crop_patch(prev_jnt_uvd_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=96,pad_width=0)
        offset = norm_offset_uvd(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd_pred,offset_depth_range=offset_depth_range,
                                                                    patch_size=patch_size,hand_width=96)
        print 'u beyond [0,1]',numpy.where(offset[:,:,0]<0)[0].shape[0]+numpy.where(offset[:,:,0]>1)[0].shape[0]
        print 'v beyond [0,1]',numpy.where(offset[:,:,1]<0)[0].shape[0]+numpy.where(offset[:,:,1]>1)[0].shape[0]
        print 'd beyond [0,1]',numpy.where(offset[:,:,2]<0)[0].shape[0]+numpy.where(offset[:,:,2]>1)[0].shape[0]
        # show_patch_offset(p0,p1,p2,offset_uvd,r0,joint_label_uvd[:,jnt_idx,:],patch_size=patch_size,patch_pad_width=4)
        p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
        p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
        p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
        # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
        return p0, p1, p2, offset.reshape((offset.shape[0],offset.shape[1]*offset.shape[2]))
    else:

        p0,p1,p2,new_cur_uvd,new_prev_uvd = crop_patch_enlarge(cur_uvd=cur_uvd,prev_uvd_pred=prev_jnt_uvd_pred,prev_uvd_gr=prev_jnt_uvd_gr,r0=r0,r1=r1,r2=r2,num_enlarge=num_enlarge,
                                                               patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=96,pad_width=0,batch_size=batch_size)
        # p0,p1,p2,new_cur_uvd,new_prev_uvd = crop_patch_enlarge2(cur_uvd,prev_jnt_uvd_pred,prev_jnt_uvd_gr,r0,r1,r2,num_enlarge=num_enlarge,
        #                                                        patch_size=patch_size,patch_pad_width=4,hand_width=96,pad_width=0,batch_size=batch_size)
        offset = norm_offset_uvd(cur_uvd=new_cur_uvd,prev_uvd=new_prev_uvd,offset_depth_range=offset_depth_range,
                                                                    patch_size=patch_size,hand_width=96)
        print 'u beyond [0,1]',numpy.where(offset[:,:,0]<0)[0].shape[0]+numpy.where(offset[:,:,0]>1)[0].shape[0]
        print 'v beyond [0,1]',numpy.where(offset[:,:,1]<0)[0].shape[0]+numpy.where(offset[:,:,1]>1)[0].shape[0]
        print 'd beyond [0,1]',numpy.where(offset[:,:,2]<0)[0].shape[0]+numpy.where(offset[:,:,2]>1)[0].shape[0]
        # show_patch_offset(p0,p1,p2,offset,r0,new_cur_uvd,patch_size=patch_size,patch_pad_width=patch_pad_width)
        p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
        p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
        p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
        # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
        p0,p1,p2,offset_uvd = shuffle(p0,p1,p2,offset, random_state=0)
        print p0.shape,offset_uvd.shape
        return p0, p1, p2, offset.reshape((offset.shape[0],offset.shape[1]*offset.shape[2]))
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

def load_patches(path):
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['jnt_lable_uvd'][...]
    f.close()
    return r0,r1,r2,joint_label_uvd
def save_pathes():
    batch_size = 100
    # jnt_idx = [1,2,6,10,14,18]
    jnt_idx = [0,1,5,9 ,13,17]
    num_enlarge=0
    patch_size=40
    offset_depth_range=1.0
    print 'offset_depth_range ',offset_depth_range
    model_info='base_wrist_r0r1r2_uvd_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info

    dataset = 'train'
    src_path = '../../data/source/'
    path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300.h5'%(src_path,dataset)

    direct = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/center_derot/best/'
    prev_jnt_path ='%s%s_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770.npy'%(direct,dataset)

    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_path,jnt_idx=jnt_idx,
                                                                                             patch_size=patch_size,patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)

    f = h5py.File('train_patch%d_1.0'%patch_size,'w')
    f.create_dataset('r0', data=train_set_x0)
    f.create_dataset('r1', data=train_set_x1)
    f.create_dataset('r2', data=train_set_x2)
    f.create_dataset('jnt_lable_uvd', data=train_set_y)
    f.close()

    dataset = 'test'
    path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300.h5'%(src_path,dataset)
    direct = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/center_derot/best/'
    prev_jnt_path ='%s%s_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770.npy'%(direct,dataset)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_path,jnt_idx=jnt_idx,
                                                                                         patch_size=patch_size,patch_pad_width=4,offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    f = h5py.File('test_patch%d_1.0'%patch_size,'w')
    f.create_dataset('r0', data=test_set_x0)
    f.create_dataset('r1', data=test_set_x1)
    f.create_dataset('r2', data=test_set_x2)
    f.create_dataset('jnt_lable_uvd', data=test_set_y)
    f.close()

if __name__=='__main__':
    save_pathes()