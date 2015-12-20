__author__ = 'QiYE'
import h5py
import numpy
import matplotlib.pyplot as plt
from src.utils.crop_patch_norm_offset import crop_patch,norm_offset_uvd,crop_patch_enlarge
from sklearn.utils import shuffle
from src.utils.show_statistics import show_hist
def show_patch_offset_jnt(r0_patch,r1_patch,r2_patch,offset,r0,uvd,patch_size,patch_pad_width):
    # index = numpy.arange(0,10,1)
    num=10
    index = numpy.random.randint(0,r0_patch.shape[0],num)
    for k in xrange(num):
        i=index[k]
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

def show_patch_ori_offset_jnt(r0_patch,r1_patch,r2_patch,offset,r0,uvd,patch_size,hand_width,patch_pad_width):
    num=20
    index = numpy.random.randint(0,r0_patch.shape[0],num)
    for k in xrange(num):
        i=index[k]
        fig = plt.figure()
        ax= fig.add_subplot(221)
        ax.imshow(r0_patch[i],'gray')
        plt.scatter(patch_size/2+patch_pad_width,patch_size/2+patch_pad_width,c='r')
        plt.scatter(offset[i,0]*hand_width+patch_pad_width+patch_size/2,offset[i,1]*hand_width+patch_size/2+patch_pad_width,c='g')
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
def load_patches(path):
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['jnt_lable_uvd'][...]
    f.close()
    return r0,r1,r2,joint_label_uvd
def load_data_multi_mid_uvd_normalized(path,pre_jnt_path,jnt_idx,is_shuffle,patch_size=24,patch_pad_width=4,hand_width=96,hand_pad_width=0,offset_depth_range=0.8):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''
    print 'is_shuffle',is_shuffle
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    pre_idx_pred = (jnt_idx[0]-2)/4+1
    print (jnt_idx[0]-2)/4+1
    base_wrist_pred = numpy.load(pre_jnt_path)
    base_wrist_pred.shape=(base_wrist_pred.shape[0],6,3)
    prev_jnt_uvd_pred= numpy.squeeze(base_wrist_pred[:,pre_idx_pred,:])
    cur_uvd=numpy.squeeze(joint_label_uvd[:,jnt_idx,:])
    # offset = cur_uvd-pre_idx_pred
    # show_hist(offset[:,2])
    # for i in xrange(00,10,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(joint_label_uvd[i,jnt_idx,0]*96,joint_label_uvd[i,jnt_idx,1]*96,c='g')
    #     plt.show()


    p0,p1,p2 = crop_patch(prev_jnt_uvd_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_width)
    offset = norm_offset_uvd(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd_pred,offset_depth_range=offset_depth_range,
                                                                patch_size=patch_size,hand_width=hand_width)
    print 'u beyond [0,1]',numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
    print 'v beyond [0,1]',numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
    print 'd beyond [0,1]',numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]
    # show_patch_offset_jnt(p0,p1,p2,offset,r0,joint_label_uvd[:,jnt_idx,:],patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
    if is_shuffle == True:
        p0, p1, p2 ,offset= shuffle(p0, p1, p2,offset)

    # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
    return p0, p1, p2, offset

def load_data_multi_mid_uvd_normalized_enlarge(path,pre_jnt_path,jnt_idx,num_enlarge,dataset=None,patch_size=44, patch_pad_width=4,offset_depth_range=0.8,batch_size=100):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    pre_idx_pred = (jnt_idx[0]-2)/4+1
    print 'pre_idx_pred',pre_idx_pred
    base_wrist_pred = numpy.load(pre_jnt_path)
    base_wrist_pred.shape=(base_wrist_pred.shape[0],6,3)
    prev_jnt_uvd_pred= numpy.squeeze(base_wrist_pred[:,pre_idx_pred,:])

    pre_jnt_gr = jnt_idx[0]-1
    print 'pre_jnt_gr',pre_jnt_gr
    prev_jnt_uvd_gr =numpy.squeeze(joint_label_uvd[:,pre_jnt_gr,:])
    cur_uvd=numpy.squeeze(joint_label_uvd[:,jnt_idx,:])

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
        print 'u beyond [0,1]',numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
        print 'v beyond [0,1]',numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
        print 'd beyond [0,1]',numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]
        show_patch_offset_jnt(p0,p1,p2,offset,r0,joint_label_uvd[:,jnt_idx,:],patch_size=patch_size,patch_pad_width=4)
        p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
        p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
        p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
        # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
        return p0, p1, p2, offset
    else:
        p0,p1,p2,new_cur_uvd,new_prev_uvd = crop_patch_enlarge(cur_uvd,prev_jnt_uvd_pred,prev_jnt_uvd_gr,r0,r1,r2,num_enlarge=num_enlarge,
                                                               patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=96,pad_width=0,batch_size=batch_size)
        # p0,p1,p2,new_cur_uvd,new_prev_uvd = crop_patch_enlarge2(cur_uvd,prev_jnt_uvd_pred,prev_jnt_uvd_gr,r0,r1,r2,num_enlarge=num_enlarge,
        #                                                        patch_size=patch_size,patch_pad_width=4,hand_width=96,pad_width=0,batch_size=batch_size)
        offset = norm_offset_uvd(cur_uvd=new_cur_uvd,prev_uvd=new_prev_uvd,offset_depth_range=offset_depth_range,
                                                                    patch_size=patch_size,hand_width=96)
        print 'u beyond [0,1]',numpy.where(offset[:,0]<0)[0].shape[0]+numpy.where(offset[:,0]>1)[0].shape[0]
        print 'v beyond [0,1]',numpy.where(offset[:,1]<0)[0].shape[0]+numpy.where(offset[:,1]>1)[0].shape[0]
        print 'd beyond [0,1]',numpy.where(offset[:,2]<0)[0].shape[0]+numpy.where(offset[:,2]>1)[0].shape[0]
        show_patch_offset_jnt(p0,p1,p2,offset,r0,new_cur_uvd,patch_size=patch_size,patch_pad_width=patch_pad_width)
        p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
        p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
        p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
        # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
        p0,p1,p2,offset_uvd = shuffle(p0,p1,p2,offset, random_state=0)
        print p0.shape,offset_uvd.shape
        return p0, p1, p2, offset


def load_patch_offset(path):
    f = h5py.File(path, 'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    offset = f['offset'][...]

    offset[:,2] /=10.0
    # print offset[0:10]
    print numpy.min(offset[:,0])
    print numpy.max(offset[:,0])
    print numpy.min(offset[:,1])
    print numpy.max(offset[:,1])
    print numpy.min(offset[:,2])
    print numpy.max(offset[:,2])
    return r0,r1,r2,offset

def load_data_r0r1r2_top_uvd_enlarge(path,pre_jnt_path,jnt_idx,num_enlarge,dataset=None,patch_size=44, patch_pad_width=4,hand_width=96,hand_pad_witht=0,batch_size=100):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    pre_idx_pred = (jnt_idx[0]-2)/4+1
    base_wrist_pred = numpy.load(pre_jnt_path)
    base_wrist_pred.shape=(base_wrist_pred.shape[0],6,3)
    prev_jnt_uvd_pred= numpy.squeeze(base_wrist_pred[:,pre_idx_pred,:])

    pre_jnt_gr = jnt_idx[0]-1
    prev_jnt_uvd_gr =numpy.squeeze(joint_label_uvd[:,pre_jnt_gr,:])
    cur_uvd=numpy.squeeze(joint_label_uvd[:,jnt_idx,:])

    # for i in xrange(00,10,1):
    #     plt.imshow(r0[i],'gray')
    #     plt.scatter(joint_label_uvd[i,jnt_idx,0]*96,joint_label_uvd[i,jnt_idx,1]*96,c='g')
    #     plt.scatter(prev_jnt_uvd[i,0]*96,prev_jnt_uvd[i,1]*96,c='r')
    #     plt.show()
    if dataset == None:
        p0,p1,p2 = crop_patch(prev_jnt_uvd_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_witht)
        offset = cur_uvd-prev_jnt_uvd_pred
        show_patch_ori_offset_jnt(p0,p1,p2,offset,r0,cur_uvd,patch_size=patch_size,hand_width=hand_width,patch_pad_width=4)
        p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
        p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
        p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])
        # offset_uvd.shape = (offset_uvd.shape[0],offset_uvd.shape[1]*offset_uvd.shape[2])
        return p0, p1, p2, offset
    else:
        p0,p1,p2,new_cur_uvd,new_prev_uvd = crop_patch_enlarge(cur_uvd,prev_jnt_uvd_pred,prev_jnt_uvd_gr,r0,r1,r2,num_enlarge=num_enlarge,
                                                               patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=hand_pad_witht,batch_size=batch_size)

        offset = new_cur_uvd-new_prev_uvd
        print numpy.min(offset[:,0])
        print numpy.max(offset[:,0])
        print numpy.min(offset[:,1])
        print numpy.max(offset[:,1])
        print numpy.min(offset[:,2])
        print numpy.max(offset[:,2])
        # offset_uvd = norm_offset_uvd2(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd_pred, patch_size=patch_size,hand_width=96)

        show_patch_ori_offset_jnt(p0,p1,p2,offset,r0,new_cur_uvd,patch_size=patch_size,hand_width=hand_width,patch_pad_width=patch_pad_width)
        p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
        p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
        p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])

        return p0, p1, p2,offset
def load_data_patch_r0r1r2_mid_offset(path,pre_jnt_path,jnt_idx,patch_size=24, patch_pad_width=4,hand_width=96,pad_width=0):
    '''creat pathes based on ground truth
    htmap is a qunatized location for each joint
    '''

    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2 = f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    pre_idx_pred = (jnt_idx[0]-2)/4+1
    base_wrist_pred = numpy.load(pre_jnt_path)
    base_wrist_pred.shape=(base_wrist_pred.shape[0],6,3)
    prev_jnt_uvd_pred= numpy.squeeze(base_wrist_pred[:,pre_idx_pred,:])
    cur_uvd=numpy.squeeze(joint_label_uvd[:,jnt_idx,:])

    # for i in xrange(00,10,1):
    #     fig=plt.figure()
    #     ax=fig.add_subplot(131)
    #     ax.imshow(r0[i],'gray')
    #     plt.scatter(joint_label_uvd[i,5,0]*96,joint_label_uvd[i,5,1]*96,c='r')
    #     plt.scatter(prev_jnt_uvd[i,2,0]*96,prev_jnt_uvd[i,2,1]*96,c='g')
    #     ax=fig.add_subplot(132)
    #     ax.imshow(r1[i],'gray')
    #     ax=fig.add_subplot(133)
    #     ax.imshow(r2[i],'gray')
    #     plt.show()


    p0,p1,p2 = crop_patch(prev_jnt_uvd_pred,r0,r1,r2,patch_size=patch_size,patch_pad_width=patch_pad_width,hand_width=hand_width,pad_width=pad_width)
    offset = cur_uvd-prev_jnt_uvd_pred
    print numpy.min(offset[:,0])
    print numpy.max(offset[:,0])
    print numpy.min(offset[:,1])
    print numpy.max(offset[:,1])
    print numpy.min(offset[:,2])
    print numpy.max(offset[:,2])
    # offset_uvd = norm_offset_uvd2(cur_uvd=cur_uvd,prev_uvd=prev_jnt_uvd_pred, patch_size=patch_size,hand_width=96)

    # show_patch_offset_jnt(p0,p1,p2,offset,r0,joint_label_uvd[:,jnt_idx,:],patch_size=patch_size,patch_pad_width=4)
    p0.shape = (p0.shape[0], 1, p0.shape[1],p0.shape[2])
    p1.shape = (p1.shape[0], 1, p1.shape[1],p1.shape[2])
    p2.shape = (p2.shape[0], 1, p2.shape[1],p2.shape[2])

    return p0, p1, p2,offset

def create_save_patch():
    batch_size = 100
    # jnt_idx = [1,2,6,10,14,18]
    jnt_idx = [6]


    dataset = 'test'
    src_path = '../../data/source/'
    path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot.h5'%(src_path,dataset)
    # direct = '../../data/base_wrist_derot/best/'
    direct = '../../data/base_wrist_derot_enlarge/best/'
    prev_jnt_path ='%s%s_abs_base_wrist_r0r1r2_uvd_21jnts_derot_lg3_c0016_c0132_c1016_c1132_c2016_c2132_h16_h212_gm0_lm1000_yt5_ep295.npy'%(direct,dataset)
    patch_size=32
    patch_pad_width=4
    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_patch_r0r1r2_mid_offset(path,prev_jnt_path,jnt_idx=jnt_idx,patch_size=patch_size,
                                                                                                                  patch_pad_width=patch_pad_width,hand_width=96,pad_width=0)

    save_path_name = '%s%s_jnt%d_patch_base_wrist_r0r1r2_orioffset_21jnts_derot_lg3_c0016_c0132_c1016_c1132_c2016_c2132_h16_h212_gm0_lm1000_yt5_ep295.h5'%(direct,dataset,jnt_idx[0])
    f = h5py.File(save_path_name, 'w')
    f.create_dataset('r0', data=train_set_x0)
    f.create_dataset('r1', data=train_set_x1)
    f.create_dataset('r2', data=train_set_x2)
    f.create_dataset('offset', data=train_set_y)
    f.create_dataset('patch_size', data=patch_size)
    f.create_dataset('patch_pad_width', data=patch_pad_width)
    f.close()
    return

def  save_mid_patch(jnt_idx,patch_size,  offset_depth_range):
    # jnt_type='base' # jnt_type : base,mid, tip
    batch_size = 100
    # jnt_idx = [1,2,6,10,14,18]
    num_enlarge=0

    print 'offset_depth_range ',offset_depth_range
    model_info='mid%d_derot_r0r1r2_offset_21jnts_derot_lg%d_patch%d'%(jnt_idx[0],num_enlarge,patch_size)
    print model_info
    src_path = '../../data/source/'

    dataset = 'test'
    path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300.h5'%(src_path,dataset)
    direct = '../../data/final_xyz_center_wrist_base/msrc_r0r1r2_21jnts_u72v72d300_20151030/'
    prev_jnt_path ='%s%s_bw_r012_absuvd0_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep1000.npy'%(direct,dataset)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_mid_uvd_normalized(path,prev_jnt_path,jnt_idx=jnt_idx,
                                                                                        patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0,offset_depth_range=offset_depth_range)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    f = h5py.File('test_mid%d_patch%d_%.1f'%(jnt_idx[0],patch_size,offset_depth_range),'w')
    f.create_dataset('r0', data=test_set_x0)
    f.create_dataset('r1', data=test_set_x1)
    f.create_dataset('r2', data=test_set_x2)
    f.create_dataset('jnt_lable_uvd', data=test_set_y)
    f.close()



    dataset = 'train'
    path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300.h5'%(src_path,dataset)
    direct = '../../data/final_xyz_center_wrist_base/msrc_r0r1r2_21jnts_u72v72d300_20151030/'
    prev_jnt_path ='%s%s_bw_r012_absuvd0_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep1000.npy'%(direct,dataset)
    print 'prev_jnt_path',prev_jnt_path
    train_set_x0, train_set_x1,train_set_x2,train_set_y= load_data_multi_mid_uvd_normalized(path,prev_jnt_path,jnt_idx=jnt_idx,
                                                                                            patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0,offset_depth_range=offset_depth_range)
    n_train_batches = train_set_x0.shape[0]/ batch_size
    print 'n_train_batches', n_train_batches
    f = h5py.File('train_mid%d_patch%d_%.1f'%(jnt_idx[0],patch_size,offset_depth_range),'w')
    f.create_dataset('r0', data=train_set_x0)
    f.create_dataset('r1', data=train_set_x1)
    f.create_dataset('r2', data=train_set_x2)
    f.create_dataset('jnt_lable_uvd', data=train_set_y)
    f.close()




if __name__ =="__main__":
    save_mid_patch(jnt_idx=[6],patch_size=40,  offset_depth_range=0.8)

