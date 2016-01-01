__author__ = 'QiYE'
from src.utils import constants
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import err_in_ori_xyz
import matplotlib.pyplot as plt
def whole_derot_err_uvd_xyz(dataset, dataset_path_prefix,source_name,souce_name_ori,setname,pred_save_name):
    jnt_idx = range(0,21,1)
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    rot = f['rotation'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]

    derot_uvd = f['joint_label_uvd'][...]

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']

    f.close()

    path = '%s%s%s.h5'%(src_path,dataset,souce_name_ori)
    f = h5py.File(path,'r')
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    print joint_label_uvd.shape

    pred_path = '%sdata/%s/whole_derot/best/'%(dataset_path_prefix,setname)
    path = '%s%s%s.npy'%(pred_path,dataset,pred_save_name)
    whole_pred = numpy.load(path)
    whole_pred.shape=(whole_pred.shape[0],21,3)

    print whole_pred.shape
    print derot_uvd.shape

    err_uvd = numpy.mean(numpy.sqrt(numpy.sum(((whole_pred -derot_uvd)**2),axis=-1)),axis=0)
    print 'norm error', err_uvd.mean()


    for i in xrange(joint_label_uvd.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        # plt.figure()
        # plt.imshow(r0[i],'gray')
        # plt.scatter(whole_pred[i,:,0]*96,whole_pred[i,:,1]*96)
        # plt.scatter(derot_uvd[i,:,0]*96,derot_uvd[i,:,1]*96,c='r')
        for j in xrange(len(jnt_idx)):
            # whole_pred[i,j,0:2] = numpy.dot(M,numpy.array([whole_pred[i,j,0]*96,whole_pred[i,j,1]*96,1]))
            whole_pred[i,j,0:2] = (numpy.dot(M,numpy.array([whole_pred[i,j,0]*96,whole_pred[i,j,1]*96,1]))-12)/72

        # plt.figure()
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # plt.imshow(dst,'gray')
        # plt.scatter(whole_pred[i,:,0]*72+12,whole_pred[i,:,1]*72+12)
        # plt.scatter(joint_label_uvd[i,:,0]*72+12,joint_label_uvd[i,:,1]*72+12,c='r')
        # plt.scatter(tmp_uvd[i,:,0]*72+12,tmp_uvd[i,:,1]*72+12,c='g')
        # plt.show()

    err_uvd = numpy.mean(numpy.sqrt(numpy.sum((whole_pred -joint_label_uvd)**2,axis=-1)),axis=0)

    print 'ori norm error', err_uvd.mean()
    err_ori_xyz = err_in_ori_xyz(setname,whole_pred,joint_label_uvd,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)

if __name__=='__main__':
    whole_derot_err_uvd_xyz(dataset='test',setname='icvl',
                            dataset_path_prefix=constants.Data_Path,
                source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                souce_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                pred_save_name = '_whole_derot_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm400_yt0_ep1740')

    # whole_derot_err_uvd_xyz(dataset='test',setname='nyu',
    #             source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             souce_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             pred_save_name = '_whole_derot_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm400_yt0_ep925')

    # whole_derot_err_uvd_xyz(dataset='test',setname='msrc',
    #             source_name='_msrc_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300',
    #             souce_name_ori='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #             pred_save_name = '_whole_derot_21jnts_r012_conti_c0064_c01128_c1064_c11128_c2064_c21128_h18_h232_gm0_lm600_yt0_ep500')