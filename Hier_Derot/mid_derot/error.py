__author__ = 'QiYE'
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import err_in_ori_xyz
import matplotlib.pyplot as plt

def offset_to_abs(off_uvd, pre_uvd,patch_size=44,offset_depth_range=1.0,hand_width=96):

    if len(off_uvd.shape)<3:
        off_uvd[:,0:2] = (off_uvd[:,0:2]*patch_size -patch_size/2 )/hand_width
        # off_uvd[:,0:2] = (off_uvd[:,0:2]*72+12)/24
        predict_uvd= numpy.empty_like(off_uvd)
        predict_uvd[:,0:2] = pre_uvd[:,0:2]+off_uvd[:,0:2]
        off_uvd[:,2] = (off_uvd[:,2]-0.5)*offset_depth_range
        predict_uvd[:,2] = pre_uvd[:,2]+off_uvd[:,2]
        return predict_uvd
    else:
        pre_uvd.shape=(pre_uvd.shape[0],1,pre_uvd.shape[-1])
        off_uvd[:,:,0:2] = (off_uvd[:,:,0:2]*patch_size -patch_size/2 )/hand_width
        # off_uvd[:,0:2] = (off_uvd[:,0:2]*72+12)/24
        predict_uvd= numpy.empty_like(off_uvd)
        predict_uvd[:,:,0:2] = pre_uvd[:,:,0:2]+off_uvd[:,:,0:2]
        off_uvd[:,:,2] = (off_uvd[:,:,2]-0.5)*offset_depth_range
        predict_uvd[:,:,2] = pre_uvd[:,:,2]+off_uvd[:,:,2]
        return predict_uvd



def mid_derot_err_uvd_xyz(dataset,setname,dataset_path_prefix,source_name,source_name_ori,prev_jnt_name,pred_save_name,jnt_idx,patch_size=24,offset_depth_range=0.8):

    src_path =   '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    rot = f['rotation'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    f.close()

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']


    path = '%s%s%s.h5'%(src_path,dataset,source_name_ori)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    f.close()

    direct ='%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    pre_idx_pred = (jnt_idx[0]-2)/4+1
    print (jnt_idx[0]-2)/4+1
    prev_jnt_uvd = numpy.squeeze(numpy.load(prev_jnt_path)[:,pre_idx_pred,:])
    print prev_jnt_uvd.shape

    direct =  '%sdata/%s/hier_derot/mid/best/'%(dataset_path_prefix,setname)
    # direct =  '../../data/%s/base_wrist_spacial/'%setname
    uvd_pred_offset =  numpy.load("%s%s_offset%s.npy"%(direct,dataset,pred_save_name))
    print uvd_pred_offset.shape

    predict_uvd = offset_to_abs(uvd_pred_offset, prev_jnt_uvd,patch_size=patch_size,offset_depth_range=offset_depth_range)


    #
    direct ='%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    numpy.save("%s%s_absuvd0%s.npy"%(direct,dataset,pred_save_name),predict_uvd)
    """"rot the the norm view to original rotatioin view"""
    for i in xrange(uvd_gr.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        # plt.figure()
        # plt.imshow(r0[i],'gray')
        # plt.scatter(predict_uvd[i,0]*96,predict_uvd[i,1]*96)
        predict_uvd[i,0:2] = (numpy.dot(M,numpy.array([predict_uvd[i,0]*96,predict_uvd[i,1]*96,1]))-12)/72

        # plt.figure()
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # plt.imshow(dst,'gray')
        # plt.scatter(predict_uvd[i,0]*72+12,predict_uvd[i,1]*72+12)
        # plt.scatter(uvd_gr[i,jnt_idx,0]*72+12,uvd_gr[i,jnt_idx,1]*72+12,c='r')
        # plt.show()

    xyz_pred = err_in_ori_xyz(setname,predict_uvd,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)
    direct ='%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    numpy.save("%s%s_xyz%s.npy"%(direct,dataset,pred_save_name),xyz_pred)
    numpy.save("%s%s_absuvd%s.npy"%(direct,dataset,pred_save_name),predict_uvd)


if __name__=='__main__':

    mid_derot_err_uvd_xyz(dataset='test',
                          setname='icvl',
                          dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
                             source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                             source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                             prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
                                 pred_save_name='_mid14_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep395',
                                 patch_size=40,
                                 offset_depth_range=0.4,
                                 jnt_idx = [14])


    #
    # mid_derot_err_uvd_xyz(dataset='test',
    #                       setname='icvl',
    #                       dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #                          source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
    #                              pred_save_name='_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep385',
    #                              patch_size=40,
    #                              offset_depth_range=0.4,
    #                              jnt_idx = [10])
    #
    # mid_derot_err_uvd_xyz(dataset='test',
    #                       setname='icvl',
    #                       dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #                          source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
    #                              pred_save_name='_mid6_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep305',
    #                              patch_size=40,
    #                              offset_depth_range=0.4,
    #                              jnt_idx = [6])
    #
    #
    #
    # mid_derot_err_uvd_xyz(dataset='test',
    #                       setname='icvl',
    #                       dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #                          source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
    #                              pred_save_name='_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep385',
    #                              patch_size=40,
    #                              offset_depth_range=0.4,
    #                              jnt_idx = [10])


    # mid_derot_err_uvd_xyz(dataset='train',
    #                       setname='icvl',
    #                       dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #                          source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          prev_jnt_name='_absuvd0_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep1935',
    #                              pred_save_name='_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep100',
    #                              patch_size=40,
    #                              offset_depth_range=0.4,
    #                              jnt_idx = [2])


    # mid_derot_err_uvd_xyz(dataset='test',
    #                       setname='nyu',
    #                       dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #                          source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #                          source_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #                          prev_jnt_name='_absuvd0_base_wrist_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep2000',
    #                              pred_save_name='_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1770',
    #                              patch_size=40,
    #                              offset_depth_range=0.6,
    #                              jnt_idx = [2])
