from src.utils import constants

__author__ = 'QiYE'
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import err_in_ori_xyz


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



def top_derot_err_uvd_xyz(dataset,setname,dataset_path_prefix,source_name,source_name_ori,
                          jnt_idx,prev_jnt_name,patch_size,offset_depth_range,uvd_pred_offset_path,final_save_name):

    src_path =  '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path ='%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    rot = f['rotation'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    f.close()

    path ='%s%s%s.h5'%(src_path,dataset,source_name_ori)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    f.close()

    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_xyz_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('%sdata/%s/source/%s_%s_roixy_21joints.mat' % (dataset_path_prefix,setname,dataset,setname))
    roixy = keypoints['roixy']


    direct = '%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    prev_jnt_uvd = numpy.load(prev_jnt_path)

    direct = '%sdata/%s/hier_derot/top/best/'%(dataset_path_prefix,setname)
    uvd_pred_offset =  numpy.load("%s%s%s.npy"%(direct,dataset,uvd_pred_offset_path))
    uvd_pred_offset.shape=(uvd_pred_offset.shape[0],3)
    # predict_uvd=uvd_pred_offset+prev_jnt_uvd
    predict_uvd = offset_to_abs(uvd_pred_offset, prev_jnt_uvd,patch_size=patch_size,offset_depth_range=offset_depth_range)

    direct = '%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    numpy.save("%s%s_absuvd0%s.npy"%(direct,dataset,final_save_name),predict_uvd)
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
    direct = '%sdata/%s/hier_derot/final_xyz_uvd/'%(dataset_path_prefix,setname)
    numpy.save("%s%s_xyz%s.npy"%(direct,dataset,final_save_name),xyz_pred)
    numpy.save("%s%s_absuvd%s.npy"%(direct,dataset,final_save_name),predict_uvd)

if __name__=='__main__':

    # icvl
    mid_jnt_name=[]
    mid_jnt_name.append('_absuvd0_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep100')
    mid_jnt_name.append('_absuvd0_mid6_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep305')
    mid_jnt_name.append('_absuvd0_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep385')
    mid_jnt_name.append('_absuvd0_mid14_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep395')
    mid_jnt_name.append('_absuvd0_mid18_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep605')



    idx = 7
    top_derot_err_uvd_xyz(dataset='train',
               setname='icvl',
            dataset_path_prefix=constants.Data_Path,
                             source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                             source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
               jnt_idx=[idx],
                patch_size=40,
                offset_depth_range=0.4,
               prev_jnt_name=mid_jnt_name[(idx-3)/4],
                         uvd_pred_offset_path = '_offset_top7_r012_21jnts_derot_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep465',
                         final_save_name='_top7_r012_21jnts_derot_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm200_yt0_ep465')



    # mid_jnt_name=[]
    # mid_jnt_name.append('_absuvd0_mid2_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep400')
    # mid_jnt_name.append('_absuvd0_mid6_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep600')
    # mid_jnt_name.append('_absuvd0_mid10_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep500')
    # mid_jnt_name.append('_absuvd0_mid14_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep450')
    # mid_jnt_name.append('_absuvd0_mid18_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep200')
    # jnt_idx=19
    # top_derot_err_uvd_xyz(dataset='train',
    #                       setname='msrc',
    #                       prev_jnt_name=mid_jnt_name[(jnt_idx-3)/4],
    #                              uvd_pred_offset_path='_top19_offset_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm400_yt5_ep360',
    #                              final_save_name='_top19_r012_21jnts_derot_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm400_yt5_ep360',
    #                              patch_size=40,
    #                              offset_depth_range=0.8,
    #                              jnt_idx = [jnt_idx])
