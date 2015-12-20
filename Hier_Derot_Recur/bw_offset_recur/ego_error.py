__author__ = 'QiYE'
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import err_in_ori_xyz
import matplotlib.pyplot as plt
from src.utils.rotation import get_rot

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



def derot_err_uvd_xyz(setname,source_name,source_name_ori,dataset,uvd_pred_offset_path,jnt_idx ):

    src_path = '../../../data/%s/hier_derot_recur/bw_initial/best/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    pred_uvd_derot = f['pred_uvd_derot'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]
    pred_uvd = f['pred_uvd'][...]
    gr_uvd = f['gr_uvd'][...]
    rot = f['rotation'][...]
    f.close()

    src_path ='../../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name_ori)
    f = h5py.File(path,'r')
    r0 = numpy.squeeze(f['r0'][...][jnt_idx])
    r1 =  numpy.squeeze(f['r1'][...][jnt_idx])
    r2=  numpy.squeeze(f['r2'][...][jnt_idx])
    uvd_gr = f['joint_label_uvd'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]

    f.close()

    keypoints = scipy.io.loadmat('../../../data/%s/source/%s_%s_xyz_21joints.mat' % (setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('../../../data/%s/source/%s_%s_roixy_21joints.mat' % (setname,dataset,setname))
    roixy = keypoints['roixy']



    print jnt_idx
    idx = (jnt_idx[0]+3)/4
    print 'idx in bw',idx
    pred_uvd_derot.shape=(pred_uvd_derot.shape[0],6,3)
    pred_uvd.shape=(pred_uvd.shape[0],6,3)
    gr_uvd_derot.shape=(gr_uvd_derot.shape[0],6,3)
    gr_uvd.shape=(pred_uvd.shape[0],6,3)

    prev_jnt_uvd_derot = numpy.squeeze(pred_uvd_derot[:,idx,:])
    gr_jnt_uvd_derot = numpy.squeeze(gr_uvd_derot[:,idx,:])
    gr_offset=gr_jnt_uvd_derot-prev_jnt_uvd_derot

    print gr_offset.shape
    err = numpy.mean(numpy.sqrt(numpy.sum((gr_offset)**2,axis=-1)),axis=0)
    print 'err bf upd', err

    direct =  '../../../data/%s/hier_derot_recur/bw_offset/'%setname
    uvd_pred_offset =  numpy.load("%s%s%s.npy"%(direct,dataset,uvd_pred_offset_path))/10.0
    print uvd_pred_offset.shape

    cost = numpy.mean((numpy.sum((uvd_pred_offset-gr_offset)**2,axis=-1)),axis=0)
    print 'cost',cost
    # err = numpy.mean((numpy.sum((uvd_pred_offset+prev_jnt_uvd-gr_jnt_uvd)**2,axis=-1)),axis=0)
    # print 'arr af upd', err
    predict_uvd=uvd_pred_offset+prev_jnt_uvd_derot
    print predict_uvd.shape
    err = numpy.mean(numpy.sqrt(numpy.sum((gr_jnt_uvd_derot -predict_uvd )**2,axis=-1)),axis=0)
    print 'arr af upd', err
    # numpy.save("%s%s_abs_base_wrist_r0r1r2_uvd_21jnts_derot_lg3_c0016_c0132_c1016_c1132_c2016_c2132_h16_h212_gm0_lm1000_yt5_ep295.npy"%(direct,dataset),predict_uvd)
    """"rot the the norm view to original rotatioin view"""
    for i in xrange(uvd_gr.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        predict_uvd[i,0:2] = (numpy.dot(M,numpy.array([predict_uvd[i,0]*96,predict_uvd[i,1]*96,1]))-12)/72

        # plt.figure()
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # plt.imshow(dst,'gray')
        # plt.scatter(predict_uvd[i,0]*72+12,predict_uvd[i,1]*72+12)
        # plt.scatter(uvd_gr[i,jnt_idx,0]*72+12,uvd_gr[i,jnt_idx,1]*72+12,c='r')
        plt.show()
    err = numpy.mean(numpy.sqrt(numpy.sum((numpy.squeeze(uvd_gr[:,jnt_idx,:]) -predict_uvd )**2,axis=-1)),axis=0)
    print err
    xyz_pred = err_in_ori_xyz(setname,predict_uvd,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)
    # direct = '../../data/final_xyz_center_wrist_base/msrc_r0r1r2_21jnts_u72v72d300_20151030/'
    # numpy.save("%s%s_base_wrist_r0r1r2_xyz_21jnts_derot_lg3_c0016_c0132_c1016_c1132_c2016_c2132_h16_h212_gm0_lm1000_yt5_ep295.npy"%(direct,dataset),xyz_pred)


if __name__=='__main__':

    # derot_err_uvd_xyz(setname='nyu',
    #                              dataset='test',
    #                          source_name='_recur1_patch_uvd_derot_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm3000_yt0_ep815',
    #                          source_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #                              uvd_pred_offset_path='_uvd_bw0_r012_egoff_c0064_h11_h21_gm0_lm3000_yt0_ep655',
    #                              jnt_idx = [0])


    derot_err_uvd_xyz(setname='icvl',
                                 dataset='test',
                             source_name='_recur1_patch_uvd_derot_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep2380',
                             source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                                 uvd_pred_offset_path='_uvd_bw5_r012_egoff_c0032_h11_h22_gm0_lm1000_yt0_ep275',
                                 jnt_idx = [17])
    # jnt_idx = 0,1,5,9,13,17

