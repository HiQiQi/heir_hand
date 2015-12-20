__author__ = 'QiYE'
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import err_in_ori_xyz
import matplotlib.pyplot as plt
def whole_err_uvd_xyz(dataset,souce_name_ori,setname,pred_save_name):
    jnt_idx = range(0,21,1)
    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,souce_name_ori)
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]

    orig_pad_border=f['orig_pad_border'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    drange = depth_dmin_dmax[:,1]-depth_dmin_dmax[:,0]
    pred_path = '../../data/%s/whole/best/'%setname
    path = '%s%s%s.npy'%(pred_path,dataset,pred_save_name)
    whole_pred = numpy.load(path)
    whole_pred.shape=(whole_pred.shape[0],21,3)

    err_uvd = numpy.mean(numpy.sqrt(numpy.sum((whole_pred -joint_label_uvd)**2,axis=-1)),axis=0)
    print 'norm error', err_uvd
    for i in numpy.random.randint(0,r0.shape[0],5):
        plt.figure()
        plt.imshow(r0[i],'gray')
        plt.scatter(whole_pred[i,:,0]*72+12,whole_pred[i,:,1]*72+12)
        plt.scatter(joint_label_uvd[i,:,0]*72+12,joint_label_uvd[i,:,1]*72+12,c='r')
        plt.show()

    keypoints = scipy.io.loadmat('../../data/%s/source/%s_%s_xyz_21joints.mat' % (setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('../../data/%s/source/%s_%s_roixy_21joints.mat' %(setname, dataset,setname))
    roixy = keypoints['roixy']
    err_ori_xyz = err_in_ori_xyz(setname,whole_pred,joint_label_uvd,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)

if __name__=='__main__':
    whole_err_uvd_xyz(dataset='test',setname='nyu',
                souce_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
                pred_save_name = '_whole_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm300_yt0_ep905')

    # whole_err_uvd_xyz(dataset='test',setname='icvl',
    #             source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             souce_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             pred_save_name = '_whole_21jnts_r012_conti_c0016_c0132_c1016_c1132_c2016_c2132_h18_h232_gm0_lm600_yt0_ep1360')