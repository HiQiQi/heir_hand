__author__ = 'QiYE'
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import err_in_ori_xyz
import matplotlib.pyplot as plt
def center_derot_err_uvd_xyz(dataset,setname,source_name,source_name_ori,pred_save_name):
    # dataset='test'
    # src_path = '../../data/source/'
    # path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot.h5'%(src_path,dataset)
    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    rot = f['rotation'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    derot_uvd = f['joint_label_uvd'][...]

    keypoints = scipy.io.loadmat('../../data/%s/source/%s_%s_xyz_21joints.mat' % (setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('../../data/%s/source/%s_%s_roixy_21joints.mat' % (setname,dataset,setname))
    roixy = keypoints['roixy']

    f.close()

    path = '%s%s%s.h5'%(src_path,dataset,source_name_ori)
    f = h5py.File(path,'r')
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    # jnt_idx = [0,1,5,9 ,13,17]
    jnt_idx = [0,9 ]
    # idx = [1,2,6,10 ,14,18]
    print jnt_idx
    center_gr = numpy.mean(joint_label_uvd[:,jnt_idx,:],axis=1)
    derot_center_gr = numpy.mean(derot_uvd[:,jnt_idx,:],axis=1)


    pred_path = '../../data/%s/center_derot/best/'%setname
    path = '%s%s_uvd%s.npy'%(pred_path,dataset,pred_save_name)
    center_pred = numpy.load(path)
    print numpy.mean(numpy.sqrt(numpy.sum((center_pred-derot_center_gr)**2,axis=-1)),axis=0)
    direct ='../../data/%s/final_xyz_uvd/'%setname
    numpy.save("%s%s_absuvd0%s.npy"%(direct,dataset,pred_save_name),center_pred)
    for i in xrange(joint_label_uvd.shape[0]):
        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(center_pred[i,0]*96,center_pred[i,1]*96)
        # plt.scatter(derot_center_gr[i,0]*96,derot_center_gr[i,1]*96,c='g')
        # plt.scatter(derot_uvd[i,:,0]*96,derot_uvd[i,:,1]*96,c='r')

        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        center_pred[i,0:2] = numpy.dot(M,numpy.array([center_pred[i,0]*96,center_pred[i,1]*96,1]))
        rot_center_gr = numpy.dot(M,numpy.array([derot_center_gr[i,0]*96,derot_center_gr[i,1]*96,1]))
        # depth_in_r0 = r0[i,center_pred[i,1],center_pred[i,1]]
        # diff = abs(depth_in_r0-center_pred[i,2])
        # if abs(depth_in_r0-center_pred[i,2])>0.2:
        #     center_pred[i,2] = depth_in_r0

        center_pred[i,0:2] = (center_pred[i,0:2]-12)/72

        #
        # ax = fig.add_subplot(122)
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # ax.imshow(dst,'gray')
        # plt.scatter(center_pred[i,0]*72+12,center_pred[i,1]*72+12)
        # plt.scatter(center_gr[i,0]*72+12,center_gr[i,1]*72+12,c='g')
        # plt.scatter(rot_center_gr[0],rot_center_gr[1],c='w')
        # plt.scatter(joint_label_uvd[i,:,0]*72+12,joint_label_uvd[i,:,1]*72+12,c='r')
        # plt.show()


    xyz_pred = err_in_ori_xyz(setname,center_pred,joint_label_uvd,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type='center',jnt_idx=jnt_idx)
    direct = '../../data/%s/final_xyz_uvd/'%setname
    numpy.save("%s%s_xyz%s.npy"%(direct,dataset,pred_save_name),xyz_pred)
    numpy.save("%s%s_absuvd%s.npy"%(direct,dataset,pred_save_name),center_pred)
if __name__=='__main__':
    # center_derot_err_uvd_xyz(dataset = 'train',
    #                          setname='icvl',
    #                          source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          source_name_ori='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #                          pred_save_name='_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep1445')

    center_derot_err_uvd_xyz(dataset = 'train',setname='nyu',
                             source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
                             source_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
                             pred_save_name='_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep1000')