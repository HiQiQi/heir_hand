__author__ = 'QiYE'
import h5py
import numpy
dataset = 'train'
src_path = '../../data/source/'
# path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_20151030.h5'%(src_path,dataset)
# f = h5py.File(path,'r')
#
# r0 = f['r0'][...]
# r1 = f['r1'][...]
# r2= f['r2'][...]
# joint_label_uvd = f['joint_label_uvd'][...]
# depth_dmin_dmax = f['depth_dmin_dmax'][...]
# f.close()
# for i in xrange(100):
#     print numpy.max(r0[i])
#     print numpy.min(r0[i])
#
#     print depth_dmin_dmax[i]
#     print depth_dmin_dmax[i,1]-depth_dmin_dmax[i,0]
#
#     print numpy.max(joint_label_uvd[i,:,2])
#     print numpy.min(joint_label_uvd[i,:,2])
# print numpy.where(joint_label_uvd[:,:,2]>1)[0].shape[0]
# print numpy.where(joint_label_uvd[:,:,2]>1.1)[0].shape[0]
# print numpy.where(joint_label_uvd[:,:,2]>1.2)[0].shape[0]
# print numpy.where(joint_label_uvd[:,:,2]<0)[0].shape[0]
import scipy.io
keypoints = scipy.io.loadmat('../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/source/%s_uvd_msrc_21joints.mat' % dataset)
jnt_uvd = keypoints['uvd']
for i in xrange(jnt_uvd.shape[0]):
    if numpy.max(jnt_uvd[i, :, 2]) -  numpy.min(jnt_uvd[i, :, 2]) >210:
        print numpy.max(jnt_uvd[i, :, 2]) -  numpy.min(jnt_uvd[i, :, 2])
        print i
    if numpy.max(jnt_uvd[i, :, 0]) -  numpy.min(jnt_uvd[i, :, 0]) >150:
        print 'u'
        print numpy.max(jnt_uvd[i, :, 0]) -  numpy.min(jnt_uvd[i, :, 0])
        print i
    if numpy.max(jnt_uvd[i, :, 1]) -  numpy.min(jnt_uvd[i, :, 1]) >150:
        print 'v'
        print numpy.max(jnt_uvd[i, :, 1]) -  numpy.min(jnt_uvd[i, :, 1])
        print i