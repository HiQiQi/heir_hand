__author__ = 'QiYE'
import h5py
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
import numpy
import scipy.io

name='roixy'
test_xyz_ori = scipy.io.loadmat('../../data/nyu/source/test_nyu_%s_21joints_ori.mat'%name)[name]
rs = cross_validation.ShuffleSplit(test_xyz_ori.shape[0], n_iter=1,test_size=2000, random_state=0)
print len(rs)
train_idx, test_idx = next(iter(rs))
print train_idx.shape, test_idx.shape
test_xyz = test_xyz_ori[test_idx]
scipy.io.savemat('../../data/nyu/source/test_nyu_%s_21joints.mat'%name,{name: test_xyz})


train_xyz_ori = scipy.io.loadmat('../../data/nyu/source/train_nyu_%s_21joints_ori.mat'%name)[name]
num=76712-train_xyz_ori.shape[0]
idx = train_idx[0:num]
train_xyz = new_r0 = numpy.concatenate([train_xyz_ori,test_xyz_ori[idx]])
scipy.io.savemat('../../data/nyu/source/train_nyu_%s_21joints.mat'%name, {name: train_xyz})



# src_path = '../../data/nyu/source/'
# path = '%stest_nyu_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300.h5'%(src_path)
# f = h5py.File(path,'r')
# test_r0 = f['r0'][...]
# test_r1 = f['r1'][...]
# test_r2= f['r2'][...]
# test_joint_label_uvd = f['joint_label_uvd'][...]
# bbox = f['bbox'][...]
# depth_dmin_dmax = f['depth_dmin_dmax'][...]
# margin_around_center_hand= f['margin_around_center_hand'][...]
# norm_center_hand_width = f['norm_center_hand_width'][...]
# orig_pad_border = f['orig_pad_border'][...]
#
#
# rs = cross_validation.ShuffleSplit(test_r0.shape[0], n_iter=1,test_size=2000, random_state=0)
#
# print len(rs)
#
# train_idx, test_idx = next(iter(rs))
# print train_idx.shape, test_idx.shape
#
# path = '%stest_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300.h5'%(src_path)
# f_shf = h5py.File(path,'w')
# print f.keys()
#
# f.close()
# f_shf.create_dataset('r0',data=test_r0[test_idx])
# f_shf.create_dataset('r1',data=test_r1[test_idx])
# f_shf.create_dataset('r2',data=test_r2[test_idx])
# f_shf.create_dataset('joint_label_uvd',data=test_joint_label_uvd[test_idx])
# f_shf.create_dataset('bbox',data=bbox[test_idx])
# f_shf.create_dataset('depth_dmin_dmax',data=depth_dmin_dmax[test_idx])
#
# f_shf.create_dataset('margin_around_center_hand',data=margin_around_center_hand)
# f_shf.create_dataset('norm_center_hand_width',data=norm_center_hand_width)
# f_shf.create_dataset('orig_pad_border',data=orig_pad_border)
#
# f_shf.create_dataset('ori_test_idx',data=test_idx)
# print f_shf.keys()
# f_shf.close()
#
#
# path = '%strain_nyu_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300.h5'%(src_path)
# f = h5py.File(path,'r')
# train_r0 = f['r0'][...]
# train_r1 = f['r1'][...]
# train_r2= f['r2'][...]
# train_joint_label_uvd = f['joint_label_uvd'][...]
# train_bbox = f['bbox'][...]
# train_depth_dmin_dmax = f['depth_dmin_dmax'][...]
# train_margin_around_center_hand= f['margin_around_center_hand'][...]
# train_norm_center_hand_width = f['norm_center_hand_width'][...]
# train_orig_pad_border = f['orig_pad_border'][...]
#
#
# num=76712-train_r0.shape[0]
# idx = train_idx[0:num]
# print idx.shape
# new_r0 = numpy.concatenate([train_r0,test_r0[idx]])
# new_r1 = numpy.concatenate([train_r1,test_r1[idx]])
# new_r2 = numpy.concatenate([train_r2,test_r2[idx]])
# new_uvd = numpy.concatenate([train_joint_label_uvd,test_joint_label_uvd[idx]])
# new_bbox = numpy.concatenate([train_bbox,bbox[idx]])
# new_depth_dmin_dmax = numpy.concatenate([train_depth_dmin_dmax,depth_dmin_dmax[idx]])
# path = '%strain_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300.h5'%(src_path)
# f_shf = h5py.File(path,'w')
#
# f_shf.create_dataset('r0',data=new_r0)
# f_shf.create_dataset('r1',data=new_r1)
# f_shf.create_dataset('r2',data=new_r2)
# f_shf.create_dataset('joint_label_uvd',data=new_uvd)
# f_shf.create_dataset('bbox',data=new_bbox)
# f_shf.create_dataset('depth_dmin_dmax',data=new_depth_dmin_dmax)
#
# f_shf.create_dataset('margin_around_center_hand',data=train_margin_around_center_hand)
# f_shf.create_dataset('norm_center_hand_width',data=train_norm_center_hand_width)
# f_shf.create_dataset('orig_pad_border',data=train_orig_pad_border)
#
# f_shf.create_dataset('ori_test_idx',data=idx)
#
# f_shf.close()
#
#
#
# print new_r0.shape
#
