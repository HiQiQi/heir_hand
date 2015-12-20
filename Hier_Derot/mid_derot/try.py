__author__ = 'QiYE'
import h5py
import numpy
dataset = 'test'
src_path = '../../data/source/'
path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300.h5'%(src_path,dataset)
f = h5py.File(path,'r')
r0 = f['r0'][...]
r1 = f['r1'][...]
r2 = f['r2'][...]
joint_label_uvd = f['joint_label_uvd'][...]
f.close()


jnt_idx = [6]
cur_uvd_gr = numpy.squeeze(joint_label_uvd[:,jnt_idx,:])

direct = '../../data/final_xyz_center_wrist_base/msrc_r0r1r2_21jnts_u72v72d300_20151030/'
prev_jnt_path ='%s%s_base_wrist_r012_absuvd_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep1000.npy'%(direct,dataset)
prev_jnt_uvd_pred = numpy.load(prev_jnt_path)
pre_idx = [2]
prev_jnt_uvd =numpy.squeeze(joint_label_uvd[:,pre_idx,:])


offset=cur_uvd_gr - prev_jnt_uvd

print numpy.min(offset[:,2]),numpy.max(offset[:,2])
