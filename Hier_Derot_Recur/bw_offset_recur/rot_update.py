__author__ = 'QiYE'
import numpy
import h5py
from math import pi


def get_rot(joint_label_uvd,setname,i,j):

    if setname=='nyu':
        vect = joint_label_uvd[:,i,0:2] - joint_label_uvd[:,j,0:2]#the index is valid for 21joints
    if setname=='msrc':
        vect = joint_label_uvd[:,j,0:2] - joint_label_uvd[:,i,0:2]#the index is valid for 21joints

    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/pi*180)
    print numpy.where(rot==180)[0].shape[0]
    rot[numpy.where(rot==180)] =179
    return rot


setname= 'nyu'
dataset='test'

# prev_jnt_name="_whole_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm300_yt0_ep885"
# prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
# prev_jnt_uvd = numpy.load(prev_jnt_path)
# prev_jnt_uvd.shape = (prev_jnt_uvd.shape[0],21,3)
# upd_rot = get_rot(prev_jnt_uvd,setname,0,9)
direct = '../../../data/%s/hier_derot_recur/bw_offset/best/'%setname
path ='%s%s_iter2_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020.h5'%(direct,dataset)
"""for msrc dataset and nyu dataset"""
f = h5py.File(path,'r')
pred_uvd_iter2 = f['pred_uvd'][...]
gr_uvd_iter2 = f['gr_uvd'][...]
pred_rot_iter2 = f['rotation'][...]
f.close()
pred_uvd_iter2.shape = (pred_uvd_iter2.shape[0],6,3)
gr_uvd_iter2.shape = (gr_uvd_iter2.shape[0],6,3)
pred_rot_iter2=get_rot(pred_uvd_iter2,'nyu',0,3)

direct = '../../../data/%s/hier_derot_recur/bw_initial/best/'%setname
path ='%s%s_recur1_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020.h5'%(direct,dataset)
"""for msrc dataset and nyu dataset"""
f = h5py.File(path,'r')
pred_rot_iter1 = f['rotation'][...]
f.close()



direct = '../../../data/%s/source/'%setname
path ='%s%s_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300.h5'%(direct,dataset)
"""for msrc dataset and nyu dataset"""

f = h5py.File(path,'r')
pred_rot = f['rotation'][...]
f.close()

direct = '../../../data/%s/source/'%setname
path ='%s%s_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300.h5'%(direct,dataset)
f = h5py.File(path,'r')
r1 = f['r1'][...]
r2= f['r2'][...]
joint_label_uvd = f['joint_label_uvd'][...]
f.close()
gr_rot = get_rot(joint_label_uvd,setname,0,9)

upd_gr_err_iter1= numpy.abs(pred_rot_iter1-gr_rot).mean()
upd_gr_err_iter2 = numpy.abs(pred_rot_iter2-gr_rot).mean()
pred_gr_err = numpy.abs(pred_rot-gr_rot).mean()

print 'iter1',upd_gr_err_iter1
print 'iter2',upd_gr_err_iter2
print 'rot pred by cnn',pred_gr_err