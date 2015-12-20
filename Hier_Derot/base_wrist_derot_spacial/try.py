__author__ = 'QiYE'
import h5py
import numpy
dataset = 'test'
# src_path = '../../data/source/'
# path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300.h5'%(src_path,dataset)
#
# direct = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/center_derot/best/'
# prev_jnt_path ='%s%s_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770.npy'%(direct,dataset)
# f = h5py.File(path,'r')
# r0 = f['r0'][...]
# r1 = f['r1'][...]
# r2 = f['r2'][...]
# joint_label_uvd = f['joint_label_uvd'][...]
# f.close()


# prev_jnt_uvd_pred = numpy.load(prev_jnt_path)
# pre_idx = [0,9]
# prev_jnt_uvd_gr =numpy.mean(joint_label_uvd[:,pre_idx,:],axis=1)
# jnt_idx = [0,1,5,9 ,13,17]
# cur_uvd = joint_label_uvd[:,jnt_idx,:]
# offset=cur_uvd-prev_jnt_uvd_pred.reshape((prev_jnt_uvd_pred.shape[0],1,prev_jnt_uvd_pred.shape[1]))
# for i in xrange(6):
#     print numpy.min(offset[:,i,2]),numpy.max(offset[:,i,2])




# import theano.tensor as T
# import theano
# Y_ = T.matrix()
# Y = T.matrix()
# dist = T.sum(T.sqr(Y_[:,:]-Y[:,:]),axis=-1)
# regu =T.switch(1-dist>0,0,1-dist)
# test = theano.function([Y_,Y],[dist,regu])
# a=numpy.ones((3,2),dtype='float32')
# a[2,:]=0
# b=numpy.zeros((3,2),dtype='float32')
# o1,o2= test(b,a)
# print o1,o2

# import theano.tensor as T
# import theano
# import numpy
# Y_ = T.matrix()
# Y = T.matrix()
# v0 = Y_[:,:]
# nv0 =T.sqrt(T.sum(v0*v0,axis=-1))
# v1 = Y[:,:]
# nv1 = T.sqrt(T.sum(v1*v1,axis=-1))
# inner = T.abs_(T.sum(v1*v0,axis=-1))
# dist = inner / nv0/nv1
# regu = T.switch(dist > 0.55,0,dist)
#
# test = theano.function([Y_,Y],[inner,nv0,nv1,dist,regu])
# a=numpy.ones((3,4),dtype='float32')
# b=numpy.ones((3,4),dtype='float32')*0.5
# o1,o2,o3,o4,o5= test(a,b)
# print o1,o2,o3,o4,o5
#
#
#
# import scipy.io
# keypoints = scipy.io.loadmat('../../data/joint_uvd/xyz_msrc_%s_22joints.mat' % dataset)
# jnt_idx = [1,2,6,10,14,18]
# xyz_true = keypoints['xyz'][:,jnt_idx,:]
#
# v0 = xyz_true[:,0,:] - xyz_true[:,3,:]
# nv0 =numpy.sqrt(numpy.sum(v0*v0,axis=-1))
# v1 = xyz_true[:,2,:] - xyz_true[:,3,:]
# nv1 = numpy.sqrt(numpy.sum(v1*v1,axis=-1))
# ang_cos = numpy.abs(numpy.divide(numpy.sum(v1*v0,axis=-1),nv0*nv1))
# print numpy.mean(ang_cos)
# print numpy.arccos(numpy.mean(ang_cos))/3.1415926*180
# print numpy.min(ang_cos)
# print numpy.arccos(numpy.min(ang_cos))/3.1415926*180
# print numpy.max(ang_cos)
# print numpy.arccos(numpy.max(ang_cos))/3.1415926*180
# print '\n'
# direct = '../../data/msrc_r0r1r2_21jnts_u72v72d300_20151030/'
# xyz_center    numpy.save("%s%s_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770.npy"%(direct,dataset),xyz_pred)
# xyz_pred = numpy.load("%s_abs_base_wrist_r0r1r2_uvd_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep655.npy"%(dataset))
#
# v0 = xyz_pred[:,0,:] - xyz_pred[:,3,:]
# nv0 =numpy.sqrt(numpy.sum(v0*v0,axis=-1))
# v1 = xyz_pred[:,2,:] - xyz_pred[:,3,:]
# nv1 = numpy.sqrt(numpy.sum(v1*v1,axis=-1))
# ang_cos = numpy.abs(numpy.divide(numpy.sum(v1*v0,axis=-1),nv0*nv1))
# print numpy.mean(ang_cos)
# print numpy.arccos(numpy.mean(ang_cos))/3.1415926*180
# print numpy.min(ang_cos)
# print numpy.arccos(numpy.min(ang_cos))/3.1415926*180
# print numpy.max(ang_cos)
# print numpy.arccos(numpy.max(ang_cos))/3.1415926*180
# loc =  numpy.where(ang_cos>0.5)
# print loc[0].shape
# for i in xrange(50):
#     idx = loc[0][i]
#     jnt_idx=[0,3,2]
#     print numpy.arccos(ang_cos[idx])/3.1415926*180
#     diff = xyz_true[idx]-xyz_pred[idx]
#     print numpy.abs(diff)[jnt_idx]
#     print '32 pred',numpy.sqrt(numpy.sum((xyz_pred[idx][3]-xyz_pred[idx][2])**2))
#     print '32 gr',numpy.sqrt(numpy.sum((xyz_true[idx][3]-xyz_true[idx][2])**2))
#     print '30 pred',numpy.sqrt(numpy.sum((xyz_pred[idx][3]-xyz_pred[idx][0])**2))
#     print '30 gr',numpy.sqrt(numpy.sum((xyz_true[idx][3]-xyz_true[idx][0])**2))