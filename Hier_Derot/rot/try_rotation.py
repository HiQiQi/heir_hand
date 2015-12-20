__author__ = 'QiYE'

"""a snippet to find out that  rot_x, rot_y and rot_z in the msrc dataset stand for"""
import numpy
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as  plt
dataset = 'test'
src_path = '../../data/source/'
path = '%smsrc_%s_r0_r1_r2_uvd_bbox.h5'%(src_path,dataset)


f = h5py.File(path,'r')
r0 = f['r0'][...]
f.close()


rot = loadmat('C:/Proj/Proj_CNN_Hier/data/icvl/src/train_icvl_glb_rot_joints.mat')
glb_rot = rot['global_rot']

# loc = numpy.where(((glb_rot[:,0]>-.3) & (glb_rot[:,0]<.3)) & ((glb_rot[:,1]>2.2) & (glb_rot[:,1]<3.2))& ((glb_rot[:,2]>-0.3) & (glb_rot[:,2]<0.3)))
# print loc[0].shape
# for i in loc[0]:
#     print 'rotation', glb_rot[i]/3.1415926*180
#     plt.figure()
#     plt.imshow(r0[i],'gray')
#     plt.show()
#
# loc = numpy.where(((glb_rot[:,0]>-.3) & (glb_rot[:,0]<.3)) & ((glb_rot[:,1]>-3.2) & (glb_rot[:,1]<-2.5))& ((glb_rot[:,2]>-0.3) & (glb_rot[:,2]<0.3)))
# # & (glb_rot[:,0]<0.005) & (glb_rot[:,0]>0)) &(glb_rot[:,0]<0.005) & (glb_rot[:,0]>0))
# print loc[0].shape
# for i in loc[0]:
#     print 'rotation', glb_rot[i]/3.1415926*180
#     plt.figure()
#     plt.imshow(r0[i],'gray')
#     plt.show()

# loc = numpy.where(((glb_rot[:,2]>-0.3) & (glb_rot[:,2]<0.3)))
# for i in loc[0]:
#     print 'rotation', glb_rot[i]/3.1415926*180
#     plt.figure()
#     plt.imshow(r0[i],'gray')
#     plt.show()
loc = numpy.where(((glb_rot[:,2]>-0.4) & (glb_rot[:,2]<0.4)))
for i in loc[0]:
    print 'rotation', glb_rot[i]/3.1415926*180
    plt.figure()
    plt.imshow(r0[i],'gray')
    plt.show()
