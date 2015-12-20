__author__ = 'QiYE'
from math import pi

import numpy
import cv2
from sklearn.preprocessing import normalize


def get_rot(joint_label_uvd,i,j):

    vect = joint_label_uvd[:,i,0:2] - joint_label_uvd[:,j,0:2]#the index is valid for 21joints

    # print vect_prod.shape
    vect_norm = numpy.squeeze(normalize(vect,norm='l2',axis=1))
    vect_prod =numpy.dot(vect_norm,(0,1))
    # print prod_norm.shape
    rot = numpy.arccos(vect_prod)

    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/pi*180)
    # print numpy.where(rot==180)[0].shape[0]
    rot[numpy.where(rot==180)] =179
    return rot

def rot_loc(pred_uvd, gr_uvd,rotation):
    pred_uvd_derot = numpy.empty_like(pred_uvd)
    pred_uvd_derot[:,:,2]=pred_uvd[:,:,2]
    gr_uvd_derot = numpy.empty_like(gr_uvd)
    gr_uvd_derot[:,:,2]=gr_uvd[:,:,2]

    for i in xrange(0,gr_uvd.shape[0],1):
        M = cv2.getRotationMatrix2D((48,48),-rotation[i],1)
        for j in xrange(gr_uvd.shape[1]):
            gr_uvd_derot[i,j,0:2] = numpy.dot(M,numpy.array([gr_uvd[i,j,0]*72+12,gr_uvd[i,j,1]*72+12,1]))/96
            pred_uvd_derot[i,j,0:2] = numpy.dot(M,numpy.array([pred_uvd[i,j,0]*72+12,pred_uvd[i,j,1]*72+12,1]))/96
    return pred_uvd_derot,gr_uvd_derot

def rot_img(r0,r1,r2,rotation):

    c0=r0.shape[1]
    c1=r1.shape[1]
    c2=r2.shape[2]

    for i in xrange(0,r0.shape[0],1):
        M = cv2.getRotationMatrix2D((c0/2,c0/2),-rotation[i],1)
        r0[i] = cv2.warpAffine(r0[i],M,(c0,c0),borderValue=1)


        M = cv2.getRotationMatrix2D((c1/2,c1/2),-rotation[i],1)
        r1[i] = cv2.warpAffine(r1[i],M,(c1,c1),borderValue=1)

        M = cv2.getRotationMatrix2D((c2/2,c2/2),-rotation[i],1)
        r2[i] = cv2.warpAffine(r2[i],M,(c2,c2),borderValue=1)


    return

def rot_conv_img(r0,r1,r2,pred_uvd,gr_uvd ,rotation,batch_size):
    num_ft= r0.shape[1]

    c0=r0.shape[-1]
    c1=r1.shape[-1]
    c2=r2.shape[-1]

    conv00_kern_size=5
    conv00_pool_size=4
    conv10_kern_size=5
    conv10_pool_size=2
    conv20_kern_size=5
    conv20_pool_size=2

    pred_patch_center=numpy.empty((3,pred_uvd.shape[0],pred_uvd.shape[1],pred_uvd.shape[2]),dtype='float32')
    gr_patch_center=numpy.empty((3,pred_uvd.shape[0],pred_uvd.shape[1],pred_uvd.shape[2]),dtype='float32')

    for i in xrange(0,batch_size,1):
        M = cv2.getRotationMatrix2D((c0/2,c0/2),-rotation[i],1)
        for j in xrange(num_ft):
            r0[i,j] = cv2.warpAffine(r0[i,j],M,(c0,c0),borderValue=1)

        for j in xrange(gr_uvd.shape[1]):
            tmp=(pred_uvd[i,j]*72+12-conv00_kern_size/2)/conv00_pool_size
            pred_patch_center[0,i,j,0:2] = numpy.dot(M,numpy.array([tmp[0],tmp[1],1]))/c0

            tmp=(gr_uvd[i,j]*72+12-conv00_kern_size/2)/conv00_pool_size
            gr_patch_center[0,i,j,0:2] = numpy.dot(M,numpy.array([tmp[0],tmp[1],1]))/c0


        M = cv2.getRotationMatrix2D((c1/2,c1/2),-rotation[i],1)
        for j in xrange(num_ft):
            r1[i,j] = cv2.warpAffine(r1[i,j],M,(c1,c1),borderValue=1)

        for j in xrange(gr_uvd.shape[1]):
            tmp=((pred_uvd[i,j]*72+12)/2-conv10_kern_size/2)/conv10_pool_size
            pred_patch_center[1,i,j,0:2] = numpy.dot(M,numpy.array([tmp[0],tmp[1],1]))/c1

            tmp=((gr_uvd[i,j]*72+12)/2-conv10_kern_size/2)/conv10_pool_size
            gr_patch_center[1,i,j,0:2] = numpy.dot(M,numpy.array([tmp[0],tmp[1],1]))/c1


        M = cv2.getRotationMatrix2D((c2/2,c2/2),-rotation[i],1)
        for j in xrange(num_ft):
            r2[i,j] = cv2.warpAffine(r2[i,j],M,(c2,c2),borderValue=1)

        for j in xrange(gr_uvd.shape[1]):
            tmp=((pred_uvd[i,j]*72+12)/4-conv20_kern_size/2)/conv20_pool_size
            pred_patch_center[2,i,j,0:2] = numpy.dot(M,numpy.array([tmp[0],tmp[1],1]))/c2

            tmp=((gr_uvd[i,j]*72+12)/4-conv20_kern_size/2)/conv20_pool_size
            gr_patch_center[2,i,j,0:2] = numpy.dot(M,numpy.array([tmp[0],tmp[1],1]))/c2

    return pred_patch_center,gr_patch_center

def recur_derot(r0,r1,r2,pred_uvd,gr_uvd,batch_size):
    rotation = get_rot(pred_uvd,0,3)
    pred_uvd_derot, gr_uvd_derot = rot_loc(pred_uvd,gr_uvd,rotation)

    pred_patch_center,gr_patch_center=rot_conv_img(r0,r1,r2,pred_uvd,gr_uvd,rotation,batch_size)
    # for i in numpy.random.randint(0,r0.shape[0],10):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(131)
    #     ax.imshow(r0[i,2,:,:],'gray')
    #     ax = fig.add_subplot(132)
    #     ax.imshow(r1[i,2,:,:],'gray')
    #     ax = fig.add_subplot(133)
    #     ax.imshow(r2[i,2,:,:],'gray')
    #     plt.show()
    return rotation,pred_uvd_derot,gr_uvd_derot,pred_patch_center,gr_patch_center,
# if __name__=='__main__':
#     r1=numpy.ones((10,10,3))
#     M = cv2.getRotationMatrix2D((5,5),45,1)
#     cv2.warpAffine(r1,M,(10,10),borderValue=1)
