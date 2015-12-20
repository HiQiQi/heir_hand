from src.utils import constants

__author__ = 'QiYE'
import numpy
import h5py
from math import pi
import cv2
from sklearn.utils import shuffle
def random_rot(rot,r1,r2,rot_range,rand_range):
    # rand_range = (-180,rot_range[0]-180),(rot_range[1]-180,180)
    # rot_range = (rot<=180 , rot>rot_range[0])),(rot<rot_range[1] , rot>180))
    loc_1 = numpy.where(numpy.logical_and( rot>rot_range[0],rot<=rot_range[1] ))
    n_smp = loc_1[0].shape[0]

    new_r1 = numpy.empty((n_smp,r1.shape[1],r1.shape[2]),dtype='float32')
    new_r2 = numpy.empty((n_smp,r2.shape[1],r2.shape[2]),dtype='float32')
    new_rot = numpy.empty((n_smp,),dtype='int32')
    for k in xrange(loc_1[0].shape[0]):
        i=loc_1[0][k]
        # print rot[i]
        rand_angle_1 = numpy.random.randint(rand_range[0],rand_range[1])
        rot_tmp = rot[i]+rand_angle_1
        if rot_tmp<0:
            rand_angle_1=rot_tmp+360-rot[i]
            new_rot[k]=rot_tmp+360
        else:
            if rot_tmp>=360:
                rand_angle_1=rot_tmp-360-rot[i]
                new_rot[k]=rot_tmp-360
            else:
                new_rot[k]=rot_tmp

        M = cv2.getRotationMatrix2D((24,24),rand_angle_1,1)
        new_r1[k] = cv2.warpAffine(r1[i],M,(48,48),borderValue=1)

        M = cv2.getRotationMatrix2D((12,12),rand_angle_1,1)
        new_r2[k] = cv2.warpAffine(r2[i],M,(24,24),borderValue=1)

    return new_r1,new_r2,new_rot


def load_data_r1r2_rotzdiscrete(path,setname,model_type,batch_size,is_shuffle):
    print 'is_shuffle', is_shuffle
    """for msrc dataset and nyu dataset"""
    f = h5py.File(path,'r')
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    print 'num of samples', r1.shape[0]
    # vect = joint_label_uvd[:,10,0:2] - joint_label_uvd[:,1,0:2]#the index is valid for 22joints
    if setname=='nyu':
        vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]#the index is valid for 21joints
    if setname=='msrc':
        vect = joint_label_uvd[:,9,0:2] - joint_label_uvd[:,0,0:2]#the index is valid for 21joints
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/2/pi+0.5)


    print 'norm to [0,1]'
    bin_size = 360/ constants.Num_Class
    rot = numpy.asarray(numpy.floor(rot*360),dtype='uint16')
    rot[numpy.where(rot==360)] =359
    rot_bin = rot/bin_size

    hist = numpy.zeros((constants.Num_Class,),dtype='float32')
    for i in xrange(rot_bin.shape[0]):
        hist[rot_bin[i]] +=1
    print hist
    if is_shuffle==True:
         r1, r2,rot_bin = shuffle(r1,r2,rot_bin,random_state=0)
    return  r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot_bin


    # if setname =='nyu' and model_type=='training':
    #     new_r10,new_r20,new_rot0 = random_rot(rot,r1,r2,[150,180],[-170,-30])
    #     new_r11,new_r21,new_rot1 = random_rot(rot,r1,r2,[180,210],[30,170])
    #     r1 = numpy.concatenate([r1,new_r10,new_r11])
    #     r2 = numpy.concatenate([r2,new_r20,new_r21])
    #     rot = numpy.concatenate([rot,new_rot0,new_rot1])
    #
    #     rot_bin = rot/bin_size
    #     hist = numpy.zeros((constants.Num_Class,),dtype='float32')
    #     for i in xrange(rot_bin.shape[0]):
    #         hist[rot_bin[i]] +=1
    #     print hist
    #
    # if r1.shape[0]%batch_size == 0:
    #     return  r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot_bin
    # else:
    #     if model_type == 'training':
    #         new_r1=numpy.empty((r1.shape[0]+batch_size-r1.shape[0]%batch_size,r1.shape[1],r1.shape[2]),dtype='float32')
    #         new_r1[0:r1.shape[0]]=r1
    #         new_r2=numpy.empty((r2.shape[0]+batch_size-r2.shape[0]%batch_size,r2.shape[1],r2.shape[2]),dtype='float32')
    #         new_r2[0:r2.shape[0]]=r2
    #         new_rot_bin=numpy.empty((r2.shape[0]+batch_size-r2.shape[0]%batch_size,),dtype='uint16')
    #         new_rot_bin[0:r1.shape[0]]=rot_bin
    #
    #         rand_idx = numpy.random.randint(low=r1.shape[0],high=new_r1.shape[0],size=batch_size-r1.shape[0]%batch_size)
    #         new_r1[r1.shape[0]:new_r1.shape[0]]=r1[rand_idx-r1.shape[0]]
    #         new_r2[r1.shape[0]:new_r1.shape[0]]=r2[rand_idx-r1.shape[0]]
    #         new_rot_bin[r1.shape[0]:new_r1.shape[0]]=new_rot_bin[rand_idx-r1.shape[0]]
    #
    #         new_r1, new_r2,new_rot_bin = shuffle(new_r1,new_r2,new_rot_bin,random_state=0)
    #
    #         return  new_r1.reshape(new_r1.shape[0], 1, new_r1.shape[1],new_r1.shape[2]),\
    #                 new_r2.reshape(new_r2.shape[0], 1, new_r2.shape[1],new_r2.shape[2]),\
    #                 new_rot_bin
    #     else:
    #
    #         return  r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot_bin

def load_data_r1r2_rot_conti(path,model_type,batch_size):
    """for msrc dataset and nyu dataset"""
    f = h5py.File(path,'r')

    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    print r1.shape

    vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]#the index is valid for 21joints
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/2/pi+0.5)
    rot[numpy.where(rot==1)]=0

    if r1.shape[0]%batch_size == 0:
        return  r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot.reshape(rot.shape[0],1)
    else:
        if model_type == 'training':
            new_r1=numpy.empty((r1.shape[0]+batch_size-r1.shape[0]%batch_size,r1.shape[1],r1.shape[2]),dtype='float32')
            new_r1[0:r1.shape[0]]=r1
            new_r2=numpy.empty((r2.shape[0]+batch_size-r2.shape[0]%batch_size,r2.shape[1],r2.shape[2]),dtype='float32')
            new_r2[0:r2.shape[0]]=r2
            new_rot=numpy.empty((r2.shape[0]+batch_size-r2.shape[0]%batch_size,),dtype='float32')
            new_rot[0:r1.shape[0]]=rot
            rand_idx = numpy.random.randint(low=r1.shape[0],high=new_r1.shape[0],size=batch_size-r1.shape[0]%batch_size)
            new_r1[r1.shape[0]:new_r1.shape[0]]=r1[rand_idx-r1.shape[0]]
            new_r2[r1.shape[0]:new_r1.shape[0]]=r2[rand_idx-r1.shape[0]]
            new_rot[r1.shape[0]:new_r1.shape[0]]=new_rot[rand_idx-r1.shape[0]]
            new_r1, new_r2,new_rot = shuffle(new_r1,new_r2,new_rot,random_state=0)

            return  new_r1.reshape(new_r1.shape[0], 1, new_r1.shape[1],new_r1.shape[2]),\
                    new_r2.reshape(new_r2.shape[0], 1, new_r2.shape[1],new_r2.shape[2]),\
                    new_rot.reshape(new_rot.shape[0],1)
        else:
            return  r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot.reshape(rot.shape[0],1)

def load_data_r1r2_rotzdiscrete_icvl(path,model_type,batch_size,is_shuffle):
    print 'is_shuffle', is_shuffle
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()
    # for i in xrange(100):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(132)
    #     ax.imshow(r1[i],'gray')
    #     ax = fig.add_subplot(133)
    #     ax.imshow(r2[i],'gray')
    #     plt.show()

    # vect = joint_label_uvd[:,10,0:2] - joint_label_uvd[:,1,0:2]#the index is valid for 22joints
    vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]#the index is valid for 21joints

    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/2/pi+0.5)


    print 'norm to [0,1]'
    bin = 6
    rot = numpy.asarray(numpy.floor(rot*360),dtype='uint16')
    rot[numpy.where(rot==360)] =359
    print numpy.min(rot)
    print numpy.max(rot)
    rot_bin = (rot-16)/bin
    hist = numpy.zeros((constants.Num_Class,),dtype='uint16')
    for i in xrange(rot_bin.shape[0]):
        hist[rot_bin[i]] +=1
    print hist

    if is_shuffle == True:
        r1, r2,rot_bin = shuffle(r1,r2,rot_bin,random_state=0)
    return r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot_bin

    # if model_type == 'training':
    #     new_r10,new_r20,new_rot0 = random_rot(rot,r1,r2,[150,180],[-133,-30])
    #     new_r11,new_r21,new_rot1 = random_rot(rot,r1,r2,[180,200],[30,87])
    #     r1 = numpy.concatenate([r1,new_r10,new_r11])
    #     r2 = numpy.concatenate([r2,new_r20,new_r21])
    #     rot = numpy.concatenate([rot,new_rot0,new_rot1])
    #
    #     rot_bin = (rot-16)/bin
    #     hist = numpy.zeros((constants.Num_Class,),dtype='uint16')
    #     for i in xrange(rot_bin.shape[0]):
    #         hist[rot_bin[i]] +=1
    #     print hist
    #
    #
    # if r1.shape[0]%batch_size == 0:
    #     return  r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot_bin
    # else:
    #     if model_type == 'training':
    #         new_r1=numpy.empty((r1.shape[0]+batch_size-r1.shape[0]%batch_size,r1.shape[1],r1.shape[2]),dtype='float32')
    #         new_r1[0:r1.shape[0]]=r1
    #         new_r2=numpy.empty((r2.shape[0]+batch_size-r2.shape[0]%batch_size,r2.shape[1],r2.shape[2]),dtype='float32')
    #         new_r2[0:r2.shape[0]]=r2
    #         new_rot_bin=numpy.empty((r2.shape[0]+batch_size-r2.shape[0]%batch_size,),dtype='uint16')
    #         new_rot_bin[0:r1.shape[0]]=rot_bin
    #
    #         rand_idx = numpy.random.randint(low=r1.shape[0],high=new_r1.shape[0],size=batch_size-r1.shape[0]%batch_size)
    #         new_r1[r1.shape[0]:new_r1.shape[0]]=r1[rand_idx-r1.shape[0]]
    #         new_r2[r1.shape[0]:new_r1.shape[0]]=r2[rand_idx-r1.shape[0]]
    #         new_rot_bin[r1.shape[0]:new_r1.shape[0]]=new_rot_bin[rand_idx-r1.shape[0]]
    #
    #         new_r1, new_r2,new_rot_bin = shuffle(new_r1,new_r2,new_rot_bin,random_state=0)
    #
    #         return  new_r1.reshape(new_r1.shape[0], 1, new_r1.shape[1],new_r1.shape[2]),\
    #                 new_r2.reshape(new_r2.shape[0], 1, new_r2.shape[1],new_r2.shape[2]),\
    #                 new_rot_bin
    #     else:
    #         return  r1.reshape(r1.shape[0], 1, r1.shape[1],r1.shape[2]),r2.reshape(r2.shape[0], 1, r2.shape[1],r2.shape[2]),rot_bin



    # for i in xrange(800,960,20):
    #     rot_smooth_pred =( 1.0*rot[i]*bin+constants.min_rot_icvl)-180
    #
    #
    #     M = cv2.getRotationMatrix2D((48,48),-rot_smooth_pred,1)
    #     r = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
    #     fig = plt.figure()
    #     ax= fig.add_subplot(121)
    #     ax.imshow(r0[i],'gray')
    #     ax= fig.add_subplot(122)
    #     x.imshow(r,'gray')
    #     plt.show()

