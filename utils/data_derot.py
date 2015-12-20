from src.utils import constants

__author__ = 'QiYE'
import numpy
import h5py
import matplotlib.pyplot as plt
import cv2


def derot_test(dataset,bin_size,offset_rot,set_name,source_name,rot_save_name):
    path = "../../data/%s/rot/best/"%set_name
    rot_pred = numpy.load("%s%s%s.npy"%(path,dataset,rot_save_name))

    src_path = '../../data/%s/source/'%set_name
    path = '%s%s_%s%s.h5'%(src_path,dataset,set_name,source_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    new_uvd = numpy.empty((21,2))
    rot_smooth_pred = numpy.empty((r0.shape[0],))

    num=30
    idx = numpy.random.randint(0,r0.shape[0],size=num)
    for k in xrange(num):
        i=idx[k]
        max_val = numpy.max(rot_pred[i])
        loc = numpy.where(rot_pred[i]==max_val)
        # smooth_pred[i] = (prediction[i][loc]).mean()/constants.Num_Class
        rot_smooth_pred[i] = loc[0].mean()*bin_size+offset_rot-180

        fig=plt.figure()
        ax=fig.add_subplot(231)
        ax.imshow(r0[i],'gray')
        plt.scatter(joint_label_uvd[i,0,0]*72+12,joint_label_uvd[i,0,1]*72+12)
        plt.scatter(joint_label_uvd[i,9,0]*72+12,joint_label_uvd[i,9,1]*72+12)
        ax=fig.add_subplot(232)
        ax.imshow(r1[i],'gray')
        ax=fig.add_subplot(233)
        ax.imshow(r2[i],'gray')

        M = cv2.getRotationMatrix2D((48,48),-rot_smooth_pred[i],1)
        new_r0 = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in xrange(joint_label_uvd.shape[1]):
            new_uvd[j] = numpy.dot(M,numpy.array([joint_label_uvd[i,j,0]*72+12,joint_label_uvd[i,j,1]*72+12,1]))/96

        M = cv2.getRotationMatrix2D((24,24),-rot_smooth_pred[i],1)
        new_r1 = cv2.warpAffine(r1[i],M,(48,48),borderValue=1)

        M = cv2.getRotationMatrix2D((12,12),-rot_smooth_pred[i],1)
        new_r2 = cv2.warpAffine(r2[i],M,(24,24),borderValue=1)
        ax=fig.add_subplot(234)
        ax.imshow(new_r0,'gray')
        plt.scatter(new_uvd[0,0]*96,new_uvd[0,1]*96)
        plt.scatter(new_uvd[9,0]*96,new_uvd[9,1]*96)
        ax=fig.add_subplot(235)
        ax.imshow(new_r1,'gray')
        ax=fig.add_subplot(236)
        ax.imshow(new_r2,'gray')
        plt.show()

def derot_dataset(dataset,bin_size,dataset_path_prefix,offset_rot,setname,source_name,rot_save_name):
    # path = "../../data/%s/rotation/"%set_name
    path = '%sdata/%s/hier_derot/rot/best/'%(dataset_path_prefix,setname)
    rot_pred = numpy.load("%s%s%s.npy"%(path,dataset,rot_save_name))
    print rot_pred.shape
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f_derot = h5py.File('%s%s_%s_derot2%s.h5'%(src_path,dataset,setname,source_name),'w')
    print r0.shape
    for key in f.keys():
        f.copy(key,f_derot)
    f.close()
    ori_uvd = joint_label_uvd.copy()
    vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]#the index is valid for 21joints
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/2/3.1415926+0.5)
    print 'norm to [0,1]'
    bin = 6
    rot = numpy.asarray(numpy.floor(rot*360),dtype='int32')
    rot[numpy.where(rot==360)] =359
    print numpy.min(rot)
    print numpy.max(rot)
    rot_bin = (rot-16)/bin
    hist = numpy.zeros((constants.Num_Class,),dtype='int32')
    for i in xrange(rot_bin.shape[0]):
        hist[rot_bin[i]] +=1
    print hist
    #

    rot_smooth_pred = numpy.empty((r0.shape[0],))

    for i in xrange(0,joint_label_uvd.shape[0],1):
        max_val = numpy.max(rot_pred[i])
        loc = numpy.where(rot_pred[i]==max_val)
        # smooth_pred[i] = (prediction[i][loc]).mean()/constants.Num_Class
        rot_smooth_pred[i] = loc[0].mean()*bin_size+offset_rot-180


        M = cv2.getRotationMatrix2D((48,48),-rot_smooth_pred[i],1)
        r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in xrange(joint_label_uvd.shape[1]):
            joint_label_uvd[i,j,0:2] = numpy.dot(M,numpy.array([joint_label_uvd[i,j,0]*72+12,joint_label_uvd[i,j,1]*72+12,1]))/96

        M = cv2.getRotationMatrix2D((24,24),-rot_smooth_pred[i],1)
        r1[i] = cv2.warpAffine(r1[i],M,(48,48),borderValue=1)

        M = cv2.getRotationMatrix2D((12,12),-rot_smooth_pred[i],1)
        r2[i] = cv2.warpAffine(r2[i],M,(24,24),borderValue=1)

    vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]#the index is valid for 21joints
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/2/3.1415926+0.5)
    print 'norm to [0,1]'
    bin = 6
    rot = numpy.asarray(numpy.floor(rot*360),dtype='int32')
    rot[numpy.where(rot==360)] =359
    print numpy.min(rot)
    print numpy.max(rot)
    rot_bin = (rot-16)/bin
    hist = numpy.zeros((constants.Num_Class,),dtype='int32')
    rot_bin[numpy.where(rot_bin>45)]=45
    for i in xrange(rot_bin.shape[0]):
        hist[rot_bin[i]] +=1
    print hist


    f_derot.create_dataset('rotation', data=rot_smooth_pred)
    f_derot['r0'][...]=r0
    f_derot['r1'][...]=r1
    f_derot['r2'][...]=r2
    f_derot['joint_label_uvd'][...]=joint_label_uvd
    f_derot.close()

    tmp_uvd=numpy.empty_like(joint_label_uvd)
    tmp_uvd[:,:,2]=joint_label_uvd[:,:,2]
    for i in xrange(joint_label_uvd.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot_smooth_pred[i],1)
        # plt.figure()
        # plt.imshow(r0[i],'gray')
        # plt.scatter(whole_pred[i,:,0]*96,whole_pred[i,:,1]*96)
        # plt.scatter(derot_uvd[i,:,0]*96,derot_uvd[i,:,1]*96,c='r')
        for j in xrange(21):

            tmp_uvd[i,j,0:2] = (numpy.dot(M,numpy.array([joint_label_uvd[i,j,0]*96,joint_label_uvd[i,j,1]*96,1]))-12)/72

        # plt.figure()
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # plt.imshow(dst,'gray')
        #
        # plt.scatter(ori_uvd[i,:,0]*72+12,ori_uvd[i,:,1]*72+12,s=30,c='r')
        # plt.scatter(tmp_uvd[i,:,0]*72+12,tmp_uvd[i,:,1]*72+12,s=10,c='g')
        # plt.show()
    print 'dmean', numpy.abs(tmp_uvd[:,2]-joint_label_uvd[:,2]).mean()
    err =numpy.sqrt(numpy.sum((ori_uvd -tmp_uvd)**2,axis=-1))
    print numpy.where(err>0.01)[0]
    err_uvd = numpy.mean(numpy.sqrt(numpy.sum((ori_uvd -tmp_uvd)**2,axis=-1)),axis=0)
    print 'gr error', err_uvd.mean()


    print '%s%s_%s_derot%s.h5'%(src_path,dataset,setname,source_name), 'closed'


def read_derot_dataset(dataset,set_name,file_name):

    src_path = '../../data/%s/source/'%set_name
    path = '%s%s_%s%s.h5'%(src_path,dataset,set_name,file_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]*96
    f.close()
    vect = joint_label_uvd[:,9,0:2] - joint_label_uvd[:,0,0:2]#the index is valid for 21joints
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/2/3.1415926+0.5)
    print 'norm to [0,1]'
    bin = 6
    rot = numpy.asarray(numpy.floor(rot*360),dtype='int32')
    rot[numpy.where(rot==360)] =359
    print numpy.min(rot)
    print numpy.max(rot)
    rot_bin = (rot-16)/bin
    hist = numpy.zeros((constants.Num_Class,),dtype='int32')
    for i in xrange(rot_bin.shape[0]):
        hist[rot_bin[i]] +=1
    print hist

    idx = numpy.random.randint(0,r0.shape[0],size=10)
    for k in xrange(10):
        i = idx[k]
        fig=plt.figure()
        ax=fig.add_subplot(131)
        ax.imshow(r0[i],'gray')
        plt.scatter(joint_label_uvd[i,0,0],joint_label_uvd[i,0,1])
        plt.scatter(joint_label_uvd[i,9,0],joint_label_uvd[i,9,1])
        ax=fig.add_subplot(132)
        ax.imshow(r1[i],'gray')
        ax=fig.add_subplot(133)
        ax.imshow(r2[i],'gray')
        plt.show()

if __name__ == "__main__":
    derot_dataset(dataset='train',bin_size=6,offset_rot=16,setname='icvl',
                  dataset_path_prefix=constants.Data_Path,
                  source_name='_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                  rot_save_name='_rot_r1r2_bin46_c004_c018_c104_c118_h12_h24_gm0_lm10_yt0_ep135')
    # derot_test(dataset='train',bin_size=6,offset_rot=16,set_name='icvl',
    #               source_name='_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #               rot_save_name='_rot_r1r2_bin46_c004_c018_c104_c118_h12_h24_gm0_lm10_yt0_ep100')
    # read_derot_dataset(dataset='test',
    #                    set_name = 'icvl',
    #                    file_name = '_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200')

    # derot_dataset(dataset='train',bin_size=6,offset_rot=0,set_name='nyu',
    #               source_name='_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #               rot_save_name='_rot_r1r2_bin60_c004_c018_c104_c118_h12_h24_gm0_lm1_yt0_ep30')
    # derot_dataset(dataset='train',bin_size=6,offset_rot=0,set_name='msrc',
    #               source_name='_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #               rot_save_name='_rot_r1r2_bin60_c14_c28_h14_h28_gm0_lm1_yt0_ep65')
    # read_derot_dataset(dataset='train',
    #                    set_name = 'msrc',
    #                    file_name = '_r0_r1_r2_uvd_bbox_21jnts_derot_20151030_depth300')

