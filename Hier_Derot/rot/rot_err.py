from src.utils import constants

__author__ = 'QiYE'
import numpy

from math import pi
import h5py


def rot_err(dataset,setname,source_name,pred_rot_name):
    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    if setname == 'msrc':
        vect = joint_label_uvd[:,9,0:2] - joint_label_uvd[:,0,0:2]
        rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
        loc_neg = numpy.where(vect[:,0]<0)
        rot[loc_neg] = -rot[loc_neg]
        rot = numpy.cast['float32'](rot/pi*180)
        rot[numpy.where(rot==180)] =179
    else:
        """ icvl"""
        vect = joint_label_uvd[:,9,0:2] - joint_label_uvd[:,0,0:2]
        # vect = joint_label_uvd[:,10,0:2] - joint_label_uvd[:,1,0:2]
        rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
        loc_neg = numpy.where(vect[:,0]<0)
        rot[loc_neg] = -rot[loc_neg]
        rot = numpy.cast['float32'](rot/pi*180)
        print numpy.where(rot==180)[0].shape[0]
        rot[numpy.where(rot==180)] =179


    save_path = '../../data/%s/rotation/'%setname
    prediction = numpy.load("%s%s%s.npy"%(save_path,dataset,pred_rot_name))

    smooth_pred = numpy.empty((prediction.shape[0],))
    for i in xrange(prediction.shape[0]):
        max_val = numpy.max(prediction[i])
        loc = numpy.where(prediction[i]==max_val)
        # smooth_pred[i] = (prediction[i][loc]).mean()/constants.Num_Class
        smooth_pred[i] = loc[0].mean()/ constants.Num_Class*360-180

        # fig=plt.figure()
        # ax=fig.add_subplot(121)
        # ax.imshow(r0[i],'gray')
        #
        # idx =[1,2,6,10,14,18]
        # center_uvd=numpy.mean(joint_label_uvd[:,idx,:],axis=1)
        # plt.scatter(joint_label_uvd[i,1,0]*72+12,joint_label_uvd[i,1,1]*72+12)
        # plt.scatter(joint_label_uvd[i,10,0]*72+12,joint_label_uvd[i,10,1]*72+12)
        # print i," ", 'pred ', smooth_pred[i],'label ',rot[i]
        # M = cv2.getRotationMatrix2D((center_uvd[i,0]*72+12,center_uvd[i,1]*72+12),-smooth_pred[i] ,1)
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # ax=fig.add_subplot(122)
        # ax.imshow(dst,'gray')
        # new_u,new_v = numpy.dot(M,numpy.array([joint_label_uvd[i,1,0]*72+12,joint_label_uvd[i,1,1]*72+12,1]))
        # plt.scatter(new_u,new_v)
        # new_u,new_v = numpy.dot(M,numpy.array([joint_label_uvd[i,10,0]*72+12,joint_label_uvd[i,10,1]*72+12,1]))
        # plt.scatter(new_u,new_v)
        # plt.show()

    print smooth_pred[0:30]
    print rot[0:30]
    print numpy.abs(smooth_pred-rot).mean()
def rot_err_nyu():
    dataset='test'
    src_path = '../../data/nyu/source/'
    path = '%s%s_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300.h5'%(src_path,dataset)
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    """nyu"""
    vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]
    # vect = joint_label_uvd[:,10,0:2] - joint_label_uvd[:,1,0:2]
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/pi*180)
    print numpy.where(rot==180)[0].shape[0]
    rot[numpy.where(rot==180)] =179


    save_path = '../../data/nyu/rot/best/'
    prediction = numpy.load("%s%s_rot_r1r2_bin60_c004_c018_c104_c118_h12_h24_gm0_lm0_yt0_ep55.npy"%(save_path,dataset))

    smooth_pred = numpy.empty((prediction.shape[0],))
    for i in xrange(prediction.shape[0]):
        max_val = numpy.max(prediction[i])
        loc = numpy.where(prediction[i]==max_val)
        # smooth_pred[i] = (prediction[i][loc]).mean()/constants.Num_Class
        smooth_pred[i] = loc[0].mean()/ constants.Num_Class*360-180

        # fig=plt.figure()
        # ax=fig.add_subplot(121)
        # ax.imshow(r0[i],'gray')
        #
        # idx =[1,2,6,10,14,18]
        # center_uvd=numpy.mean(joint_label_uvd[:,idx,:],axis=1)
        # plt.scatter(joint_label_uvd[i,1,0]*72+12,joint_label_uvd[i,1,1]*72+12)
        # plt.scatter(joint_label_uvd[i,10,0]*72+12,joint_label_uvd[i,10,1]*72+12)
        # print i," ", 'pred ', smooth_pred[i],'label ',rot[i]
        # M = cv2.getRotationMatrix2D((center_uvd[i,0]*72+12,center_uvd[i,1]*72+12),-smooth_pred[i] ,1)
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # ax=fig.add_subplot(122)
        # ax.imshow(dst,'gray')
        # new_u,new_v = numpy.dot(M,numpy.array([joint_label_uvd[i,1,0]*72+12,joint_label_uvd[i,1,1]*72+12,1]))
        # plt.scatter(new_u,new_v)
        # new_u,new_v = numpy.dot(M,numpy.array([joint_label_uvd[i,10,0]*72+12,joint_label_uvd[i,10,1]*72+12,1]))
        # plt.scatter(new_u,new_v)
        # plt.show()

    print smooth_pred[0:30]
    print rot[0:30]
    print numpy.where(numpy.abs(smooth_pred-rot)>50)[0].shape
    print numpy.abs(smooth_pred-rot).mean()


def rot_err_icvl(bin,dataset,dataset_path_prefix,setname,source_name,pred_rot_path):
    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    f = h5py.File(path,'r')
    r0 = f['r0'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    """ icvl"""
    vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = rot/pi*180
    rot = numpy.asarray(numpy.floor(rot),dtype='int16')
    rot[numpy.where(rot==180)] =179


    save_path =  '%sdata/%s/hier_derot/rot/best/'%(dataset_path_prefix,setname)
    prediction = numpy.load("%s%s%s.npy"%(save_path,dataset,pred_rot_path))
    print prediction.shape
    smooth_pred = numpy.empty((prediction.shape[0],))
    for i in xrange(prediction.shape[0]):
        max_val = numpy.max(prediction[i])
        loc = numpy.where(prediction[i]>=(1*max_val))
        smooth_pred[i] = loc[0].mean()*bin+16-180

        # fig=plt.figure()
        # ax=fig.add_subplot(121)
        # ax.imshow(r0[i],'gray')
        #
        # idx =[0,9]
        # center_uvd=numpy.mean(joint_label_uvd[:,idx,:],axis=1)
        # plt.scatter(joint_label_uvd[i,0,0]*72+12,joint_label_uvd[i,0,1]*72+12)
        # plt.scatter(joint_label_uvd[i,9,0]*72+12,joint_label_uvd[i,9,1]*72+12)
        # print i," ", 'pred ', smooth_pred[i],'label ',rot[i]
        # M = cv2.getRotationMatrix2D((center_uvd[i,0]*72+12,center_uvd[i,1]*72+12),-smooth_pred[i] ,1)
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # ax=fig.add_subplot(122)
        # ax.imshow(dst,'gray')
        # new_u,new_v = numpy.dot(M,numpy.array([joint_label_uvd[i,0,0]*72+12,joint_label_uvd[i,0,1]*72+12,1]))
        # plt.scatter(new_u,new_v)
        # new_u,new_v = numpy.dot(M,numpy.array([joint_label_uvd[i,9,0]*72+12,joint_label_uvd[i,9,1]*72+12,1]))
        # plt.scatter(new_u,new_v)
        # plt.show()


    print smooth_pred.shape
    print rot.shape
    print numpy.where(numpy.abs(smooth_pred-rot)>50)[0].shape
    print numpy.abs(smooth_pred-rot).mean()
def rot_err_regress(setname,source_name,dataset,rot_save_name):
    src_path = '../../data/%s/source/%s_%s%s.h5'%(setname,dataset,setname,source_name)
    f = h5py.File(src_path,'r')

    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()


    vect = joint_label_uvd[:,0,0:2] - joint_label_uvd[:,9,0:2]#the index is valid for 21joints
    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/2/pi+0.5)
    rot[numpy.where(rot==1)]=0

    rot_pred =numpy.squeeze(numpy.load('../../data/%s/rot/best/%s%s.npy'%(setname,dataset,rot_save_name)))

    print rot_pred.shape
    print rot.shape
    print rot[0:10]
    print rot_pred[0:10]
    err = numpy.mean(numpy.abs((rot-rot_pred)*360))
    print err



if __name__ == '__main__':
    rot_err_icvl(bin=6,dataset='train',
                 dataset_path_prefix=constants.Data_Path,
                 setname='icvl',
                 source_name='_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                 pred_rot_path='_rot_r1r2_bin46_c004_c018_c104_c118_h12_h24_gm0_lm10_yt0_ep135')
    # rot_err_nyu()
    # rot_err_regress(dataset='train',
    #                 setname = 'icvl',
    #                 source_name='_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             rot_save_name='_rot_r1r2_c004_c018_c104_c118_h12_h24_gm0_lm0_yt0_ep70')

    # rot_err_regress(dataset='train',
    #                 setname = 'msrc',
    #                 source_name='_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #             rot_save_name='_rot_r1r2_bin60_c14_c28_h14_h28_gm0_lm1_yt0_ep65')
    # rot_err(dataset='test',
    #                 setname = 'msrc',
    #                 source_name='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #             pred_rot_name='_rot_r1r2_bin60_c14_c28_h14_h28_gm0_lm1_yt0_ep65')