__author__ = 'QiYE'

import numpy
import Image
import h5py
import matplotlib.pyplot as plt
import scipy.io

def convert_coordinates_normalize9696_origin512424(norm_jnt_uvd,img_size,norm_center_hand_width,bbox, depth_dmin_dmax, border):
    orig_jnt_uvd = numpy.empty_like(norm_jnt_uvd)
    pad_width = (img_size - norm_center_hand_width)/2
    for i in xrange(norm_jnt_uvd.shape[0]):
        # print bbox[i,2]
        orig_jnt_uvd[i,:,0:2] = (norm_jnt_uvd[i,:,0:2]*img_size-pad_width)/norm_center_hand_width*bbox[i,2]-border
        orig_jnt_uvd[i,:,0] += bbox[i,1]
        orig_jnt_uvd[i,:,1]  += bbox[i,0]
        dmin=depth_dmin_dmax[i,0]
        dmax = depth_dmin_dmax[i,1]
        orig_jnt_uvd[i,:,2] = norm_jnt_uvd[i,:,2] *(dmax-dmin)+dmin
    return orig_jnt_uvd

def show_origin_crop():
    # data_set = 'test'
    # fram_prefix = 'testds'
    data_set = 'train'
    fram_prefix = 'random'
    dataset_dir = 'C:\\Proj\\msrc_data\\2014-05-19 personalizations global\\%s\\'%data_set
    save_path = '../../data/'
    img_size=96
    hand_width= 72
    margin_around_hand=25
    depth_range=120

    border=25

    save_path_name = '%smsrc_%s_100000_r0_r1_r2_uvd_bbox.h5'%(save_path,data_set)
    f = h5py.File(save_path_name,'r')

    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    bbox = f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    f.close()

    orig_jnt_uvd = convert_coordinates_normalize9696_origin512424(joint_label_uvd,img_size,hand_width,bbox=bbox,depth_dmin_dmax=depth_dmin_dmax,border=border)
    print orig_jnt_uvd.shape
    keypoints = scipy.io.loadmat('../../data/joint_uvd/uvd_msrc_%s_16joints_twist_fingers.mat' % data_set)
    joint_uvd = keypoints['uvd']
    for i in xrange(16):
        print orig_jnt_uvd[0,i,:]
        print joint_uvd[0,i,:]

    num_img=20
    for i in range(0,num_img,1):
        # print i
        filename_prefix = "%06d" % (i)
        depth = Image.open('%s%s_%s_depth.png' % (dataset_dir, fram_prefix, filename_prefix))
        depth = numpy.asarray(depth, dtype='uint16')


        plt.figure()
        plt.imshow(r0[i],'gray')
        plt.scatter(joint_label_uvd[i,:,0]*96,joint_label_uvd[i,:,1]*96,s=20,c='g')
        plt.show()

        plt.figure()
        plt.imshow(depth,'gray')
        plt.scatter(joint_uvd[i,:,0],joint_uvd[i,:,1],s=20,c='r')
        plt.show()



if __name__=='__main__':
    show_origin_crop()