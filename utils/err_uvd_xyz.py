__author__ = 'QiYE'
import numpy
from xyz_uvd import uvd2xyz,xyz2uvd
import matplotlib.pyplot as plt
import Image
from src.utils.crop_patch_norm_offset import offset_to_abs
import cv2

def show_ori_multijnt(setname,i,uvd_pred,new_uvd_gr):
    if setname == 'msrc':
            data_set = 'test'
            fram_prefix = 'testds'
            dataset_dir = 'C:\\Proj\\msrc_data\\2014-05-19 personalizations global\\%s\\'%data_set
            filename_prefix = "%06d" % (i)
            depth = Image.open('%s%s_%s_depth.png' % (dataset_dir, fram_prefix, filename_prefix))
            depth = numpy.asarray(depth, dtype='uint16')

            plt.imshow(depth, cmap='gray')
            plt.scatter(uvd_pred[i, :,0], uvd_pred[i, :,1], s=20, c='g')
            plt.scatter(new_uvd_gr[i,:,  0], new_uvd_gr[i, :,1], s=20, c='r')
            d_img=[]
            for k in xrange(6):

                d_img.append(depth[uvd_pred[i,k,1],uvd_pred[i,k,0]])
            # print 'r0',d_img
            print 'diff', numpy.abs(uvd_pred[i]-new_uvd_gr[i])
            # print 'pr',numpy.asarray(uvd_pred[i,:,0],dtype='int32'),numpy.asarray(uvd_pred[i,:,1],dtype='int32'),numpy.asarray(uvd_pred[i,:,2],dtype='int32')
            # print 'gr',numpy.asarray(new_uvd_gr[i,:,0],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,1],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,2],dtype='int32')
            plt.show()

    if setname == 'nyu':
            data_set = 'test'
            fram_prefix = 'testds'
            dataset_dir = 'C:\\Proj\\msrc_data\\2014-05-19 personalizations global\\%s\\'%data_set
            filename_prefix = "%06d" % (i)
            depth = Image.open('%s%s_%s.png' % (dataset_dir, fram_prefix, filename_prefix))
            depth = numpy.asarray(depth, dtype='uint16')

            plt.imshow(depth, cmap='gray')
            plt.scatter(uvd_pred[i, :,0], uvd_pred[i, :,1], s=20, c='g')
            plt.scatter(new_uvd_gr[i,:,  0], new_uvd_gr[i, :,1], s=20, c='r')
            d_img=[]
            for k in xrange(6):

                d_img.append(depth[uvd_pred[i,k,1],uvd_pred[i,k,0]])
            # print 'r0',d_img
            print 'diff', numpy.abs(uvd_pred[i]-new_uvd_gr[i])
            # print 'pr',numpy.asarray(uvd_pred[i,:,0],dtype='int32'),numpy.asarray(uvd_pred[i,:,1],dtype='int32'),numpy.asarray(uvd_pred[i,:,2],dtype='int32')
            # print 'gr',numpy.asarray(new_uvd_gr[i,:,0],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,1],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,2],dtype='int32')
            plt.show()
    if setname == 'icvl':
            data_set = 'test'
            fram_prefix = 'depth_1_'
            dataset_dir = 'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\ICVL_dataset_v2_msrc_format\\%s\\depth\\'%data_set
            filename_prefix = "%07d" % (i+1)
            depth = Image.open('%s%s%s.png' % (dataset_dir, fram_prefix, filename_prefix))
            depth = numpy.asarray(depth, dtype='uint16')

            plt.imshow(depth, cmap='gray')
            plt.scatter(uvd_pred[i, :,0], uvd_pred[i, :,1], s=20, c='g')
            plt.scatter(new_uvd_gr[i,:,  0], new_uvd_gr[i, :,1], s=20, c='r')
            d_img=[]
            for k in xrange(6):

                d_img.append(depth[uvd_pred[i,k,1],uvd_pred[i,k,0]])
            # print 'r0',d_img
            print 'diff', numpy.abs(uvd_pred[i]-new_uvd_gr[i])
            # print 'pr',numpy.asarray(uvd_pred[i,:,0],dtype='int32'),numpy.asarray(uvd_pred[i,:,1],dtype='int32'),numpy.asarray(uvd_pred[i,:,2],dtype='int32')
            # print 'gr',numpy.asarray(new_uvd_gr[i,:,0],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,1],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,2],dtype='int32')
            plt.show()
    return
def show_ori_jnt(setname,i,uvd_pred,new_uvd_gr):
    if setname == 'msrc':
            data_set = 'test'
            fram_prefix = 'testds'
            dataset_dir = 'C:\\Proj\\msrc_data\\2014-05-19 personalizations global\\%s\\'%data_set
            filename_prefix = "%06d" % (i)
            depth = Image.open('%s%s_%s_depth.png' % (dataset_dir, fram_prefix, filename_prefix))
            depth = numpy.asarray(depth, dtype='uint16')

            plt.imshow(depth, cmap='gray')
            plt.scatter(uvd_pred[i,0], uvd_pred[i, 1], s=20, c='g')
            plt.scatter(new_uvd_gr[i,0], new_uvd_gr[i, 1], s=20, c='r')
            print 'diff', numpy.abs(uvd_pred[i]-new_uvd_gr[i])
            # print 'pr',numpy.asarray(uvd_pred[i,:,0],dtype='int32'),numpy.asarray(uvd_pred[i,:,1],dtype='int32'),numpy.asarray(uvd_pred[i,:,2],dtype='int32')
            # print 'gr',numpy.asarray(new_uvd_gr[i,:,0],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,1],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,2],dtype='int32')
            plt.show()

    if setname == 'nyu':
            data_set = 'test'
            fram_prefix = 'testds'
            dataset_dir = 'C:\\Proj\\msrc_data\\2014-05-19 personalizations global\\%s\\'%data_set
            filename_prefix = "%06d" % (i)
            depth = Image.open('%s%s_%s.png' % (dataset_dir, fram_prefix, filename_prefix))
            depth = numpy.asarray(depth, dtype='uint16')

            plt.imshow(depth, cmap='gray')
            plt.scatter(uvd_pred[i, :,0], uvd_pred[i, :,1], s=20, c='g')
            plt.scatter(new_uvd_gr[i,:,  0], new_uvd_gr[i, :,1], s=20, c='r')
            # print 'r0',d_img
            print 'diff', numpy.abs(uvd_pred[i]-new_uvd_gr[i])
            # print 'pr',numpy.asarray(uvd_pred[i,:,0],dtype='int32'),numpy.asarray(uvd_pred[i,:,1],dtype='int32'),numpy.asarray(uvd_pred[i,:,2],dtype='int32')
            # print 'gr',numpy.asarray(new_uvd_gr[i,:,0],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,1],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,2],dtype='int32')
            plt.show()
    if setname == 'icvl':
            data_set = 'test'
            fram_prefix = 'depth_1_'
            dataset_dir = 'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\ICVL_dataset_v2_msrc_format\\%s\\depth\\'%data_set
            filename_prefix = "%07d" % (i+1)
            depth = Image.open('%s%s%s.png' % (dataset_dir, fram_prefix, filename_prefix))
            depth = numpy.asarray(depth, dtype='uint16')

            plt.imshow(depth, cmap='gray')
            plt.scatter(uvd_pred[i, 0], uvd_pred[i, 1], s=20, c='g')
            plt.scatter(new_uvd_gr[i,0], new_uvd_gr[i, 1], s=20, c='r')
            # print 'r0',d_img
            print 'diff', numpy.abs(uvd_pred[i]-new_uvd_gr[i])
            # print 'pr',numpy.asarray(uvd_pred[i,:,0],dtype='int32'),numpy.asarray(uvd_pred[i,:,1],dtype='int32'),numpy.asarray(uvd_pred[i,:,2],dtype='int32')
            # print 'gr',numpy.asarray(new_uvd_gr[i,:,0],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,1],dtype='int32'),numpy.asarray(new_uvd_gr[i,:,2],dtype='int32')
            plt.show()
    return
def get_uvd_in_crop_img(uvd,ori_d1d2w,dmin_dmax,orig_pad_border):

    new_uvd = numpy.empty_like(uvd)

    for i in xrange(uvd.shape[0]):
        new_uvd[i,0]= (uvd[i,0]+orig_pad_border-ori_d1d2w[i,1])/ori_d1d2w[i,2]
        new_uvd[i,1]= (uvd[i,1]+orig_pad_border-ori_d1d2w[i,0])/ori_d1d2w[i,2]
        new_uvd[i,2]=(uvd[i,2]-dmin_dmax[i,0])/(dmin_dmax[i,1]-dmin_dmax[i,0])
    return new_uvd


def err_in_ori_xyz(setname,ft,uvd,xyz,roixy,ori_d1d2w,dmin_dmax,orig_pad_border,jnt_type = None,jnt_idx=range(0,21,1)):

    if (len(jnt_idx) != 1) & (jnt_type !='center'):

        uvd_pred = numpy.empty_like(ft)
        new_uvd_gr = numpy.empty_like(ft)
        uvd_gr = uvd[:,jnt_idx,:]
        xyz_gr = xyz[:,jnt_idx,:]

        for i in xrange(xyz.shape[0]):
            uvd_pred[i,:,0] =  ft[i,:,0]*ori_d1d2w[i,2]+ori_d1d2w[i,1]-orig_pad_border
            uvd_pred[i,:,1] =  ft[i,:,1]*ori_d1d2w[i,2]+ori_d1d2w[i,0]-orig_pad_border
            uvd_pred[i,:,2] = ft[i,:,2]*(dmin_dmax[i,1]-dmin_dmax[i,0])+dmin_dmax[i,0]
            new_uvd_gr[i,:,0] =  uvd_gr[i,:,0]*ori_d1d2w[i,2]+ori_d1d2w[i,1]-orig_pad_border
            new_uvd_gr[i,:,1] =  uvd_gr[i,:,1]*ori_d1d2w[i,2]+ori_d1d2w[i,0]-orig_pad_border
            new_uvd_gr[i,:,2] = uvd_gr[i,:,2]*(dmin_dmax[i,1]-dmin_dmax[i,0])+dmin_dmax[i,0]

        err_uvd = numpy.mean(numpy.sqrt(numpy.sum((uvd_pred -new_uvd_gr)**2,axis=-1)),axis=0)
        print 'uvd in ori uvd',err_uvd
        print 'uvd in ori uvd mean',err_uvd.mean()

        xyz_pred = uvd2xyz(setname,uvd_pred,roixy,jnt_type=None)
        err = numpy.mean(numpy.sqrt(numpy.sum((xyz_pred -xyz_gr)**2,axis=-1)),axis=0)
        print 'xyz err', err
        print 'average all jnts', numpy.mean(err)

        return xyz_pred,err

    if (len(jnt_idx)==1) & (jnt_type==None):
        uvd_gr = numpy.squeeze(uvd[:,jnt_idx,:])
        xyz_gr = numpy.squeeze(xyz[:,jnt_idx,:])
    if jnt_type == 'center':
        uvd_gr = numpy.mean(uvd[:,jnt_idx,:],axis=1)
        xyz_gr =numpy.mean(xyz[:,jnt_idx,:],axis=1)

    uvd_pred = numpy.empty_like(ft)
    new_uvd_gr = numpy.empty_like(ft)

    for i in xrange(xyz_gr.shape[0]):

        uvd_pred[i,0] =  ft[i,0]*ori_d1d2w[i,2]+ori_d1d2w[i,1]-orig_pad_border
        uvd_pred[i,1] =  ft[i,1]*ori_d1d2w[i,2]+ori_d1d2w[i,0]-orig_pad_border
        uvd_pred[i,2] = ft[i,2]*(dmin_dmax[i,1]-dmin_dmax[i,0])+dmin_dmax[i,0]

        new_uvd_gr[i,0] =  uvd_gr[i,0]*ori_d1d2w[i,2]+ori_d1d2w[i,1]-orig_pad_border
        new_uvd_gr[i,1] =  uvd_gr[i,1]*ori_d1d2w[i,2]+ori_d1d2w[i,0]-orig_pad_border
        new_uvd_gr[i,2] = uvd_gr[i,2]*(dmin_dmax[i,1]-dmin_dmax[i,0])+dmin_dmax[i,0]

    err_uvd = numpy.mean(numpy.sqrt(numpy.sum((uvd_pred -new_uvd_gr)**2,axis=-1)),axis=0)
    print 'uvd in ori uvd',err_uvd

    xyz_pred = uvd2xyz(setname,uvd_pred,roixy,jnt_type='single')
    err = numpy.mean(numpy.sqrt(numpy.sum((xyz_pred -xyz_gr)**2,axis=-1)),axis=0)
    print 'xyz err', err

    return xyz_pred,err



def uvd_to_xyz_error_single(setname,uvd_pred_offset,rot,prev_jnt_uvd_derot,patch_size,jnt_idx ,offset_depth_range,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border):


    predict_uvd = offset_to_abs(uvd_pred_offset, prev_jnt_uvd_derot,patch_size=patch_size,offset_depth_range=offset_depth_range)

    """"rot the the norm view to original rotatioin view"""
    for i in xrange(uvd_gr.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        # plt.figure()
        # plt.imshow(r0[i],'gray')
        # plt.scatter(predict_uvd[i,0]*96,predict_uvd[i,1]*96)
        predict_uvd[i,0:2] = (numpy.dot(M,numpy.array([predict_uvd[i,0]*96,predict_uvd[i,1]*96,1]))-12)/72

        # plt.figure()
        # dst = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        # plt.imshow(dst,'gray')
        # plt.scatter(predict_uvd[i,0]*72+12,predict_uvd[i,1]*72+12)
        # plt.scatter(uvd_gr[i,jnt_idx,0]*72+12,uvd_gr[i,jnt_idx,1]*72+12,c='r')
        # plt.show()

    xyz_pred,err= err_in_ori_xyz(setname,predict_uvd,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)
    return xyz_pred,err

def uvd_to_xyz_error(setname,uvd_pred_offset,rot,prev_jnt_uvd_derot,patch_size,jnt_idx ,offset_depth_range,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border):


    uvd_pred_offset.shape=(uvd_pred_offset.shape[0],6,3)
    predict_uvd = offset_to_abs(uvd_pred_offset, prev_jnt_uvd_derot,patch_size=patch_size,offset_depth_range=offset_depth_range)

    """"rot the the norm view to original rotatioin view"""
    for i in xrange(uvd_gr.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),rot[i],1)
        for j in xrange(len(jnt_idx)):
            predict_uvd[i,j,0:2] = (numpy.dot(M,numpy.array([predict_uvd[i,j,0]*96,predict_uvd[i,j,1]*96,1]))-12)/72
    xyz_pred,err = err_in_ori_xyz(setname,predict_uvd,uvd_gr,xyz_true,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border,jnt_type=None,jnt_idx=jnt_idx)

    return xyz_pred,err



def xyz_to_uvd_derot(xyz,setname,rot,jnt_idx,roixy,rect_d1d2w,depth_dmin_dmax,orig_pad_border):
    uvd = xyz2uvd(xyz=xyz,setname=setname,roixy=roixy, jnt_type='single')
    uvd_norm = get_uvd_in_crop_img(uvd,ori_d1d2w=rect_d1d2w,dmin_dmax=depth_dmin_dmax,orig_pad_border=orig_pad_border)
    for i in xrange(xyz.shape[0]):
        M = cv2.getRotationMatrix2D((48,48),-rot[i],1)

        uvd_norm[i,0:2] = (numpy.dot(M,numpy.array([uvd_norm[i,0]*72+12,uvd_norm[i,1]*72+12,1])))/96
    return uvd_norm

