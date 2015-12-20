__author__ = 'QiYE'

import theano
import theano.tensor as T
from load_data import  load_data_multi_base_uvd_normalized,get_thresh_bw_jnts,load_norm_offset_uvd
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import set_params
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import uvd_to_xyz_error
from src.utils import read_save_format

import sys
def test_model(setname,dataset, source_name,prev_jnt_uvd_derot,batch_size,jnt_idx,patch_size,offset_depth_range,num_enlarge,h1_out_factor,h2_out_factor,model_path):
    print 'offset_depth_range ',offset_depth_range
    model_info='uvd_bw_r012_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info

    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_uvd_derot,is_shuffle=False,
                                                                                         jnt_idx=jnt_idx,
                                                                                         patch_size=patch_size,patch_pad_width=4,
                                                                                         offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches


    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32

    model = CNN_Model_multi3(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
                      img_size_0 = img_size_0,
                      img_size_1=img_size_1,
                      img_size_2=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 5,
                pool_c00= 4,
                c01= c2,
                kernel_c01= 6,
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 5,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)

    cost =  model.cost(Y)


    save_path =    '../../data/%s/hier_derot/base_wrist/best/'%setname
    model_save_path = "%s%s.npy"%(save_path,model_path)
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output ],on_unused_input='ignore')

    cost_nbatch = 0

    uvd_offset_norm = numpy.empty_like(test_set_y)
    for minibatch_index in xrange(n_test_batches):
        # print minibatch_index
        x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
        x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
        x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
        y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

        cost_ij,uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)

        cost_nbatch+=cost_ij
        uvd_offset_norm[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] = uvd_batch

    print 'test cost', cost_nbatch/n_test_batches
    return uvd_offset_norm



def test_model_msrc(setname,dataset, source_name,prev_jnt_uvd_derot,batch_size,jnt_idx,patch_size,offset_depth_range,num_enlarge,h1_out_factor,h2_out_factor,model_path):
    print 'offset_depth_range ',offset_depth_range
    model_info='uvd_bw_r012_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info

    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_uvd_derot,is_shuffle=False,
                                                                                         jnt_idx=jnt_idx,
                                                                                         patch_size=patch_size,patch_pad_width=4,
                                                                                         offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    path = '%strain%s.h5'%(src_path,source_name)

    prev_jnt_path = '../../data/%s/hier_derot/final_xyz_uvd/train_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770.npy'%setname
    train_set_y= load_norm_offset_uvd(path,prev_jnt_path, jnt_idx=jnt_idx,
                                     patch_size=patch_size,
                                     offset_depth_range=offset_depth_range,hand_width=96)
    thresh_bw_jnts = get_thresh_bw_jnts(train_set_y.reshape(train_set_y.shape[0],6,3),ratio=2)

    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches


    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32

    model = CNN_Model_multi3(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
                      img_size_0 = img_size_0,
                      img_size_1=img_size_1,
                      img_size_2=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 5,
                pool_c00= 4,
                c01= c2,
                kernel_c01= 6,
                pool_c01= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 5,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)

    cost = model.sum_of_cost(Y,ratio=1,thresh=thresh_bw_jnts)
    cost_1 = model.cost(Y)
    cost_2 = model.cost_bw_jnts(Y,thresh=thresh_bw_jnts)

    save_path =    '../../data/%s/hier_derot/base_wrist/best/'%setname
    model_save_path = "%s%s.npy"%(save_path,model_path)
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2,model.layers[-1].output ],on_unused_input='ignore')

    cost_nbatch = 0
    cost_loc_nbatch=0
    cost_rot_nbatch=0
    uvd_offset_norm = numpy.empty_like(test_set_y)
    for minibatch_index in xrange(n_test_batches):
        # print minibatch_index
        x0 = test_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
        x1 = test_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
        x2 = test_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
        y = test_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]

        cost_ij,cost_loc,cost_rot,uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
        cost_loc_nbatch+=cost_loc
        cost_rot_nbatch+=cost_rot
        cost_nbatch+=cost_ij
        uvd_offset_norm[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] = uvd_batch

    print 'test cost', cost_nbatch/n_test_batches,cost_loc_nbatch/n_test_batches,cost_rot_nbatch/n_test_batches
    return uvd_offset_norm

def get_base_wrist_loc_err(setname,xyz_jnt_path):

    ''''change the path: xyz location of the palm center, file format can be npy or mat'''

    dataset='test'
    jnt_idx = [0,1,5,9 ,13,17]
    if setname =='msrc':
        prev_jnt_path = '../../data/%s/hier_derot/final_xyz_uvd/test_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770.npy'%setname
        source_name='_msrc_derot_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
        source_name_ori='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
        model_path='param_cost_base_wrist_r012_uvd_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep1000'
        batch_size = 100
        patch_size=56
        offset_depth_range=1.0
        h1_out_factor=4
        h2_out_factor=16
    else:
        if setname=='nyu':
            source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300'
            source_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300'
            prev_jnt_path = '../../data/%s/hier_derot/final_xyz_uvd/test_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep1000.npy'%setname
            model_path='param_cost_uvd_base_wrist_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep2000'
            h1_out_factor=2
            h2_out_factor=4
            batch_size = 100
            patch_size=56
            offset_depth_range=0.8
        else:
            if setname=='icvl':
                source_name='_msrc_derot_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
                source_name_ori='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
                prev_jnt_path = '../../data/%s/hier_derot/final_xyz_uvd/test_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770.npy'%setname
                model_path='param_cost_base_wrist_r012_uvd_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm10000_yt0_ep1000'
                h1_out_factor=12
                h2_out_factor=24
                batch_size = 133
                patch_size=56
                offset_depth_range=0.8
            else:
                sys.exit('dataset name shoudle be icvl/nyu/msrc')


    file_format='npy'

    '''don't touch the following part!!!!'''
    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    print path
    f = h5py.File(path,'r')
    r0=f['r0'][...]
    rot = f['rotation'][...]
    rect_d1d2w=f['bbox'][...]
    depth_dmin_dmax = f['depth_dmin_dmax'][...]
    orig_pad_border=f['orig_pad_border'][...]
    # derot_uvd = f['joint_label_uvd'][...]
    f.close()

    keypoints = scipy.io.loadmat('../../data/%s/source/%s_%s_xyz_21joints.mat' % (setname,dataset,setname))
    xyz_true = keypoints['xyz']
    keypoints = scipy.io.loadmat('../../data/%s/source/%s_%s_roixy_21joints.mat' % (setname,dataset,setname))
    roixy = keypoints['roixy']


    path = '%s%s%s.h5'%(src_path,dataset,source_name_ori)
    f = h5py.File(path,'r')
    uvd_gr = f['joint_label_uvd'][...]
    f.close()

    prev_jnt_uvd_derot=read_save_format.load(prev_jnt_path,format=file_format)
    if setname =='msrc':
        uvd_offset_norm = test_model_msrc(setname=setname,
                    dataset=dataset,
                    source_name=source_name,
                    model_path=model_path,
                    batch_size = batch_size,
                    jnt_idx = jnt_idx,
                    num_enlarge=0,
                    patch_size=patch_size,
                    offset_depth_range=offset_depth_range,
                    h1_out_factor=h1_out_factor,
                    h2_out_factor=h2_out_factor,
                    prev_jnt_uvd_derot=prev_jnt_uvd_derot
            )
    else:
        uvd_offset_norm = test_model(setname=setname,
                    dataset=dataset,
                    source_name=source_name,
                    model_path=model_path,
                    batch_size = batch_size,
                    jnt_idx = jnt_idx,
                    num_enlarge=0,
                    patch_size=patch_size,
                    offset_depth_range=offset_depth_range,
                    h1_out_factor=h1_out_factor,
                    h2_out_factor=h2_out_factor,
                    prev_jnt_uvd_derot=prev_jnt_uvd_derot
            )

    xyz,err = uvd_to_xyz_error(setname=setname,uvd_pred_offset=uvd_offset_norm,rot=rot,
                           prev_jnt_uvd_derot=prev_jnt_uvd_derot,patch_size=patch_size,jnt_idx =jnt_idx,offset_depth_range=offset_depth_range,
                           uvd_gr=uvd_gr,xyz_true=xyz_true,
                           roixy=roixy,rect_d1d2w=rect_d1d2w,depth_dmin_dmax=depth_dmin_dmax,orig_pad_border=orig_pad_border)

    read_save_format.save(xyz_jnt_path[0],data=xyz[:,0,:],format='mat')
    read_save_format.save(xyz_jnt_path[1],data=xyz[:,1,:],format='mat')
    read_save_format.save(xyz_jnt_path[2],data=xyz[:,2,:],format='mat')
    read_save_format.save(xyz_jnt_path[3],data=xyz[:,3,:],format='mat')
    read_save_format.save(xyz_jnt_path[4],data=xyz[:,4,:],format='mat')
    read_save_format.save(xyz_jnt_path[5],data=xyz[:,5,:],format='mat')

    print 'jnt err for base wrist', jnt_idx, err
    print 'mean jnt err for base wrist',err.mean()


if __name__ == '__main__':
    """change the NUM_JNTS in src/constants.py to 6"""
    setname='nyu'
    xyz_jnt_save_path =   ['D:\\msrc_tmp\\jnt0_xyz.mat',
                           'D:\\msrc_tmp\\jnt1_xyz.mat',
                           'D:\\msrc_tmp\\jnt5_xyz.mat',
                           'D:\\msrc_tmp\\jnt9_xyz.mat',
                           'D:\\msrc_tmp\\jnt13_xyz.mat',
                           'D:\\msrc_tmp\\jnt17_xyz.mat']


    get_base_wrist_loc_err(setname,xyz_jnt_save_path)