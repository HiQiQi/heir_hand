__author__ = 'QiYE'
import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi_mid_uvd_normalized
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import update_params,get_gradients,update_params2,set_params
import time
import h5py
import numpy
import cv2
import scipy.io
from src.utils.err_uvd_xyz import uvd_to_xyz_error_single,xyz_to_uvd_derot
from src.utils import read_save_format
import matplotlib.pyplot as plt
import sys

def test_model(dataset,setname, source_name,prev_jnt_uvd_derot,batch_size,jnt_idx,patch_size,offset_depth_range,c1,c2,h1_out_factor,h2_out_factor,model_path):

    print 'offset_depth_range ',offset_depth_range
    num_enlarge=0
    model_info='mid%d_offset_r012_21jnts_derot_lg%d_patch%d'%(jnt_idx[0],num_enlarge,patch_size)
    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_mid_uvd_normalized(path,prev_jnt_uvd_derot,jnt_idx=jnt_idx,
                                                                                        patch_size=patch_size,patch_pad_width=4,hand_width=96,hand_pad_width=0,offset_depth_range=offset_depth_range)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches
    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')

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
                kernel_c01= 4,
                pool_c01= 2 ,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 1,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 2 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)
    cost = model.cost(Y)

    save_path =    '../../data/%s/hier_derot/mid/best/'%setname
    model_save_path = "%s%s.npy"%(save_path,model_path)
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output], on_unused_input='ignore')

    cost_nbatch = 0
    uvd_offset_norm = numpy.empty_like(test_set_y)
    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x0 = test_set_x0[slice_idx]
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
        uvd_offset_norm[slice_idx] = uvd_batch
        cost_nbatch+=cost_ij
    print 'cost', cost_nbatch/n_test_batches
    return uvd_offset_norm

def get_mid_loc_err(setname,file_format,prev_jnt_path,xyz_jnt_path):
    jnt_idx_all5 = [[2],[6],[10],[14],[18]]

    dataset='test'

    if setname =='msrc':
        source_name='_msrc_derot_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
        source_name_ori='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
        model_path=['param_cost_mid2_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep400',
                    'param_cost_mid6_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep600',
                    'param_cost_mid10_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep500',
                    'param_cost_mid14_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep450',
                    'param_cost_mid18_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep200'
                    ]
        batch_size = 100
        patch_size=40
        offset_depth_range=0.8
    else:
        if setname=='nyu':
            source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300'
            source_name_ori='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300'
            model_path=['param_cost_offset_mid2_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1770',
                        'param_cost_offset_mid6_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1430',
                        'param_cost_offset_mid10_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm300_yt0_ep1805',
                        'param_cost_offset_mid14_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm600_yt0_ep1290',
                        'param_cost_offset_mid18_r012_21jnts_derot_lg0_patch40_c0014_c0128_c1014_c1128_c2014_c2128_h12_h24_gm0_lm400_yt0_ep665'
                        ]
            batch_size = 100
            patch_size=40
            offset_depth_range=0.6
        else:
            if setname=='icvl':
                source_name='_msrc_derot_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
                source_name_ori='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300'
                model_path=['param_cost_mid2_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep400',
                            'param_cost_mid6_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep600',
                            'param_cost_mid10_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep500',
                            'param_cost_mid14_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep450',
                            'param_cost_mid18_offset_r012_21jnts_derot_lg0_patch40_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm499_yt5_ep200'
                            ]
                batch_size = 133
                patch_size=40
                offset_depth_range=0.8
            else:
                sys.exit('dataset name shoudle be icvl/nyu/msrc')

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
    err_all5 = []
    for i,jnt_idx in enumerate(jnt_idx_all5):
        prev_jnt_xyz=read_save_format.load(prev_jnt_path[i],format=file_format)
        prev_jnt_uvd_derot = xyz_to_uvd_derot(prev_jnt_xyz,setname='msrc',rot=rot,jnt_idx=jnt_idx,
                                              roixy=roixy,rect_d1d2w=rect_d1d2w,depth_dmin_dmax=depth_dmin_dmax,orig_pad_border=orig_pad_border)

        uvd_offset_norm = test_model(setname=setname,
                    dataset=dataset,
                    source_name=source_name,
                    model_path=model_path[i],
                    batch_size = batch_size,
                    jnt_idx = jnt_idx,
                    patch_size=patch_size,
                    offset_depth_range=offset_depth_range,
                    c1=16,c2=32,
                    h1_out_factor=2,
                    h2_out_factor=4,
                    prev_jnt_uvd_derot=prev_jnt_uvd_derot
            )
        xyz,err = uvd_to_xyz_error_single(setname=setname,uvd_pred_offset=uvd_offset_norm,rot=rot,
                               prev_jnt_uvd_derot=prev_jnt_uvd_derot,patch_size=patch_size,jnt_idx =jnt_idx,offset_depth_range=offset_depth_range,
                               uvd_gr=uvd_gr,xyz_true=xyz_true,
                               roixy=roixy,rect_d1d2w=rect_d1d2w,depth_dmin_dmax=depth_dmin_dmax,orig_pad_border=orig_pad_border)
        err_all5.append(err)
        read_save_format.save(xyz_jnt_path[i],data=xyz,format='mat')

    print 'jnt err for mid ', jnt_idx_all5, err_all5
    print 'mean jnt err for mid',numpy.array(err_all5).mean()


if __name__ == '__main__':
    """change the NUM_JNTS in src/constants.py to 1"""
    ''''change the path: xyz location of the palm center, file format can be npy or mat'''
    setname='msrc'
    file_format='mat'
    prev_jnt_path =['D:\\msrc_tmp\\jnt1_xyz.mat',
                  'D:\\msrc_tmp\\jnt5_xyz.mat',
                  'D:\\msrc_tmp\\jnt9_xyz.mat',
                  'D:\\msrc_tmp\\jnt13_xyz.mat',
                  'D:\\msrc_tmp\\jnt17_xyz.mat']

    xyz_jnt_save_path=['D:\\msrc_tmp\\jnt2_xyz.mat',
                  'D:\\msrc_tmp\\jnt6_xyz.mat',
                  'D:\\msrc_tmp\\jnt10_xyz.mat',
                  'D:\\msrc_tmp\\jnt14_xyz.mat',
                  'D:\\msrc_tmp\\jnt18_xyz.mat']



    get_mid_loc_err(setname,file_format,prev_jnt_path,xyz_jnt_save_path)
