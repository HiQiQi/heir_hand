__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy

from load_data import  load_data_multi_base_uvd_normalized, get_thresh_bw_jnts,load_norm_offset_uvd
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import set_params


def test_model(setname,dataset, source_name,prev_jnt_name,batch_size,jnt_idx,patch_size,offset_depth_range,num_enlarge,h1_out_factor,h2_out_factor,model_path,offset_save_path):
    print 'offset_depth_range ',offset_depth_range
    model_info='uvd_bw_r012_21jnts_derot_lg%d_patch%d'%(num_enlarge,patch_size)
    print model_info

    src_path = '../../data/%s/source/'%setname
    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    direct = '../../data/%s/final_xyz_uvd/'%setname
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,prev_jnt_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_base_uvd_normalized(path,prev_jnt_path,is_shuffle=False,
                                                                                         jnt_idx=jnt_idx,
                                                                                         patch_size=patch_size,patch_pad_width=4,
                                                                                         offset_depth_range=offset_depth_range,hand_width=96,hand_pad_width=0)
    path = '%strain%s.h5'%(src_path,source_name)

    direct = '../../data/%s/final_xyz_uvd/'%setname
    prev_jnt_path ='%strain%s.npy'%(direct,prev_jnt_name)
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

    save_path =    '../../data/%s/base_wrist_spacial/'%setname
    model_save_path = "%s%s.npy"%(save_path,model_path)
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,cost_1,cost_2,model.layers[-1].output ],on_unused_input='ignore')

    cost_nbatch = 0
    cost_loc_nbatch=0
    cost_rot_nbatch=0
    uvd_norm = numpy.empty_like(test_set_y)
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
        uvd_norm[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] = uvd_batch

    print 'test cost', cost_nbatch/n_test_batches,cost_loc_nbatch/n_test_batches,cost_rot_nbatch/n_test_batches


    numpy.save("%s%s%s.npy"%(save_path,dataset,offset_save_path),uvd_norm)

    # cost_nbatch = 0
    # cost_loc_nbatch=0
    # cost_rot_nbatch=0
    # uvd_norm = numpy.empty_like(train_set_y)
    # for minibatch_index in xrange(n_train_batches):
    #     # print minibatch_index
    #     x0 = train_set_x0[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    #     x1 = train_set_x1[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    #     x2 = train_set_x2[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    #     y = train_set_y[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    #
    #     cost_ij,cost_loc,cost_rot,uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
    #     cost_loc_nbatch+=cost_loc
    #     cost_rot_nbatch+=cost_rot
    #     cost_nbatch+=cost_ij
    #     uvd_norm[minibatch_index * batch_size: (minibatch_index + 1) * batch_size] = uvd_batch
    #
    # print 'train cost', cost_nbatch/n_train_batches,cost_loc_nbatch/n_train_batches,cost_rot_nbatch/n_train_batches
    # numpy.save("%strain%s.npy"%(save_path,offset_save_path),uvd_norm)

if __name__ == '__main__':
    # test_model(setname='icvl',
    #             dataset='test',
    #             source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
    #             prev_jnt_name='_absuvd0_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep1445',
    #             batch_size = 133,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             num_enlarge=0,
    #             patch_size=56,
    #             offset_depth_range=0.6,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             model_path='param_cost_uvd_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep745',
    #             offset_save_path='_uvd_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep745'
    #     )


    test_model(setname='msrc',
                dataset='test',
                source_name='_icvl_derot_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                prev_jnt_name='_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770',
                batch_size = 100,
                jnt_idx = [0,1,5,9 ,13,17],
                num_enlarge=0,
                patch_size=56,
                offset_depth_range=0.6,
                h1_out_factor=2,
                h2_out_factor=4,
                model_path='param_cost_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm399_yt0_ep770',
                offset_save_path='_uvd_bw_r012_21jnts_derot_lg0_patch56_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm9999_yt0_ep745'
        )

